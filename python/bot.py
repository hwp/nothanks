import socketio
import random
import logging

logger = logging.getLogger(__name__)

class PlayerState:
    def __init__(self, data):
        self.name= data["name"]
        self.cards = data["cards"]
        self.chips= data["chips"]


class TurnState:
    def __init__(self, data, bot_id):
        self.matchId = data["matchId"]
        self.current = data["currentCard"]
        self.pot = data["pot"]
        self.n_players = len(data["players"])

        player_seq_id = next(i for i, p in enumerate(data["players"]) if p["botId"] == bot_id)
        self.you = PlayerState(data["players"][player_seq_id])
        self.others = [
            PlayerState(data["players"][(player_seq_id + i) % self.n_players])
            for i in range(1, self.n_players)
        ]


class Bot:
    def __init__(self, name, server_url, namespace, p_pass):
        self.name = name
        self.server_url = server_url
        self.namespace = namespace
        self.match_states = {}
        self.p_pass = p_pass

        self.sio = socketio.Client(
            reconnection=True,
            reconnection_delay=1,
            reconnection_delay_max=4,
        )
        self.sio.on("registered", self.on_registered, namespace=self.namespace)
        self.sio.on("matchStarted", self.on_match_started, namespace=self.namespace)
        self.sio.on("turn", self.on_turn, namespace=self.namespace)
        self.sio.on("matchUpdate", self.on_match_update, namespace=self.namespace)
        self.sio.on("matchEnded", self.on_match_ended, namespace=self.namespace)
        self.sio.on("disconnect", self.on_disconnect, namespace=self.namespace)
        self.sio.on("connect_error", self.on_connect_error, namespace=self.namespace)

    def on_registered(self, msg):
        self.bot_id = msg["botId"]
        self.sio.emit("enqueue", namespace=self.namespace)

    def on_match_started(self, msg):
        match_id = msg.get("matchId")
        self.match_states[match_id] = self.init_match()

    def on_turn(self, msg):
        match_id = msg.get("matchId")
        turn_state = TurnState(msg, self.bot_id)
        match_state = self.match_states[match_id]

        decision = self.choose_action(turn_state, match_state)
        logger.debug(f"[{self.name}] chooses {decision} "
              f"(card {turn_state.current}, pot {turn_state.pot}, "
              f"chips {turn_state.you.chips}).")

        self.sio.emit("botAction",
                      {"matchId": match_id, "action": decision},
                      namespace=self.namespace)

    def on_match_update(self, msg):
        # no need to react to match_update
        pass

    def on_match_ended(self, msg):
        match_id = msg.get("matchId")
        winners = msg["winners"]

        if self.bot_id in winners:
            if len(winners) > 1:
                result = "draw"
            else:
                result = "win"
        else:
            result = "lose"

        n_players = len(msg["standings"])
        player_seq_id = next( i for i, p in enumerate(msg["standings"])
                                if p["botId"] == self.bot_id)
        score = msg["standings"][player_seq_id]["totalScore"]
        others = [
            msg["standings"][(player_seq_id + i) % n_players]["totalScore"]
            for i in range(1, n_players)
        ]

        logger.info(f"[{self.name}] match ended with {result} ({score=}, {others=})")

        self.match_end_feedback(self.match_states[match_id], result, score, others)
        del self.match_states[match_id]

    def on_disconnect(self, msg):
        logger.info(f"[{self.name}] disconnected: {msg}")

    def on_connect_error(self, msg):
        logger.info(f"[{self.name}] connection error: {msg}")

    def connect(self):
        self.sio.connect(f"{self.server_url}",  namespaces=[self.namespace])
        self.sio.emit("registerBot", {"name": self.name}, namespace=self.namespace)

    def disconnect(self):
        self.sio.disconnect()

    def init_match(self):
        # no match memory needed for dumb strategy
        pass

    def choose_action(self, turn_state, match_state) -> str:
        if turn_state.you.chips <= 0:
            return "take"

        if turn_state.current - turn_state.pot <= 0:
            return "take"

        return "pass" if random.random() < self.p_pass else "take"

    def match_end_feedback(self, match_state, result, score, others):
        # no feedback is need for the dumb strategy
        pass


