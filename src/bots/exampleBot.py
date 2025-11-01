import os
import sys
import random
import string
import time
import socketio
import signal

# ==== CONFIG ====
_NAMESPACE = '/bots'
SERVER_URL = os.getenv("BOT_SERVER_URL", "http://localhost:3000")
BOT_COUNT = int(os.getenv("BOT_COUNT", "3"))
BASE_NAME = os.getenv("BOT_NAME", "SamplePythonBot")

# ==== TYPES ====
class TurnState:
    def __init__(self, data):
        self.matchId = data.get("matchId")
        self.currentCard = data.get("currentCard")
        self.pot = data.get("pot", 0)
        self.you = data.get("you", {"chips": 0, "cards": []})

# ==== GLOBAL ====
bots = []

# ==== LOGIC ====
def choose_action(state: TurnState) -> str:
    if state.currentCard is None:
        return "take"
    if state.you.get("chips", 0) <= 0:
        return "take"

    cards = state.you.get("cards", [])
    min_card = min(cards) if cards else float("inf")
    potential_score = state.currentCard - state.pot
    if potential_score <= min_card:
        return "take"

    return "pass" if random.random() < 0.5 else "take"


def spawn_bot(name: str):
    sio = socketio.Client(
        reconnection=True,
        reconnection_delay=1,
        reconnection_delay_max=4,
    )

    bot = {"name": name, "socket": sio, "matchId": None}

    @sio.on("registered", namespace=_NAMESPACE)
    def on_registered(payload):
        stats = payload.get("stats")
        if stats:
            print(f"[{name}] stats {stats}")
        sio.emit("enqueue", namespace=_NAMESPACE)

    @sio.on("matchStarted", namespace=_NAMESPACE)
    def on_match_started(state):
        bot["matchId"] = state.get("matchId")
        players = ", ".join(p["name"] for p in state.get("players", []))
        print(f"[{name}] match started against {players}")

    @sio.on("turn", namespace=_NAMESPACE)
    def on_turn(data):
        if not data:
            return
        state = TurnState(data)
        bot["matchId"] = state.matchId
        decision = choose_action(state)
        print(f"[{name}] chooses {decision} (card {state.currentCard}, pot {state.pot}, chips {state.you.get('chips')}).")
        sio.emit("botAction", {"matchId": state.matchId, "action": decision}, namespace=_NAMESPACE)

    @sio.on("matchUpdate", namespace=_NAMESPACE)
    def on_match_update(state):
        if not state:
            return
        bot["matchId"] = state.get("matchId")

    @sio.on("matchEnded", namespace=_NAMESPACE)
    def on_match_ended(summary):
        if not summary:
            return
        standings = summary.get("standings", [])
        winners = summary.get("winners", [])
        placement = next((i + 1 for i, e in enumerate(standings) if e["name"] == name), None)
        score = next((e.get("totalScore", "n/a") for e in standings if e["name"] == name), "n/a")
        bot_id = next((e["botId"] for e in standings if e["name"] == name), "")
        win = bot_id in winners
        print(f"[{name}] match ended — place {placement}/{len(standings)} (score {score}){' ✅' if win else ''}")
        bot["matchId"] = None

    @sio.event(namespace=_NAMESPACE)
    def disconnect(reason):
        print(f"[{name}] disconnected: {reason}")

    @sio.event(namespace=_NAMESPACE)
    def connect_error(data):
        print(f"[{name}] connection error: {data}")

    sio.connect(f"{SERVER_URL}",  namespaces=["/bots"])
    sio.emit("registerBot", {"name": name}, namespace=_NAMESPACE)
    return bot


# ==== MAIN ====
def shutdown(signal_received=None, frame=None):
    print("\nShutting down bots…")
    for bot in bots:
        try:
            bot["socket"].disconnect()
        except Exception:
            pass
    time.sleep(0.2)
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdown)

    for i in range(BOT_COUNT):
        suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=3))
        name = f"{BASE_NAME}-{i+1}-{suffix}"
        bots.append(spawn_bot(name))

    while True:
        time.sleep(1)
