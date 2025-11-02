/* eslint-disable no-console */
import { io, Socket } from "socket.io-client";

type TurnState = {
  matchId: string;
  currentCard: number | null;
  pot: number;
  you: {
    chips: number;
    cards: number[];
  };
};

const SERVER_URL = process.env.BOT_SERVER_URL || "http://localhost:3000";
const BOT_COUNT = Number.parseInt(process.env.BOT_COUNT || "3", 10);
const BASE_NAME = process.env.BOT_NAME || "SampleBot";
const BASE_SECRET =
  process.env.BOT_SECRET || `sample-secret-${Math.random().toString(36).slice(2, 10)}`;

interface SampleBot {
  name: string;
  socket: Socket;
  matchId: string | null;
  secret: string;
}

const bots: SampleBot[] = [];

for (let i = 0; i < BOT_COUNT; i += 1) {
  const name = `${BASE_NAME}-${i + 1}-${Math.random().toString(36).slice(2, 5)}`;
  const secret = BOT_COUNT > 1 ? `${BASE_SECRET}-${i + 1}` : BASE_SECRET;
  bots.push(spawnBot(name, secret));
}

function spawnBot(name: string, secret: string): SampleBot {
  const socket = io(`${SERVER_URL}/bots`, {
    transports: ["websocket", "polling"],
    reconnection: true,
    reconnectionDelay: 1000,
    reconnectionDelayMax: 4000,
  });

  const bot: SampleBot = {
    name,
    socket,
    matchId: null,
    secret,
  };

  socket.on("connect", () => {
    console.log(`[${name}] connected, registering…`);
    socket.emit("registerBot", { name, secret }, (ack?: { ok?: boolean; error?: string; rating?: number }) => {
      if (!ack?.ok) {
        console.error(`[${name}] registration failed: ${ack?.error || "unknown error"}`);
        return;
      }
      console.log(`[${name}] ready (rating ${ack.rating}).`);
      socket.emit("enqueue");
    });
  });

  socket.on("registered", (payload: { stats?: unknown }) => {
    if (payload?.stats) {
      console.log(`[${name}] stats`, payload.stats);
    }
  });

  socket.on("matchStarted", (state: { matchId?: string; players: Array<{ name: string }> }) => {
    bot.matchId = state.matchId ?? null;
    console.log(
      `[${name}] match started against ${state.players.map((player) => player.name).join(", ")}`,
    );
  });

  socket.on("turn", (state: TurnState) => {
    if (!state) {
      return;
    }
    bot.matchId = state.matchId;
    const decision = chooseAction(state);
    console.log(
      `[${name}] chooses ${decision} (card ${state.currentCard}, pot ${state.pot}, chips ${state.you.chips}).`,
    );
    socket.emit("botAction", { matchId: state.matchId, action: decision });
  });

  socket.on("matchUpdate", (state: { matchId?: string }) => {
    if (!state) {
      return;
    }
    bot.matchId = state.matchId ?? null;
  });

  socket.on("matchEnded", (summary: {
    standings: Array<{ name: string; totalScore?: number; botId: string }>;
    winners: string[];
  }) => {
    if (!summary) {
      return;
    }
    const placement =
      summary.standings.findIndex((entry) => entry.name === name) + 1;
    const score =
      summary.standings.find((entry) => entry.name === name)?.totalScore ?? "n/a";
    const win = summary.winners.includes(
      summary.standings.find((entry) => entry.name === name)?.botId ?? "",
    );
    console.log(
      `[${name}] match ended — place ${placement}/${summary.standings.length} (score ${score})${
        win ? " ✅" : ""
      }`,
    );
    bot.matchId = null;
  });

  socket.on("disconnect", (reason: string) => {
    console.log(`[${name}] disconnected: ${reason}`);
  });

  socket.on("error", (error: unknown) => {
    console.error(`[${name}] socket error:`, error);
  });

  return bot;
}

function chooseAction(state: TurnState): "pass" | "take" {
  if (!state || state.currentCard == null) {
    return "take";
  }

  if (state.you.chips <= 0) {
    return "take";
  }

  const minCard = Math.min(...(state.you.cards.length ? state.you.cards : [Infinity]));
  const potentialScore = state.currentCard - state.pot;
  if (potentialScore <= minCard) {
    return "take";
  }

  return Math.random() < 0.5 ? "pass" : "take";
}

process.on("SIGINT", () => {
  console.log("\nShutting down bots…");
  bots.forEach((bot) => {
    try {
      bot.socket.disconnect();
    } catch {
      // ignore
    }
  });
  setTimeout(() => process.exit(0), 200);
});
