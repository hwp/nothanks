/* eslint-disable no-console */
import { io, Socket } from "socket.io-client";
import { calculateScore } from "../server/gameUtils";

type MatchPlayer = {
  botId: string;
  name: string;
  chips: number;
  cards: number[];
  connected: boolean;
  isTurn: boolean;
};

type MatchState = {
  matchId: string;
  currentCard: number | null;
  pot: number;
  deckCount: number;
  removedCount: number;
  players: MatchPlayer[];
  history: Array<{ timestamp: number; message: string }>;
};

type TurnPayload = MatchState & {
  you: {
    name: string;
    chips: number;
    cards: number[];
  };
};

type MatchSummary = {
  matchId: string;
  standings: Array<{
    botId: string;
    name: string;
    totalScore: number;
    cards: number[];
    chips: number;
  }>;
  winners: string[];
};

type MatchStartedPayload = {
  matchId?: string;
  players: Array<{ name: string }>;
};

type RegisteredPayload = {
  stats?: {
    games: number;
    wins: number;
    losses: number;
    draws: number;
    rating: number;
    winRate: number;
  };
};

type RegisterAck =
  | { ok: true; rating?: number }
  | { ok?: false; error?: string };

const SERVER_URL = process.env.BOT_SERVER_URL || "http://localhost:3000";
const BOT_NAME = process.env.BOT_NAME || "SmartBot";
const BOT_COUNT = Number.parseInt(process.env.BOT_COUNT || "1", 10);
const BASE_SECRET =
  process.env.BOT_SECRET || `smart-secret-${Math.random().toString(36).slice(2, 10)}`;

interface SmartBotInstance {
  name: string;
  socket: Socket;
  currentMatchId: string | null;
  seenCards: Set<number>;
  playerSnapshot: Map<
    string,
    {
      name: string;
      cards: number[];
      chips: number;
      connected: boolean;
      isTurn: boolean;
    }
  >;
  secret: string;
}

const bots: SmartBotInstance[] = [];

for (let index = 0; index < BOT_COUNT; index += 1) {
  const suffix = BOT_COUNT > 1 ? `-${index + 1}` : "";
  const secret = BOT_COUNT > 1 ? `${BASE_SECRET}-${index + 1}` : BASE_SECRET;
  bots.push(createBot(`${BOT_NAME}${suffix}`, secret));
}

function createBot(name: string, secret: string): SmartBotInstance {
  const socket = io(`${SERVER_URL}/bots`, {
    transports: ["websocket", "polling"],
    reconnection: true,
    reconnectionDelay: 1000,
    reconnectionDelayMax: 4000,
  });

  const bot: SmartBotInstance = {
    name,
    socket,
    currentMatchId: null,
    seenCards: new Set<number>(),
    playerSnapshot: new Map(),
    secret,
  };

  socket.on("connect", () => {
    console.log(`[${name}] Connected. Registering…`);
    socket.emit("registerBot", { name, secret }, (ack: RegisterAck | undefined) => {
      if (!ack?.ok) {
        console.error(`[${name}] Registration failed: ${ack?.error ?? "unknown"}`);
        return;
      }
      console.log(`[${name}] Registered with rating ${ack.rating ?? "n/a"}. Ready to queue.`);
      socket.emit("enqueue");
    });
  });

  socket.on("registered", (payload: RegisteredPayload) => {
    if (payload?.stats) {
      console.log(`[${name}] Current stats ->`, payload.stats);
    }
  });

  socket.on("matchStarted", (state: MatchStartedPayload) => {
    handleMatchStarted(bot, state);
  });

  socket.on("matchUpdate", (state: MatchState) => {
    handleMatchUpdate(bot, state);
  });

  socket.on("turn", (state: TurnPayload) => {
    handleTurn(bot, state);
  });

  socket.on("matchEnded", (summary: MatchSummary) => {
    handleMatchEnded(bot, summary);
  });

  socket.on("disconnect", (reason: string) => {
    console.log(`[${name}] Disconnected: ${reason}`);
  });

  socket.on("error", (error: unknown) => {
    console.error(`[${name}] Socket error:`, error);
  });

  return bot;
}

function handleMatchStarted(bot: SmartBotInstance, state: MatchStartedPayload): void {
  bot.currentMatchId = state?.matchId ?? null;
  resetMatchMemory(bot);
  ingestPlayers(bot, state?.players ?? []);
  console.log(
    `[${bot.name}] Match started vs ${state.players
      .map((player) => player.name)
      .join(", ")}`,
  );
}

function handleMatchUpdate(bot: SmartBotInstance, state: MatchState): void {
  if (!state) {
    return;
  }
  bot.currentMatchId = state.matchId;
  ingestPlayers(bot, state.players ?? []);
}

function handleTurn(bot: SmartBotInstance, state: TurnPayload): void {
  if (!state || state.matchId !== bot.currentMatchId) {
    return;
  }
  ingestPlayers(bot, state.players ?? []);

  const action = chooseAction(bot, state);
  bot.socket.emit("botAction", { matchId: state.matchId, action });
}

function handleMatchEnded(bot: SmartBotInstance, summary: MatchSummary): void {
  if (!summary) {
    return;
  }
  const me = summary.standings.find((entry) => entry.name === bot.name);
  const place =
    summary.standings.findIndex((entry) => entry.name === bot.name) + 1;
  const info =
    me && typeof me.totalScore === "number"
      ? `score ${me.totalScore}`
      : "score n/a";
  const didWin = me ? summary.winners.includes(me.botId) : false;
  console.log(
    `[${bot.name}] Match ended — place ${place}/${summary.standings.length} (${info})${
      didWin ? " ✅" : ""
    }`,
  );
  bot.currentMatchId = null;
  setTimeout(() => {
    bot.socket.emit("enqueue");
  }, 500);
}

function resetMatchMemory(bot: SmartBotInstance): void {
  bot.seenCards.clear();
  bot.playerSnapshot.clear();
}

function ingestPlayers(
  bot: SmartBotInstance,
  players: Array<{ botId?: string; name: string; cards?: number[]; chips?: number; connected?: boolean; isTurn?: boolean }>,
): void {
  players.forEach((player) => {
    if (!player.botId) {
      return;
    }
    bot.playerSnapshot.set(player.botId, {
      name: player.name,
      cards: [...(player.cards ?? [])],
      chips: player.chips ?? 0,
      connected: player.connected !== false,
      isTurn: Boolean(player.isTurn),
    });
    (player.cards ?? []).forEach((card) => bot.seenCards.add(card));
  });
}

function chooseAction(bot: SmartBotInstance, state: TurnPayload): "pass" | "take" {
  const you = state.players.find((player) => player.name === bot.name);
  if (!you) {
    return "take";
  }
  const currentCard = state.currentCard;
  if (currentCard == null) {
    return "take";
  }
  const pot = state.pot ?? 0;
  const chips = you.chips ?? 0;
  const cards = [...(you.cards ?? [])].sort((a, b) => a - b);

  if (chips <= 0) {
    return "take";
  }

  const currentScore = calculateScore(cards, chips);
  const scoreIfTakeNow = calculateScore([...cards, currentCard], chips + pot);
  const deltaTake = scoreIfTakeNow - currentScore;

  const scoreIfPassImmediate = calculateScore(cards, chips - 1);
  const deltaPassImmediate = scoreIfPassImmediate - currentScore;

  const passOutcome = simulatePass(bot, state, you, currentCard);
  let deltaPass: number;
  if (passOutcome.takenByOther) {
    deltaPass = deltaPassImmediate;
  } else {
    const futureChips = chips - 1;
    const futurePot = pot + passOutcome.passes;
    const futureScore = calculateScore([...cards, currentCard], futureChips + futurePot);
    deltaPass = futureScore - currentScore + passOutcome.passes * 0.2;
  }

  if (deltaTake <= deltaPass - 0.5) {
    return "take";
  }
  if (deltaPass <= deltaTake - 0.5) {
    return "pass";
  }

  const runPotential = countRunNeighbors(cards, currentCard);
  if (runPotential >= 2 && deltaTake <= deltaPass + 1) {
    return "take";
  }

  if (pot >= 3 && deltaTake <= deltaPass + 2) {
    return "take";
  }

  if (chips <= 2 && deltaTake <= deltaPass + 1.5) {
    return "take";
  }

  if (cards.length === 0 && currentCard <= 16 && pot >= 1) {
    return "take";
  }

  if (passOutcome.forcedSelf && pot < 3 && deltaPass < deltaTake + 0.5) {
    return "pass";
  }

  if (deltaPass <= deltaTake) {
    return "pass";
  }
  return "take";
}

function simulatePass(
  bot: SmartBotInstance,
  state: MatchState,
  you: MatchPlayer,
  card: number,
): { takenByOther: boolean; passes: number; forcedSelf: boolean } {
  const players = state.players ?? [];
  const pot = state.pot ?? 0;
  const yourIndex = players.findIndex((player) => player.name === bot.name);
  if (yourIndex === -1) {
    return { takenByOther: false, passes: 1, forcedSelf: true };
  }
  let passingPot = pot + 1;
  let passes = 1;

  for (let offset = 1; offset < players.length; offset += 1) {
    const idx = (yourIndex + offset) % players.length;
    const player = players[idx];
    if (!player) {
      continue;
    }
    const snapshot = bot.playerSnapshot.get(player.botId);
    const chips = snapshot?.chips ?? player.chips ?? 0;
    const otherCards = [...(snapshot?.cards ?? player.cards ?? [])];

    if (chips <= 0) {
      return { takenByOther: true, passes, forcedSelf: false };
    }

    const currentScore = calculateScore(otherCards, chips);
    const scoreIfTake = calculateScore([...otherCards, card], chips + passingPot);
    const scoreIfPass = calculateScore(otherCards, chips - 1);
    const deltaTake = scoreIfTake - currentScore;
    const deltaPass = scoreIfPass - currentScore;

    if (deltaTake <= deltaPass - 0.25 || deltaTake <= 0) {
      return { takenByOther: true, passes, forcedSelf: false };
    }

    passingPot += 1;
    passes += 1;
  }

  return { takenByOther: false, passes, forcedSelf: true };
}

function countRunNeighbors(cards: number[], card: number): number {
  let count = 0;
  if (cards.includes(card - 1)) {
    count += 1;
  }
  if (cards.includes(card + 1)) {
    count += 1;
  }
  return count;
}

process.on("SIGINT", () => {
  console.log("\nStopping smart bots…");
  bots.forEach((bot) => {
    try {
      bot.socket.disconnect();
    } catch {
      // ignore cleanup errors
    }
  });
  setTimeout(() => process.exit(0), 200);
});
