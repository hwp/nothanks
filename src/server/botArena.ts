import { Application } from "express";
import { Namespace, Server, Socket } from "socket.io";
import {
  CARDS,
  CHIPS_PER_PLAYER,
  HIDDEN_CARDS,
  calculateScore,
  shuffle,
} from "./gameUtils";

type AckPayload =
  | { ok: true; botId: string; rating: number; stats: BotStatsSnapshot }
  | { ok: false; error: string };

type BotStatsSnapshot = {
  games: number;
  wins: number;
  losses: number;
  draws: number;
  rating: number;
  winRate: number;
};

type BotProfile = {
  id: string;
  secret: string;
  name: string;
  rating: number;
  games: number;
  wins: number;
  losses: number;
  draws: number;
  lastSeen: number;
};

type ActiveBot = {
  id: string;
  secret: string;
  name: string;
  socket: Socket | null;
  connected: boolean;
  profile: BotProfile;
  currentMatchId: string | null;
};

type MatchParticipant = {
  botId: string;
  name: string;
};

type StandingEntry = {
  botId: string;
  name: string;
  totalScore: number;
  cards: number[];
  chips: number;
};

type MatchSummary = {
  matchId: string;
  standings: StandingEntry[];
  winners: string[];
};

const DEFAULT_RATING = 1200;
const MIN_RATING = 100;
const MATCH_SIZE = 3;
const TURN_TIMEOUT_MS = 5000;
const ELO_K = 32;

export default class BotArena {
  private readonly namespace: Namespace;

  private readonly profilesBySecret = new Map<string, BotProfile>();

  private readonly profilesById = new Map<string, BotProfile>();

  readonly activeBots = new Map<string, ActiveBot>();

  private readonly matches = new Map<string, BotMatch>();

  private readonly matchByBotId = new Map<string, string>();

  private startNewSeasonLock = false;

  private seasonId = 0;

  constructor(io: Server, app?: Application) {
    this.namespace = io.of("/bots");
    this.namespace.on("connection", this.handleConnection.bind(this));

    if (app) {
      app.get("/api/bots/ratings", (_req, res) => {
        res.json(this.getLeaderboard());
      });
    }
  }

  private handleConnection(socket: Socket): void {
    socket.once("registerBot", (payload: { name?: string; secret?: string } = {}, ack?: (res: AckPayload) => void) => {
      this.registerBot(socket, payload, ack);
    });
    socket.on("botAction", (payload: { matchId?: string; action?: string } = {}) => {
      this.handleBotAction(socket, payload);
    });
    socket.on("enqueue", () => {
      // no op
    });
    socket.on("disconnect", () => {
      this.handleDisconnect(socket);
    });
  }

  private registerBot(
    socket: Socket,
    payload: { name?: string; secret?: string },
    ack?: (res: AckPayload) => void,
  ): void {
    const secret = typeof payload.secret === "string" ? payload.secret.trim() : "";
    if (!secret) {
      this.sendAck(ack, { ok: false, error: "Secret key is required." });
      socket.disconnect(true);
      return;
    }

    const name = (payload.name ?? "").trim().slice(0, 36) || "Bot";

    let profile = this.profilesBySecret.get(secret);
    if (profile) {
      const existingBot = this.activeBots.get(profile.id);
      if (existingBot && existingBot.socket && existingBot.socket.id !== socket.id) {
        this.sendAck(ack, { ok: false, error: "Bot with this secret is already connected." });
        socket.disconnect(true);
        return;
      }
      profile.name = name;
    } else {
      const id = this.generateBotId(name);
      profile = {
        id,
        secret,
        name,
        rating: DEFAULT_RATING,
        games: 0,
        wins: 0,
        losses: 0,
        draws: 0,
        lastSeen: Date.now(),
      };
      this.profilesBySecret.set(secret, profile);
      this.profilesById.set(id, profile);
    }

    const bot: ActiveBot = {
      id: profile.id,
      secret: profile.secret,
      name: profile.name,
      socket,
      connected: true,
      profile,
      currentMatchId: null,
    };
    this.activeBots.set(bot.id, bot);
    socket.data.botId = bot.id;
    socket.data.botSecret = profile.secret;
    profile.lastSeen = Date.now();

    this.sendAck(ack, {
      ok: true,
      botId: bot.id,
      rating: Math.round(profile.rating),
      stats: this.buildStats(profile),
    });
    socket.emit("registered", {
      botId: bot.id,
      rating: Math.round(profile.rating),
      stats: this.buildStats(profile),
    });

    const existingMatchId = this.matchByBotId.get(bot.id);
    if (existingMatchId) {
      const match = this.matches.get(existingMatchId);
      if (match) {
        match.reconnectBot(bot);
        return;
      }
      this.matchByBotId.delete(bot.id);
    }

    this.tryStartNewSeason();
  }

  private handleBotAction(socket: Socket, payload: { matchId?: string; action?: string }): void {
    const bot = this.getBotBySocket(socket);
    if (!bot) {
      return;
    }
    const { matchId, action } = payload;
    if (!matchId || typeof matchId !== "string") {
      return;
    }
    const match = this.matches.get(matchId);
    if (!match) {
      return;
    }
    match.receiveAction(bot.id, action);
  }

  private handleDisconnect(socket: Socket): void {
    const bot = this.getBotBySocket(socket);
    if (!bot) {
      return;
    }
    bot.connected = false;
    bot.socket = null;
    this.activeBots.delete(bot.id);

    const matchId = this.matchByBotId.get(bot.id);
    if (matchId) {
      const match = this.matches.get(matchId);
      match?.botDisconnected(bot.id);
    }
  }

  private tryStartNewSeason(): void {
    if (this.matches.size > 0) {
      // season still running
      return;
    }

    if (this.startNewSeasonLock) {
      return;
    }
    this.startNewSeasonLock = true;

    // check connection
    for (const [botId, bot] of this.activeBots.entries()) {
      if (!bot.connected) {
        this.activeBots.delete(botId);
      }
    }

    if (this.activeBots.size < MATCH_SIZE) {
      this.startNewSeasonLock = false;
      return;
    }

    // log
    console.log(`Starting season #${this.seasonId} with ${this.activeBots.size} bots`);
    this.seasonId += 1;

    // shuffle all active bots
    const bots = Array.from(this.activeBots.values());
    for (let i = bots.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [bots[i], bots[j]] = [bots[j], bots[i]];
    }

    // start matches
    for (let i = 0; i < bots.length; i += MATCH_SIZE) {
      const participants = bots.slice(i, i + MATCH_SIZE);
      if (participants.length === MATCH_SIZE) {
        this.startMatch(participants);
      }
    }

    this.startNewSeasonLock = false;
  }

  private startMatch(participants: ActiveBot[]): void {
    const match = new BotMatch(this, participants);
    this.matches.set(match.id, match);
    participants.forEach((bot) => {
      this.matchByBotId.set(bot.id, match.id);
    });
    match.start();
  }

  finishMatch(match: BotMatch): void {
    this.matches.delete(match.id);
    match.participants.forEach((participant) => {
      this.matchByBotId.delete(participant.botId);
    });

    if (this.matches.size === 0) {
      this.tryStartNewSeason();
    }
  }

  updateStatsFromMatch(summary: MatchSummary): void {
    summary.standings.forEach((entry) => {
      const profile = this.profilesById.get(entry.botId);
      if (!profile) {
        return;
      }
      profile.games += 1;
      profile.lastSeen = Date.now();
      if (summary.winners.includes(entry.botId)) {
        if (summary.winners.length > 1) {
          profile.draws += 1;
        } else {
          profile.wins += 1;
        }
      } else {
        profile.losses += 1;
      }
    });

    this.applyElo(summary.standings);
  }

  private applyElo(standings: StandingEntry[]): void {
    const deltas = new Map<string, number>();
    for (let i = 0; i < standings.length; i += 1) {
      for (let j = i + 1; j < standings.length; j += 1) {
        const first = standings[i];
        const second = standings[j];
        const profileA = this.profilesById.get(first.botId);
        const profileB = this.profilesById.get(second.botId);
        if (!profileA || !profileB) {
          continue;
        }
        const ratingA = profileA.rating;
        const ratingB = profileB.rating;
        const expectedA = 1 / (1 + 10 ** ((ratingB - ratingA) / 400));
        const expectedB = 1 - expectedA;

        let resultA = 0.5;
        if (first.totalScore < second.totalScore) {
          resultA = 1;
        } else if (first.totalScore > second.totalScore) {
          resultA = 0;
        }
        const resultB = 1 - resultA;

        const deltaA = ELO_K * (resultA - expectedA);
        const deltaB = ELO_K * (resultB - expectedB);

        deltas.set(first.botId, (deltas.get(first.botId) || 0) + deltaA);
        deltas.set(second.botId, (deltas.get(second.botId) || 0) + deltaB);
      }
    }

    deltas.forEach((delta, botId) => {
      const profile = this.profilesById.get(botId);
      if (!profile) {
        return;
      }
      profile.rating = Math.max(MIN_RATING, Math.round(profile.rating + delta));
    });
  }

  getLeaderboard(): Array<
    Pick<BotProfile, "id" | "name" | "games" | "wins" | "losses" | "draws" | "lastSeen"> & {
      rating: number;
      winRate: number;
    }
  > {
    const profiles = Array.from(this.profilesById.values());
    profiles.sort((a, b) => b.rating - a.rating || a.name.localeCompare(b.name));
    return profiles.map((profile) => ({
      id: profile.id,
      name: profile.name,
      rating: Math.round(profile.rating),
      games: profile.games,
      wins: profile.wins,
      losses: profile.losses,
      draws: profile.draws,
      winRate: profile.games ? profile.wins / profile.games : 0,
      lastSeen: profile.lastSeen,
    }));
  }

  private getBotBySocket(socket: Socket): ActiveBot | null {
    const botId = socket.data?.botId as string | undefined;
    if (!botId) {
      return null;
    }
    return this.activeBots.get(botId) ?? null;
  }

  private sendAck(ack: ((res: AckPayload) => void) | undefined, payload: AckPayload): void {
    if (typeof ack === "function") {
      ack(payload);
    }
  }

  private buildStats(profile: BotProfile): BotStatsSnapshot {
    return {
      games: profile.games,
      wins: profile.wins,
      losses: profile.losses,
      draws: profile.draws,
      rating: Math.round(profile.rating),
      winRate: profile.games ? profile.wins / profile.games : 0,
    };
  }

  private generateBotId(name: string): string {
    const slug =
      name
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/^-|-$/g, "")
        .slice(0, 32) || "bot";
    let suffix = Math.random().toString(36).slice(2, 6);
    let candidate = `${slug}-${suffix}`;
    while (this.profilesById.has(candidate)) {
      suffix = Math.random().toString(36).slice(2, 6);
      candidate = `${slug}-${suffix}`;
    }
    return candidate;
  }
}

class BotMatch {
  readonly id: string;

  readonly participants: MatchParticipant[];

  private deck: number[] = [];

  private removedCards: number[] = [];

  private currentCard: number | null = null;

  private pot = 0;

  private turnIndex = 0;

  private players: Array<{
    botId: string;
    name: string;
    chips: number;
    cards: number[];
    connected: boolean;
  }> = [];

  private turnTimer: NodeJS.Timeout | null = null;

  private history: Array<{ timestamp: number; message: string }> = [];

  constructor(
    private readonly arena: BotArena,
    participantBots: ActiveBot[],
  ) {
    this.participants = participantBots.map((bot) => ({
      botId: bot.id,
      name: bot.name,
    }));
    this.id = `match-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;

    participantBots.forEach((bot) => {
      bot.currentMatchId = this.id;
    });
  }

  start(): void {
    this.deck = shuffle([...CARDS]);
    this.removedCards = this.deck.splice(0, HIDDEN_CARDS);
    this.currentCard = this.deck.shift() ?? null;
    this.pot = 0;
    this.turnIndex = 0;
    this.players = this.participants.map((participant) => ({
      botId: participant.botId,
      name: participant.name,
      chips: CHIPS_PER_PLAYER,
      cards: [],
      connected: true,
    }));

    this.broadcast("matchStarted", this.buildPublicState());
    if (this.currentCard !== null) {
      this.promptPlayer();
    } else {
      this.finish();
    }
  }

  reconnectBot(bot: ActiveBot): void {
    const participant = this.participants.find((entry) => entry.botId === bot.id);
    if (!participant) {
      return;
    }
    this.sendToBot(bot.id, "matchResumed", {
      matchId: this.id,
      state: this.buildBotView(bot.id),
    });
    if (this.players[this.turnIndex]?.botId === bot.id) {
      this.promptPlayer();
    }
  }

  receiveAction(botId: string, action?: string): void {
    const current = this.players[this.turnIndex];
    if (!current || current.botId !== botId) {
      return;
    }
    const normalized = action === "pass" ? "pass" : "take";
    if (this.turnTimer) {
      clearTimeout(this.turnTimer);
      this.turnTimer = null;
    }
    if (normalized === "pass" && current.chips <= 0) {
      this.applyTake(current);
      this.advanceAfterTake(current);
      return;
    }
    if (normalized === "pass") {
      this.applyPass(current);
      this.advanceTurn();
    } else {
      this.applyTake(current);
      this.advanceAfterTake(current);
    }
  }

  botDisconnected(botId: string): void {
    const player = this.players.find((entry) => entry.botId === botId);
    if (player) {
      player.connected = false;
    }
    if (this.players[this.turnIndex]?.botId === botId) {
      this.receiveAction(botId, "take");
    }
  }

  private applyPass(player: { chips: number; name: string }): void {
    if (player.chips > 0) {
      player.chips -= 1;
      this.pot += 1;
      this.log(`${player.name} passed.`);
    } else {
      this.log(`${player.name} tried to pass with no chips.`);
    }
  }

  private applyTake(player: { cards: number[]; name: string; chips: number }): void {
    if (this.currentCard === null) {
      return;
    }
    player.cards.push(this.currentCard);
    if (this.pot > 0) {
      player.chips += this.pot;
    }
    this.log(
      `${player.name} took ${this.currentCard}${
        this.pot ? ` and ${this.pot} chips` : ""
      }.`,
    );
    this.pot = 0;
    this.currentCard = this.deck.shift() ?? null;
  }

  private advanceTurn(): void {
    if (this.players.length === 0) {
      return;
    }
    for (let i = 0; i < this.players.length; i += 1) {
      this.turnIndex = (this.turnIndex + 1) % this.players.length;
      if (this.players[this.turnIndex].connected) {
        break;
      }
    }
    this.broadcast("matchUpdate", this.buildPublicState());
    if (this.currentCard === null) {
      this.finish();
      return;
    }
    this.promptPlayer();
  }

  private advanceAfterTake(player: { botId: string; connected: boolean }) {
    this.broadcast("matchUpdate", this.buildPublicState());
    if (this.currentCard === null) {
      this.finish();
      return;
    }
    const current = this.players[this.turnIndex];
    if (current && current.connected && current.botId === player.botId) {
      this.promptPlayer();
    } else {
      this.advanceTurn();
    }
  }

  private promptPlayer(): void {
    if (this.turnTimer) {
      clearTimeout(this.turnTimer);
    }
    const current = this.players[this.turnIndex];
    if (!current) {
      return;
    }
    if (!current.connected) {
      this.advanceTurn();
      return;
    }
    this.sendToBot(current.botId, "turn", this.buildBotView(current.botId));
    this.turnTimer = setTimeout(() => {
      this.turnTimer = null;
      const action = this.pickFallbackAction(current);
      this.receiveAction(current.botId, action);
    }, TURN_TIMEOUT_MS);
  }

  private pickFallbackAction(player: { chips: number }): string {
    if (player.chips <= 0) {
      return "take";
    }
    return Math.random() < 0.5 ? "pass" : "take";
  }

  private buildBotView(botId: string) {
    const player = this.players.find((entry) => entry.botId === botId);
    return {
      matchId: this.id,
      you: {
        name: player?.name ?? "",
        chips: player?.chips ?? 0,
        cards: (player?.cards ?? []).slice().sort((a, b) => a - b),
      },
      currentCard: this.currentCard,
      pot: this.pot,
      deckCount: this.deck.length,
      removedCount: this.removedCards.length,
      players: this.players.map((entry, index) => ({
        botId: entry.botId,
        name: entry.name,
        chips: entry.chips,
        cards: entry.cards.slice().sort((a, b) => a - b),
        isTurn: index === this.turnIndex,
        connected: entry.connected,
      })),
      history: this.history.slice(-5),
    };
  }

  private buildPublicState() {
    return {
      matchId: this.id,
      currentCard: this.currentCard,
      pot: this.pot,
      deckCount: this.deck.length,
      removedCount: this.removedCards.length,
      players: this.players.map((entry, index) => ({
        botId: entry.botId,
        name: entry.name,
        chips: entry.chips,
        cards: entry.cards.slice().sort((a, b) => a - b),
        isTurn: index === this.turnIndex,
        connected: entry.connected,
      })),
      history: this.history.slice(-10),
    };
  }

  private finish(): void {
    if (this.turnTimer) {
      clearTimeout(this.turnTimer);
      this.turnTimer = null;
    }
    const standings: StandingEntry[] = this.players.map((entry) => ({
      botId: entry.botId,
      name: entry.name,
      totalScore: calculateScore(entry.cards, entry.chips),
      cards: entry.cards.slice().sort((a, b) => a - b),
      chips: entry.chips,
    }));
    standings.sort((a, b) => a.totalScore - b.totalScore);
    const bestScore = standings[0]?.totalScore ?? 0;
    const winners = standings
      .filter((entry) => entry.totalScore === bestScore)
      .map((entry) => entry.botId);

    this.broadcast("matchEnded", {
      matchId: this.id,
      standings,
      winners,
    });

    standings.forEach((entry) => {
      const bot = this.arena.activeBots.get(entry.botId);
      if (bot) {
        bot.currentMatchId = null;
      }
    });

    this.arena.updateStatsFromMatch({
      matchId: this.id,
      standings,
      winners,
    });
    this.arena.finishMatch(this);
  }

  private broadcast(event: string, payload: unknown): void {
    this.participants.forEach((participant) => {
      this.sendToBot(participant.botId, event, payload);
    });
  }

  private sendToBot(botId: string, event: string, payload: unknown): void {
    const bot = this.arena.activeBots.get(botId);
    if (!bot || !bot.connected || !bot.socket) {
      return;
    }
    bot.socket.emit(event, payload);
  }

  private log(message: string): void {
    this.history.push({
      timestamp: Date.now(),
      message,
    });
    if (this.history.length > 50) {
      this.history.splice(0, this.history.length - 50);
    }
  }
}
