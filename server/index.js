const path = require("path");
const express = require("express");
const http = require("http");
const { Server } = require("socket.io");

const PORT = process.env.PORT || 3000;
const CARDS = Array.from({ length: 33 }, (_, idx) => idx + 3);
const HIDDEN_CARDS = 9;
const CHIPS_PER_PLAYER = 11;
const MAX_NAME_LENGTH = 24;
const MAX_EVENTS = 50;

const app = express();
const server = http.createServer(app);
const io = new Server(server);

const rooms = new Map();

app.use(express.static(path.join(__dirname, "..", "public")));

app.get("/room/:roomId", (_req, res) => {
  res.sendFile(path.join(__dirname, "..", "public", "index.html"));
});

io.on("connection", (socket) => {
  socket.on("joinRoom", (payload = {}, ack) => {
    const { roomId, name, playerId } = payload;
    if (!roomId || typeof roomId !== "string") {
      emitError(socket, "Room id required.");
      return;
    }
    const trimmedName = `${name || ""}`.trim().slice(0, MAX_NAME_LENGTH) || "Player";

    const room = getOrCreateRoom(roomId);

    const existingByName = room.players.find((p) =>
      equalsIgnoreCase(p.name, trimmedName),
    );
    if (existingByName && existingByName.connected && existingByName.socketId !== socket.id) {
      emitError(socket, "That name is already seated. Try a different one.");
      return;
    }

    let player = existingByName || null;
    if (player) {
      player.name = trimmedName;
      player.connected = true;
      player.socketId = socket.id;
      addEvent(room, `${player.name} rejoined.`);
    } else {
      const id = generatePlayerId(room, trimmedName);
      player = {
        id,
        name: trimmedName,
        chips: 0,
        cards: [],
        connected: true,
        socketId: socket.id,
      };
      room.players.push(player);
      addEvent(room, `${player.name} joined the lobby.`);
    }

    if (!room.hostId || !room.players.some((p) => p.id === room.hostId)) {
      room.hostId = room.players[0]?.id || null;
    }

    socket.join(roomId);
    socket.data.roomId = roomId;
    socket.data.playerId = player.id;

    if (room.state !== "inProgress") {
      player.chips = CHIPS_PER_PLAYER;
      player.cards = [];
    } else if (room.turnIndex === -1) {
      const nextIndex = room.players.findIndex((entry) => entry.connected);
      if (nextIndex !== -1) {
        room.turnIndex = nextIndex;
      }
    }

    const state = sanitizeRoom(room);
    if (typeof ack === "function") {
      ack({ ok: true, playerId: player.id, roomId, state });
    }
    broadcastState(roomId);
  });

  socket.on("startGame", () => {
    const room = getRoomForSocket(socket);
    if (!room) {
      emitError(socket, "Join a room before starting a game.");
      return;
    }

    if (room.state === "inProgress") {
      emitError(socket, "Game already in progress.");
      return;
    }

    if (room.players.length < 2) {
      emitError(socket, "Need at least two players to start.");
      return;
    }

    const player = findPlayer(room, socket.data.playerId);
    if (!player || room.hostId !== player.id) {
      emitError(socket, "Only the host can start the game.");
      return;
    }

    startGame(room, player.id);
    broadcastState(room.id);
  });

  socket.on("playerAction", (payload = {}) => {
    const { action } = payload;
    const room = getRoomForSocket(socket);
    if (!room) {
      emitError(socket, "Join a room first.");
      return;
    }

    if (room.state !== "inProgress") {
      emitError(socket, "No active game.");
      return;
    }

    const playerIndex = room.players.findIndex((p) => p.id === socket.data.playerId);
    if (playerIndex === -1) {
      emitError(socket, "Player not found in this room.");
      return;
    }

    const player = room.players[playerIndex];
    if (player.socketId !== socket.id) {
      emitError(socket, "You are not the active connection for this player.");
      return;
    }

    if (playerIndex !== room.turnIndex) {
      emitError(socket, "It is not your turn.");
      return;
    }

    if (action === "pass") {
      const error = handlePass(room, player);
      if (error) {
        emitError(socket, error);
        return;
      }
    } else if (action === "take") {
      const error = handleTake(room, player);
      if (error) {
        emitError(socket, error);
        return;
      }
    } else {
      emitError(socket, "Unknown action.");
      return;
    }

    broadcastState(room.id);
  });

  socket.on("disconnect", () => {
    const room = getRoomForSocket(socket);
    if (!room) {
      return;
    }
    const playerIndex = room.players.findIndex((p) => p.id === socket.data.playerId);
    if (playerIndex === -1) {
      cleanupRoomIfEmpty(room.id);
      return;
    }

    const player = room.players[playerIndex];
    player.connected = false;
    player.socketId = null;

    if (room.state === "lobby") {
      room.players.splice(playerIndex, 1);
      addEvent(room, `${player.name} left the lobby.`);
      if (room.hostId === player.id) {
        room.hostId = room.players[0]?.id || null;
      }
    } else {
      addEvent(room, `${player.name} disconnected.`);
      if (room.turnIndex === playerIndex) {
        advanceTurn(room);
      }
    }

    cleanupRoomIfEmpty(room.id);
    broadcastState(room.id);
  });
});

server.listen(PORT, () => {
  // eslint-disable-next-line no-console
  console.log(`Server listening on http://localhost:${PORT}`);
});

function emitError(socket, message) {
  socket.emit("errorMessage", message);
}

function getOrCreateRoom(roomId) {
  if (!rooms.has(roomId)) {
    rooms.set(roomId, {
      id: roomId,
      state: "lobby",
      players: [],
      hostId: null,
      deck: [],
      removedCards: [],
      currentCard: null,
      pot: 0,
      turnIndex: -1,
      events: [],
    });
  }
  return rooms.get(roomId);
}

function getRoomForSocket(socket) {
  const roomId = socket.data.roomId;
  if (!roomId) {
    return null;
  }
  return rooms.get(roomId) || null;
}

function findPlayer(room, playerId) {
  return room.players.find((p) => p.id === playerId) || null;
}

function startGame(room, initiatorId) {
  room.deck = shuffle([...CARDS]);
  room.removedCards = room.deck.splice(0, HIDDEN_CARDS);
  room.currentCard = null;
  room.pot = 0;
  room.state = "inProgress";
  room.events = [];

  room.players.forEach((player) => {
    player.cards = [];
    player.chips = CHIPS_PER_PLAYER;
  });

  const firstIndex = Math.max(
    0,
    room.players.findIndex((p) => p.id === initiatorId),
  );
  room.turnIndex = firstIndex;

  drawNextCard(room);
  addEvent(room, "Game started.");
}

function handlePass(room, player) {
  if (!room.currentCard && room.currentCard !== 0) {
    return "No card to pass on.";
  }
  if (player.chips <= 0) {
    return "You have no chips left. You must take the card.";
  }

  player.chips -= 1;
  room.pot += 1;
  addEvent(room, `${player.name} said no thanks.`);
  advanceTurn(room);
  return null;
}

function handleTake(room, player) {
  if (!room.currentCard && room.currentCard !== 0) {
    return "No card to take.";
  }

  const playerIndex = room.players.findIndex((entry) => entry.id === player.id);
  player.cards.push(room.currentCard);
  if (room.pot > 0) {
    player.chips += room.pot;
  }
  addEvent(room, `${player.name} took card ${room.currentCard}.`);
  room.pot = 0;

  drawNextCard(room);
  if (room.state === "inProgress") {
    if (playerIndex !== -1 && room.players[playerIndex]?.connected) {
      room.turnIndex = playerIndex;
    } else {
      advanceTurn(room);
    }
  }
  return null;
}

function drawNextCard(room) {
  if (!room.deck.length) {
    finishGame(room);
    return;
  }
  room.currentCard = room.deck.shift();
}

function advanceTurn(room) {
  if (room.players.length === 0) {
    room.turnIndex = -1;
    return;
  }
  let nextIndex = room.turnIndex;
  for (let i = 0; i < room.players.length; i += 1) {
    nextIndex = (nextIndex + 1) % room.players.length;
    const candidate = room.players[nextIndex];
    if (candidate.connected) {
      room.turnIndex = nextIndex;
      return;
    }
  }
  room.turnIndex = -1;
}

function finishGame(room) {
  if (room.state === "finished") {
    return;
  }
  room.state = "finished";
  room.currentCard = null;
  room.turnIndex = -1;

  const scores = room.players.map((p) => ({
    id: p.id,
    score: calculateScore(p.cards, p.chips),
  }));
  const bestScore = Math.min(...scores.map((entry) => entry.score));
  const winners = scores.filter((entry) => entry.score === bestScore);
  const winnerNames = winners
    .map((entry) => findPlayer(room, entry.id)?.name || "Player")
    .join(", ");
  addEvent(room, `Game finished. Winner: ${winnerNames} (${bestScore}).`);
}

function addEvent(room, message) {
  room.events.push({
    timestamp: Date.now(),
    message,
  });
  if (room.events.length > MAX_EVENTS) {
    room.events.splice(0, room.events.length - MAX_EVENTS);
  }
}

function sanitizeRoom(room) {
  const players = room.players.map((player, index) => {
    const cards = [...player.cards].sort((a, b) => a - b);
    return {
      id: player.id,
      name: player.name,
      chips: player.chips,
      cards,
      score: calculateScore(cards, player.chips),
      connected: player.connected,
      isHost: player.id === room.hostId,
      isTurn: room.turnIndex === index && room.state === "inProgress",
    };
  });

  const winnerIds =
    room.state === "finished"
      ? computeWinnerIds(players)
      : [];

  return {
    roomId: room.id,
    state: room.state,
    players,
    pot: room.pot,
    currentCard: room.currentCard,
    deckCount: room.deck.length,
    removedCount: room.removedCards.length,
    hostId: room.hostId,
    winnerIds,
    events: room.events.slice(-15),
  };
}

function broadcastState(roomId) {
  const room = rooms.get(roomId);
  if (!room) {
    return;
  }
  io.to(roomId).emit("stateUpdate", sanitizeRoom(room));
}

function computeWinnerIds(players) {
  if (!players.length) {
    return [];
  }
  const lowest = Math.min(...players.map((player) => player.score));
  return players.filter((player) => player.score === lowest).map((player) => player.id);
}

function calculateScore(cards, chips) {
  if (!cards.length) {
    return -chips;
  }
  let total = 0;
  let previous = null;
  cards
    .slice()
    .sort((a, b) => a - b)
    .forEach((card) => {
      if (previous === null || card !== previous + 1) {
        total += card;
      }
      previous = card;
    });
  return total - chips;
}

function cleanupRoomIfEmpty(roomId) {
  const room = rooms.get(roomId);
  if (!room) {
    return;
  }
  if (room.players.length === 0) {
    rooms.delete(roomId);
  }
}

function shuffle(list) {
  const array = [...list];
  for (let i = array.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
  return array;
}

function equalsIgnoreCase(a, b) {
  if (typeof a !== "string" || typeof b !== "string") {
    return false;
  }
  return a.localeCompare(b, undefined, { sensitivity: "accent" }) === 0;
}

function generatePlayerId(room, name) {
  const base = name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "") || "player";

  let attempt = 0;
  while (attempt < 5) {
    const suffix = Math.random().toString(36).slice(2, 6);
    const candidate = `${base}-${suffix}`;
    const exists = room.players.some((player) => player.id === candidate);
    if (!exists) {
      return candidate;
    }
    attempt += 1;
  }
  return `${base}-${Date.now()}`;
}
