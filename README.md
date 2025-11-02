# No Thanks Online

This is a lightweight real-time implementation of the card game *No Thanks!* powered by Node.js, Express, and Socket.IO.

## Getting Started

```bash
npm install
npm run dev        # start the TypeScript server with live reload
```

For a production build:

```bash
npm run build
npm run start      # compiles + runs dist/server/index.js
```

By default the server listens on port `3000`. With the server running, open <http://localhost:3000/> in a browser.

## Creating & Joining Games

- Visit `/` and choose **Create Random Room** or enter a custom room code.
- Share the URL in the form `/room/<room-code>` with your friends so they can join the same table.
- Enter a display name and click **Join Game**. Rejoin later with the same name to reclaim your seat.

## Playing

- The lobby host (the first player in the room) can start the game once at least two players are present.
- Gameplay follows the classic rules: pay a chip to pass (`No Thanks`) or take the current card and any chips on it.
- Scores update live; when the deck is exhausted the game ends and the lowest total wins.

## Development Notes

- The codebase is written in TypeScript. Server, bots, and scripts live under `src/` and compile to `dist/`.
- `npm run dev` launches `src/server/index.ts` with `ts-node-dev`; `npm run build` emits JS into `dist/` and the browser bundles into `public/`.
- The browser UI sources live in `src/client/` and compile to ESM files served from `public/`.
- Styling is in `public/styles.css`; feel free to extend the UI there or in `src/client/client.ts`.

## Bot Arena API

Automate your own No Thanks bot via the Socket.IO namespace at `/bots`.

1. Connect using the Socket.IO client and emit `registerBot` with a secret key once:

   ```js
   import { io } from "socket.io-client";

   const socket = io("/bots");
   const secret = process.env.MY_BOT_SECRET || "my-super-secret-key";
   socket.emit("registerBot", { name: "MyBot", secret }, (ack) => {
     if (!ack.ok) {
       console.error("Registration failed:", ack.error);
     } else {
       console.log("Ready! rating =", ack.rating);
     }
   });
   ```

2. Wait for `turn` events. Each payload contains the current card, pot, deck information, and your bot’s cards/chips. Reply with `botAction`:

   ```js
   socket.on("turn", (state) => {
     const { matchId } = state;
     const action = Math.random() < 0.5 ? "pass" : "take";
     socket.emit("botAction", { matchId, action }); // action: "pass" or "take"
   });
   ```

3. After every update you’ll also receive `matchUpdate` snapshots. When a game ends, a `matchEnded` event includes standings and the winners.

Bots automatically queue into matches against other connected bots. The secret key is the stable identity used for Elo—use a value only you know so other users can't hijack your rating. Ratings are Elo-based and update after each game. View the live ladder at `/bots` or pull JSON stats from `/api/bots/ratings`.

### Included Bots

- `src/bots/exampleBot.ts` — deliberately simple heuristic-runner useful for smoke tests. Run with `npm run bot:example` (set `BOT_SECRET` if you want to reuse the same rating across sessions).
- `src/bots/smartBot.ts` — a stronger bot that weighs score deltas, evaluates whether opponents are likely to take a card, and simulates passing loops before choosing an action. Run with:

  ```bash
  BOT_SECRET=my-unique-secret BOT_NAME=Smartie BOT_COUNT=2 npm run bot:smart
  ```

- `npm run demo` builds the project, spins up the compiled server on a temporary port, and launches a trio of smart bots (override with `DEMO_BOT_SCRIPT=dist/bots/exampleBot.js` to swap implementations).
