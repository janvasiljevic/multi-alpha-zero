# Threeway Chess libraries

This repository contains (TS/JS/Wasm) libraries for the Threeway Chess game.

## Repository structure

- `wasm/` Typescript library using Web Assembly (Wasm) for the game logic. Can be used in the browser or in Node.js.
- `js/` Typescript library for the game logic. Same as `wasm/` but compiled using `wasm2js` tool so it can be used in EcmaScript environments that don't support Wasm (e.g. Node.js < 8.0 or React Native).
- `examples/server` NestJS server that uses the `wasm` library. Uses Yarn v4 and SWC.
- `examples/frontend` React web app that uses the `js` library. Uses Yarn v4 and Vite.

Examples don't show how to use the libraries, but how to set up a project that uses them. All the documentation is in the libraries themselves.

For other general documentation refer to `Guide.pdf`.

# Missing from this library ATM:

Algebraic notation for moves -> In the works
