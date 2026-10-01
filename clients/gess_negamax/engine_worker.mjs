// One single-threaded instance of the Gess web app's negamax engine (the Rust
// search in ~/calr3gh/gess/wasm-engine, built to WebAssembly by its build.sh).
// Reads one JSON request per line on stdin, {board, to_move, quiet_moves, ms},
// and writes one JSON line per request: {from, to, depth, score, ms} or {error}.
// Board, cells and sides are tdgauntlet's gess/1 encoding (games/gess.md); the
// engine indexes cells the same way but codes stones 1 = white, 2 = black.
import { readFileSync } from 'node:fs';
import { createInterface } from 'node:readline';

const wasmPath = process.argv[2];
const memory = new WebAssembly.Memory({ initial: 512, maximum: 1024, shared: true });
const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), {
  env: { memory, now: () => Date.now(), random: () => Math.random() },
});
const ex = instance.exports;

for await (const line of createInterface({ input: process.stdin })) {
  try {
    const req = JSON.parse(line);
    const bytes = new Uint8Array(400);
    for (let i = 0; i < 400; i++) bytes[i] = req.board[i] === 'W' ? 1 : req.board[i] === 'B' ? 2 : 0;
    new Uint8Array(memory.buffer, ex.board_ptr(), 400).set(bytes);
    ex.set_quiet_moves(req.quiet_moves);
    const t0 = Date.now();
    // best_move(white, budget ms, transposition table on, thread 0, new TT generation)
    const found = ex.best_move(req.to_move === 1 ? 1 : 0, req.ms, 1, 0, 1);
    if (!found) {
      process.stdout.write(JSON.stringify({ error: 'no move found' }) + '\n');
      continue;
    }
    const out = new Int32Array(memory.buffer, ex.out_ptr(), 8);
    process.stdout.write(JSON.stringify({
      from: out[1] * 20 + out[2], to: out[3] * 20 + out[4], depth: out[5], score: out[6], ms: Date.now() - t0,
    }) + '\n');
  } catch (e) {
    process.stdout.write(JSON.stringify({ error: String(e) }) + '\n');
  }
}
