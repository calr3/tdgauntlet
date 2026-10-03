// Reference outputs from the Gess web app's own engine (its WebAssembly build), for
// tests/identity.rs: the port must reproduce them exactly.
//
//   ~/node22/bin/node clients/gess_negamax_rs/tests/make_identity_fixture.mjs \
//       results/gess_e24_negamax.json clients/gess_negamax_rs/tests/data/identity.json.gz
//
// Positions are sampled from a tdgauntlet result file's games. For each: the root moves
// (`gen_root`: cells, outcome, key), every root move's score at depths 1-3 on the exact TT-off
// path (`search_root`), and the best score at depths 1-4 on the production TT-on path
// (`search_to_depth`). `random()` is never called by these entry points.
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import { gzipSync } from 'node:zlib';
import { execFileSync } from 'node:child_process';

const [resultsPath, outPath] = process.argv.slice(2);
const wasmPath = `${process.env.HOME}/calr3gh/gess/wasm-engine/target/wasm32-unknown-unknown/release/gess_engine_wasm.wasm`;
const memory = new WebAssembly.Memory({ initial: 512, maximum: 1024, shared: true });
const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), {
  env: { memory, now: () => Date.now(), random: () => { throw new Error('random() called'); } },
});
const ex = instance.exports;

// Replay games with pgx's rules (the reference rules) to get positions: a small Python helper
// prints the wire states of every position of every game.
const states = JSON.parse(execFileSync('python3', ['-c', `
import json, sys, numpy as np, jax, jax.numpy as jnp
from pgx._src.games import gess as G
game = G.Game(); step = jax.jit(game.step)
D = json.load(open(sys.argv[1])); OPEN = {o['id']: o['state'] for o in D['openings']}
out = []
for g in D['games'][::3]:
    st = OPEN[g['opening']]; x = game.init()
    b = np.array(['.BW'.index(c) for c in st['board']], dtype=np.asarray(x.board).dtype)
    x = x._replace(board=jnp.asarray(b.reshape(np.asarray(x.board).shape)), color=jnp.int32(st['to_move']), no_capture_turns=jnp.int32(st['quiet_moves']))
    for i, m in enumerate(g['moves']):
        if i % 7 == 3:
            out.append({'board': ''.join('.BW'[v] for v in np.asarray(x.board).reshape(-1)), 'to_move': int(x.color), 'quiet_moves': int(x.no_capture_turns)})
        x = step(step(x, m['move']['from']), m['move']['to'])
print(json.dumps(out[:150]))
`, resultsPath], { env: { ...process.env, JAX_PLATFORMS: 'cpu' }, maxBuffer: 1 << 28 }).toString());

function load(st) {
  const bytes = new Uint8Array(400);
  for (let i = 0; i < 400; i++) bytes[i] = st.board[i] === 'W' ? 1 : st.board[i] === 'B' ? 2 : 0;
  new Uint8Array(memory.buffer, ex.board_ptr(), 400).set(bytes);
  ex.set_quiet_moves(st.quiet_moves);
}
const out32 = (n) => new Int32Array(memory.buffer, ex.out_ptr(), n);
const cell = (r, c) => r * 20 + c;

const records = [];
const t0 = Date.now();
for (const [k, st] of states.entries()) {
  const white = st.to_move === 1 ? 1 : 0;
  load(st);
  const n = ex.gen_root(white);
  const g = out32(1 + 6 * n);
  const moves = [];
  for (let i = 0; i < n; i++) {
    const [fr, fc, tr, tc, outcome, key] = g.subarray(1 + 6 * i, 7 + 6 * i);
    moves.push([cell(fr, fc), cell(tr, tc), outcome, key]);
  }
  const rec = { state: st, moves, search_root: {}, search_to_depth: {} };
  if (k < 60) {
    for (const d of [1, 2, 3]) {
      load(st);
      const m = ex.search_root(white, d, 0);
      const o = out32(2 + 5 * m);
      const scores = [];
      for (let i = 0; i < m; i++) scores.push(o[2 + 5 * i + 4]);
      rec.search_root[d] = { best: o[0], scores };
    }
  }
  for (const d of [1, 2, 3, 4]) {
    load(st);
    rec.search_to_depth[d] = ex.search_to_depth(white, d, 1);
  }
  records.push(rec);
  if (k % 25 === 0) console.error(`${k + 1}/${states.length} (${Math.round((Date.now() - t0) / 1000)} s)`);
}
mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(outPath, gzipSync(JSON.stringify(records)));
console.error(`wrote ${records.length} positions to ${outPath}`);
