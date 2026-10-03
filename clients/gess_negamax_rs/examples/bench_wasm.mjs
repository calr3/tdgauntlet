// The app's .wasm `bench_fixed` on the identity fixture's positions (see bench.rs).
//   ~/node22/bin/node clients/gess_negamax_rs/examples/bench_wasm.mjs <depth>
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
const depth = Number(process.argv[2] ?? 4);
const here = new URL('.', import.meta.url).pathname;
const recs = JSON.parse(gunzipSync(readFileSync(`${here}../tests/data/identity.json.gz`)).toString());
const wasm = `${process.env.HOME}/calr3gh/gess/wasm-engine/target/wasm32-unknown-unknown/release/gess_engine_wasm.wasm`;
const memory = new WebAssembly.Memory({ initial: 512, maximum: 1024, shared: true });
const { instance } = await WebAssembly.instantiate(readFileSync(wasm), { env: { memory, now: () => Date.now(), random: () => 0 } });
const ex = instance.exports;
let nodes = 0; const per = [];
const t0 = performance.now();
for (const r of recs) {
  const st = r.state; const bytes = new Uint8Array(400);
  for (let i = 0; i < 400; i++) bytes[i] = st.board[i] === 'W' ? 1 : st.board[i] === 'B' ? 2 : 0;
  new Uint8Array(memory.buffer, ex.board_ptr(), 400).set(bytes);
  ex.set_quiet_moves(st.quiet_moves);
  const n = ex.bench_fixed(st.to_move === 1 ? 1 : 0, depth) >>> 0;
  per.push(n); nodes += n;
}
const secs = (performance.now() - t0) / 1000;
console.log(`wasm: depth ${depth}, ${recs.length} positions, ${nodes} nodes in ${secs.toFixed(2)} s = ${(nodes / secs / 1000).toFixed(0)} knodes/s; node counts ${JSON.stringify(per)}`);
