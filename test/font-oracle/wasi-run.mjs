// Run dump.almd's wasm build under host/wasi.js — the browser's WASI — with
// the font handed over as a file, so the page's font path is the one
// sfnt_test.almd and oracle.py check natively.
//
//   node test/font-oracle/wasi-run.mjs DUMP.wasm FONT FACE CODE...
import { readFileSync } from "node:fs";
import { createWasi, WasiExit } from "../../host/wasi.js";

const [wasmPath, font, ...rest] = process.argv.slice(2);
const out = [];
const w = createWasi({
  files: { "/font": readFileSync(font) },
  args: ["dump", "/font", ...rest],
  stdout: (s) => out.push(s),
  stderr: (s) => process.stderr.write(s + "\n"),
});
const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), { wasi_snapshot_preview1: w.imports });
w.setMemory(instance.exports.memory);
let code = 0;
try { instance.exports._start(); } catch (e) { if (e instanceof WasiExit) code = e.code; else throw e; }
w.flush();
process.stdout.write(out.join("\n") + (out.length ? "\n" : ""));
process.exit(code);
