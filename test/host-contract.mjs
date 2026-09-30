// The host must implement every extern the contract declares.
//
// `src/web/gpu.almd` is the contract; `host/gpu.js` is the browser side of it.
// They drifted once already — the implementation lived in a downstream repo's
// example, so nothing checked that it kept up. This checks.
//
// Usage: node test/host-contract.mjs

import { readFile } from "node:fs/promises";

const almd = await readFile("src/web/gpu.almd", "utf8");
const js = await readFile("host/gpu.js", "utf8");

// Every `@extern(wasm, "gpu", "name")` in the contract.
const declared = [...almd.matchAll(/@extern\(wasm,\s*"gpu",\s*"([a-z0-9_]+)"\)/g)].map((m) => m[1]);

// Entry points the import object exposes. Both `name(...)` and `name: ...` forms.
// Only the import object — the returned host object below it is not part of
// the wasm contract.
const start = js.indexOf("const imports = {");
const body = js.slice(start, js.indexOf("\n  return {", start));
const implemented = new Set([
  ...[...body.matchAll(/^\s{4}([a-z0-9_]+)\s*\(/gm)].map((m) => m[1]),
  ...[...body.matchAll(/^\s{4}([a-z0-9_]+)\s*:/gm)].map((m) => m[1]),
]);

const missing = declared.filter((n) => !implemented.has(n));
const extra = [...implemented].filter((n) => !declared.includes(n));

console.log(`gpu namespace — ${declared.length} declared, ${implemented.size} implemented\n`);
for (const n of missing) console.log(`  MISSING  ${n}`);
for (const n of extra) console.log(`  extra    ${n}  (implemented but not declared)`);

if (missing.length) {
  console.error(`\nFAILED — the host does not implement ${missing.length} declared extern(s).`);
  console.error("A missing import is a LinkError at instantiation, not a build failure.");
  process.exit(1);
}
if (extra.length) {
  console.error(`\nFAILED — ${extra.length} entry point(s) exist with no declaration to justify them.`);
  console.error("Declare them in src/web/gpu.almd or remove them.");
  process.exit(1);
}
console.log(`passed — the host implements the contract exactly`);
