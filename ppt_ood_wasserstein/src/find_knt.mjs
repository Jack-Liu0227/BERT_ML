import fs from 'node:fs';
const text=fs.readFileSync('C:/Users/HK/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs','utf8');
const idx=text.indexOf('KNt=');
console.log(idx); console.log(text.slice(idx-500, idx+1000));
