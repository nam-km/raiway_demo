import fs from 'fs';
import path from 'path';
import loadMujoco from '@mujoco/mujoco';
import * as ort from 'onnxruntime-web';
import { TERRAINS } from '../src/terrain.js';
import { RaiwaySim } from '../src/sim.js';

const [terrainName = 'flat', vx = '1.0', wz = '0.0', seconds = '8', level = '1.0', policyName] = process.argv.slice(2);
const assets = path.resolve(import.meta.dirname, '../public/model');
const mj = await loadMujoco();
const vfs = new mj.MjVFS();
vfs.addBuffer('raiway.xml', new Uint8Array(fs.readFileSync(path.join(assets, 'raiway.xml'))));
for (const f of fs.readdirSync(path.join(assets, 'meshes')))
  vfs.addBuffer('meshes/' + f, new Uint8Array(fs.readFileSync(path.join(assets, 'meshes', f))));
const policy = path.join(assets, 'policies', policyName ?? JSON.parse(fs.readFileSync(path.join(assets, 'policies.json'))).default);
const cfg = JSON.parse(fs.readFileSync(path.join(policy, 'config.json')));
ort.env.wasm.numThreads = 1;
const session = await ort.InferenceSession.create(fs.readFileSync(path.join(policy, 'policy.onnx')));
const sim = new RaiwaySim(mj, vfs, cfg, session, ort, TERRAINS[terrainName](Number(level)));
sim.reset();
const t0 = performance.now();
for (let k = 0; k < 100; k++) await sim.step(0, 0);
const steps = Math.round(Number(seconds) / cfg.control_dt);
const vxs = [], wzs = [];
let fell = null;
for (let k = 0; k < steps; k++) {
  if (await sim.step(Number(vx), Number(wz))) { fell = k * cfg.control_dt; break; }
  const { lin, ang } = sim.bodyVelocity();
  vxs.push(lin[0]); wzs.push(ang[2]);
}
const tail = (a) => a.slice(a.length >> 1);
const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
const wall = (performance.now() - t0) / 1000;
console.log(JSON.stringify({ fell_at: fell, x: sim.data.qpos[0], vx_mean: fell === null ? mean(tail(vxs)) : null,
  wz_mean: fell === null ? mean(tail(wzs)) : null, est_vx: sim.estimate[0], realtime_factor: (steps + 100) * cfg.control_dt / wall }));
