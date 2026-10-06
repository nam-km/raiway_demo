import loadMujoco from '@mujoco/mujoco';
import * as ort from 'onnxruntime-web/wasm';
import { TERRAINS, describe } from './terrain.js';
import { RaiwaySim } from './sim.js';
import { Viewer } from './render.js';
import { Joystick } from './joystick.js';
import './style.css';

const ASSETS = import.meta.env.BASE_URL + 'model/';
const MESHES = ['raiway_TORSO_new', 'raiway_L_THIGH', 'raiway_L_SHANK', 'raiway_L_WHEEL', 'raiway_R_THIGH', 'raiway_R_SHANK', 'raiway_R_WHEEL'];
const ACCEL = { vx: 2.0, wz: 4.0 };
const $ = (id) => document.getElementById(id);

async function fetchBytes(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return new Uint8Array(await res.arrayBuffer());
}

function status(text) {
  $('loading').textContent = text;
}

async function main() {
  status('Loading MuJoCo…');
  const mj = await loadMujoco();
  status('Loading robot and policy…');
  ort.env.wasm.numThreads = 1;
  const [manifest, xml, ...meshes] = await Promise.all([
    fetch(ASSETS + 'policies.json').then((r) => r.json()),
    fetchBytes(ASSETS + 'raiway.xml'),
    ...MESHES.map((n) => fetchBytes(`${ASSETS}meshes/${n}.STL`)),
  ]);
  const policies = new Map();
  const loadPolicy = (name) => {
    if (!policies.has(name)) {
      const dir = `${ASSETS}policies/${name}/`;
      policies.set(name, Promise.all([fetch(dir + 'config.json').then((r) => r.json()), fetchBytes(dir + 'policy.onnx')])
        .then(async ([cfg, onnx]) => ({ name, cfg, session: await ort.InferenceSession.create(onnx, { executionProviders: ['wasm'] }) }))
        .catch((e) => { policies.delete(name); throw e; }));
    }
    return policies.get(name);
  };
  for (const p of manifest.policies) $('policy').add(new Option(p.label, p.name, false, p.name === manifest.default));
  let { cfg, session } = await loadPolicy(manifest.default);
  const vfs = new mj.MjVFS();
  vfs.addBuffer('raiway.xml', xml);
  MESHES.forEach((n, i) => vfs.addBuffer(`meshes/${n}.STL`, meshes[i]));

  const terrains = {};
  const level = () => Number($('level').value);
  const getTerrain = (name) => (terrains[`${name}:${level()}`] ??= TERRAINS[name](level()));
  const sim = new RaiwaySim(mj, vfs, cfg, session, ort, getTerrain($('terrain').value));
  sim.reset();
  const viewer = new Viewer($('view'), mj, sim);
  viewer.setTerrain(sim.terrain);
  $('loading').remove();
  $('checkpoint').textContent = cfg.checkpoint;

  const joystick = new Joystick($('joystick'));
  const keys = new Set();
  const state = { cmd: [0, 0], fallTimer: null, paused: false, speed: 1, policy: null, active: manifest.default };

  const showLevel = () => {
    const name = $('terrain').value;
    $('level').disabled = name === 'flat';
    $('level-v').textContent = level().toFixed(2);
    $('level-d').textContent = describe(name, level());
  };
  const setTerrain = () => {
    showLevel();
    sim.setTerrain(getTerrain($('terrain').value));
    viewer.setTerrain(sim.terrain);
    reset();
  };
  showLevel();
  const reset = () => {
    sim.reset();
    state.cmd = [0, 0];
    state.fallTimer = null;
    $('banner').hidden = true;
  };

  $('policy').addEventListener('change', async (e) => {
    const name = e.target.value;
    try {
      const next = await loadPolicy(name);
      if ($('policy').value === name) state.policy = next;
    } catch (err) {
      console.error(err);
      if ($('policy').value === name) $('policy').value = state.active;
    }
  });
  $('terrain').addEventListener('change', setTerrain);
  $('level').addEventListener('input', showLevel);
  $('level').addEventListener('change', setTerrain);
  $('reset').addEventListener('click', reset);
  $('pause').addEventListener('click', () => {
    state.paused = !state.paused;
    $('pause').textContent = state.paused ? 'Resume' : 'Pause';
  });
  viewer.scan.visible = $('scan').checked;
  $('scan').addEventListener('change', (e) => { viewer.scan.visible = e.target.checked; });
  $('follow').addEventListener('change', (e) => { viewer.follow = e.target.checked; });
  for (const id of ['vmax', 'wmax', 'slow']) {
    const input = $(id), out = $(id + '-v');
    const sync = () => {
      out.textContent = Number(input.value).toFixed(2);
      if (id === 'slow') state.speed = Number(input.value);
    };
    input.addEventListener('input', sync);
    sync();
  }
  const DRIVE_KEYS = new Set(['w', 'a', 's', 'd', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright']);
  window.addEventListener('keydown', (e) => {
    const key = e.key.toLowerCase();
    if (DRIVE_KEYS.has(key)) {
      e.preventDefault();
      document.activeElement?.blur();
    }
    keys.add(key);
    if (key === 'r') reset();
  }, { capture: true });
  window.addEventListener('blur', () => keys.clear());
  window.addEventListener('keyup', (e) => keys.delete(e.key.toLowerCase()));

  const velocityTarget = () => {
    const vmax = Number($('vmax').value), wmax = Number($('wmax').value);
    const axis = (pos, neg) => (pos.some((k) => keys.has(k)) ? 1 : 0) - (neg.some((k) => keys.has(k)) ? 1 : 0);
    const fwd = axis(['w', 'arrowup'], ['s', 'arrowdown']) || -joystick.y;
    const turn = axis(['a', 'arrowleft'], ['d', 'arrowright']) || -joystick.x;
    return [fwd * vmax, turn * wmax];
  };

  let last = performance.now(), acc = 0, busy = false, simClock = 0, wallClock = 0;
  const loop = async (now) => {
    requestAnimationFrame(loop);
    const dt = Math.min((now - last) / 1000, 0.1);
    last = now;
    if (busy) return;
    busy = true;
    if (state.policy) {
      ({ cfg, session } = state.policy);
      state.active = state.policy.name;
      state.policy = null;
      sim.setPolicy(cfg, session);
      viewer.buildScan();
      viewer.scan.visible = $('scan').checked;
      $('checkpoint').textContent = cfg.checkpoint;
      reset();
    }
    if (!state.paused) acc += dt * state.speed;
    const ctrl = cfg.control_dt;
    let n = 0;
    while (acc >= ctrl && n < 6) {
      acc -= ctrl;
      n++;
      if (state.fallTimer !== null) {
        state.fallTimer -= ctrl;
        if (state.fallTimer <= 0) reset();
        continue;
      }
      const target = velocityTarget();
      state.cmd = state.cmd.map((c, i) => {
        const a = (i === 0 ? ACCEL.vx : ACCEL.wz) * ctrl;
        return c + Math.min(Math.max(target[i] - c, -a), a);
      });
      const fell = await sim.step(state.cmd[0], state.cmd[1]);
      simClock += ctrl;
      const q = sim.data.qpos;
      if (fell || !sim.terrain.inside(q[0], q[1], 0.6)) {
        state.fallTimer = 1.0;
        $('banner').textContent = fell ? 'Fell — resetting' : 'Edge of map — resetting';
        $('banner').hidden = false;
      }
    }
    if (acc > ctrl * 6) acc = 0;
    viewer.render();
    const { lin, ang } = sim.bodyVelocity();
    $('hud-cmd').textContent = `${state.cmd[0].toFixed(2)} m/s  ${state.cmd[1].toFixed(2)} rad/s`;
    $('hud-act').textContent = `${lin[0].toFixed(2)} m/s  ${ang[2].toFixed(2)} rad/s`;
    wallClock += dt;
    if (wallClock > 1) {
      $('hud-rt').textContent = `${(simClock / wallClock).toFixed(2)}×`;
      simClock = wallClock = 0;
    }
    busy = false;
  };
  requestAnimationFrame(loop);
}

main().catch((e) => {
  console.error(e);
  status('Failed to load: ' + e.message);
});
