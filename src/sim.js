export const HFIELD_Z_LO = -1.0, HFIELD_Z_RANGE = 4.0;
const TERRAIN_CONTACT_BODIES = ['Segway_L_WHEEL', 'Segway_R_WHEEL', 'Segway_L_SHANK', 'Segway_R_SHANK'];

export function sceneXml(terrain, dt) {
  const rx = (terrain.nx - 1) * terrain.res / 2, ry = (terrain.ny - 1) * terrain.res / 2;
  return `<mujoco model="raiway_scene">
  <include file="raiway.xml"/>
  <option timestep="${dt}" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <asset>
    <hfield name="terrain" nrow="${terrain.ny}" ncol="${terrain.nx}" size="${rx} ${ry} ${HFIELD_Z_RANGE} 1.0"/>
  </asset>
  <worldbody>
    <geom name="terrain" type="hfield" hfield="terrain" pos="${terrain.x0 + rx} ${terrain.y0 + ry} ${HFIELD_Z_LO}"
          contype="1" conaffinity="2" solmix="0.0001"/>
  </worldbody>
</mujoco>`;
}

export class RaiwaySim {
  constructor(mj, vfs, cfg, session, ort, terrain) {
    Object.assign(this, { mj, ort });
    this.model = mj.MjModel.from_xml_string(sceneXml(terrain, cfg.simulation_dt), vfs);
    this.data = new mj.MjData(this.model);
    const m = this.model;
    this.baseId = m.body('Segway_TORSO').id;
    this.wheelIds = [m.body('Segway_L_WHEEL').id, m.body('Segway_R_WHEEL').id];
    this.terrainGeom = m.geom('terrain').id;
    const allowed = new Set(TERRAIN_CONTACT_BODIES.map((n) => m.body(n).id));
    this.forbidden = new Set();
    for (let g = 0; g < m.ngeom; g++) {
      if (!allowed.has(m.geom_bodyid[g]) && g !== this.terrainGeom) this.forbidden.add(g);
    }
    this.command = [0, 0, 0];
    this.setPolicy(cfg, session);
    this.setTerrain(terrain);
  }

  setPolicy(cfg, session) {
    Object.assign(this, { cfg, session });
    this.substeps = Math.round(cfg.control_dt / cfg.simulation_dt);
    const h = cfg.est_history;
    this.histDepth = 1 + (h.len - 1) * h.stride;
    if (cfg.scan_type === 3) {
      this.ringOffsets = [];
      cfg.scan_rings.points.forEach((n, k) => { for (let j = 0; j < n; j++) this.ringOffsets.push([k, j, n]); });
    }
    this.scanPoints = new Float32Array(2 * cfg.scan_n * 3);
  }

  setTerrain(terrain) {
    this.terrain = terrain;
    const hf = this.model.hfield_data;
    for (let i = 0; i < terrain.h.length; i++)
      hf[i] = Math.min(Math.max((terrain.h[i] - HFIELD_Z_LO) / HFIELD_Z_RANGE, 0), 1);
  }

  reset(x = 0, y = 0, yaw = 0) {
    const { mj, model: m, data: d, cfg } = this;
    mj.mj_resetData(m, d);
    d.qpos.set([x, y, this.terrain.height(x, y) + cfg.base_height, Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)], 0);
    d.qpos.set(cfg.nominal_joint, 7);
    mj.mj_forward(m, d);
    let clearance = Infinity;
    for (let i = 0; i < 2; i++) {
      const t = this.tireCenter(i);
      clearance = Math.min(clearance, t[2] - cfg.wheel_radius - this.terrain.height(t[0], t[1]));
    }
    d.qpos[2] -= clearance - 0.002;
    mj.mj_forward(m, d);
    const q = d.qpos, qd = d.qvel;
    const start = [q[7], q[8], qd[8], q[10], q[11], qd[11]];
    this.prevAction = start.slice();
    this.prevprevAction = start.slice();
    this.jointPosTarget = cfg.nominal_joint.slice();
    this.jointVelTarget = [0, 0, 0, 0, 0, 0];
    this.history = null;
    this.estimate = new Array(cfg.est_dim).fill(0);
  }

  tireCenter(i) {
    const d = this.data, b = this.wheelIds[i], off = this.cfg.tire_offset * (i === 0 ? 1 : -1);
    const p = d.xpos, R = d.xmat;
    return [p[3 * b] + R[9 * b + 1] * off, p[3 * b + 1] + R[9 * b + 4] * off, p[3 * b + 2] + R[9 * b + 7] * off];
  }

  baseRot() {
    return this.data.xmat.slice(9 * this.baseId, 9 * this.baseId + 9);
  }

  heightScan(R, cx, cy) {
    const cfg = this.cfg, n = cfg.scan_n, d = this.data, cmd = this.command;
    const scans = new Float64Array(2 * n);
    for (let i = 0; i < 2; i++) {
      const tire = this.tireCenter(i);
      const offsets = [];
      if (cfg.scan_type === 3) {
        const { spacing_x: sx, spacing_y: sy } = cfg.scan_rings;
        for (const [k, j, nk] of this.ringOffsets) {
          const a = 2 * Math.PI * j / nk;
          offsets.push([sx * k * Math.cos(a), sy * k * Math.sin(a)]);
        }
      } else {
        const sl = cfg.scan_line;
        const r = [tire[0] - d.qpos[0], tire[1] - d.qpos[1], tire[2] - d.qpos[2]];
        const rx = cx[0] * r[0] + cx[1] * r[1], ry = cy[0] * r[0] + cy[1] * r[1];
        const vix = cmd[0] - cmd[2] * ry, viy = cmd[1] + cmd[2] * rx;
        const speed = Math.hypot(vix, viy);
        const span = sl.fov_max - sl.look_min;
        const L = sl.look_min + span * Math.tanh(speed * sl.look_time / span);
        let t = [1, 0], nrm = [0, 1], kappa = 0;
        if (speed > 1e-4) {
          t = [vix / speed, viy / speed];
          nrm = [-t[1], t[0]];
          kappa = cmd[2] / speed;
        }
        for (let j = 0; j < n; j++) {
          const s = -sl.look_back + j * (sl.look_back + L) / (n - 1);
          let u = s, w = 0;
          if (Math.abs(kappa) > 1e-4) { u = Math.sin(kappa * s) / kappa; w = (1 - Math.cos(kappa * s)) / kappa; }
          offsets.push([u * t[0] + w * nrm[0], u * t[1] + w * nrm[1]]);
        }
      }
      offsets.forEach(([ox, oy], j) => {
        const px = tire[0] + ox * cx[0] + oy * cy[0];
        const py = tire[1] + ox * cx[1] + oy * cy[1];
        const pz = this.terrain.height(px, py);
        const dz = R[2] * (tire[0] - px) + R[5] * (tire[1] - py) + R[8] * (tire[2] - pz);
        scans[i * n + j] = Math.min(Math.max(dz, -2), 2);
        this.scanPoints.set([px, py, pz], 3 * (i * n + j));
      });
    }
    return scans;
  }

  observation() {
    const d = this.data, R = this.baseRot();
    const norm = Math.hypot(R[0], R[3]);
    const cx = [R[0] / norm, R[3] / norm, 0];
    const cy = [-cx[1], cx[0], 0];
    const q = d.qpos, qd = d.qvel;
    const ob = new Float64Array(this.cfg.ob_dim);
    ob.set([R[6], R[7], R[8], qd[3], qd[4], qd[5], q[7], q[8], q[10], q[11]], 0);
    ob.set(qd.slice(6, 12), 10);
    ob.set(this.prevAction, 16);
    ob.set(this.prevprevAction, 22);
    ob.set(this.heightScan(R, cx, cy), 28);
    if (this.history === null) this.history = Array.from({ length: this.histDepth }, () => ob);
    else { this.history.pop(); this.history.unshift(ob); }
    return ob;
  }

  estimatorInput() {
    const h = this.cfg.est_history, out = new Float32Array(this.cfg.est_input_dim);
    let o = 0;
    for (let k = 0; k < h.len; k++) {
      const f = this.history[k * h.stride];
      out.set(f.subarray(0, h.head), o); o += h.head;
      out.set(f.subarray(h.tail_start), o); o += f.length - h.tail_start;
    }
    return out;
  }

  touchesForbidden() {
    const d = this.data;
    let bad = false;
    const vec = d.contact;
    for (let k = 0; k < d.ncon; k++) {
      const c = vec.get(k);
      const g1 = c.geom1, g2 = c.geom2;
      c.delete();
      if (g1 !== this.terrainGeom && g2 !== this.terrainGeom) continue;
      if (this.forbidden.has(g1 === this.terrainGeom ? g2 : g1)) bad = true;
    }
    vec.delete();
    return bad;
  }

  async step(vx, wz) {
    const { cfg, ort } = this;
    this.command = [vx, 0, wz];
    const ob = this.observation();
    const out = await this.session.run({
      obs: new ort.Tensor('float32', Float32Array.from(ob), [1, cfg.ob_dim]),
      est_input: new ort.Tensor('float32', this.estimatorInput(), [1, cfg.est_input_dim]),
      command: new ort.Tensor('float32', Float32Array.of(vx, wz), [1, 2]),
    });
    this.estimate = Array.from(out.estimate.data);
    const a = out.action.data;
    const scaled = Array.from(a, (v, i) => v * cfg.action_std[i] + cfg.action_mean[i]);
    this.jointPosTarget = [scaled[0], scaled[1], 0, scaled[3], scaled[4], 0];
    this.jointVelTarget = [0, 0, scaled[2], 0, 0, scaled[5]];
    this.prevprevAction = this.prevAction;
    this.prevAction = scaled;

    const { mj, model: m, data: d } = this;
    const lim = cfg.torque_limit;
    for (let s = 0; s < this.substeps; s++) {
      for (let j = 0; j < 6; j++) {
        const tau = cfg.kp[j] * (this.jointPosTarget[j] - d.qpos[7 + j]) + cfg.kd[j] * (this.jointVelTarget[j] - d.qvel[6 + j]);
        d.qfrc_applied[6 + j] = Math.min(Math.max(tau, -lim), lim);
      }
      mj.mj_step(m, d);
      if (this.touchesForbidden() || this.data.xmat[9 * this.baseId + 8] < 0.5) return true;
    }
    return false;
  }

  bodyVelocity() {
    const R = this.baseRot(), v = this.data.qvel;
    return {
      lin: [R[0] * v[0] + R[3] * v[1] + R[6] * v[2], R[1] * v[0] + R[4] * v[1] + R[7] * v[2], R[2] * v[0] + R[5] * v[1] + R[8] * v[2]],
      ang: [v[3], v[4], v[5]],
    };
  }
}
