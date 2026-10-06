import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

export class Viewer {
  constructor(container, mj, sim) {
    Object.assign(this, { mj, sim });
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xdfe6ee);
    this.scene.fog = new THREE.Fog(0xdfe6ee, 12, 30);
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.05, 100);
    this.camera.position.set(-2.5, -2.5, 1.8);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0, 0.6);
    this.controls.enableDamping = true;
    this.follow = true;

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x667788, 1.3));
    this.sun = new THREE.DirectionalLight(0xffffff, 2.0);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    Object.assign(this.sun.shadow.camera, { left: -4, right: 4, top: 4, bottom: -4, near: 0.5, far: 20 });
    this.sun.shadow.bias = -0.0005;
    this.scene.add(this.sun, this.sun.target);

    this.buildRobot();
    this.buildScan();
    this.resize();
    window.addEventListener('resize', () => this.resize());
  }

  resize() {
    const el = this.renderer.domElement.parentElement;
    this.renderer.setSize(el.clientWidth, el.clientHeight);
    this.camera.aspect = el.clientWidth / el.clientHeight;
    this.camera.updateProjectionMatrix();
  }

  buildRobot() {
    const m = this.sim.model, meshType = this.mj.mjtGeom.mjGEOM_MESH.value;
    const cache = new Map();
    this.geoms = [];
    for (let g = 0; g < m.ngeom; g++) {
      if (m.geom_group[g] !== 1 || m.geom_type[g] !== meshType) continue;
      const id = m.geom_dataid[g];
      if (!cache.has(id)) cache.set(id, meshGeometry(m, id));
      const rgba = m.geom_rgba.slice(4 * g, 4 * g + 4);
      const mat = new THREE.MeshStandardMaterial({
        color: new THREE.Color(rgba[0], rgba[1], rgba[2]), metalness: 0.2, roughness: 0.55, flatShading: false,
      });
      const mesh = new THREE.Mesh(cache.get(id), mat);
      mesh.matrixAutoUpdate = false;
      mesh.castShadow = true;
      this.scene.add(mesh);
      this.geoms.push([g, mesh]);
    }
  }

  setTerrain(terrain) {
    if (this.terrainMesh) {
      this.scene.remove(this.terrainMesh);
      this.terrainMesh.geometry.dispose();
    }
    const { nx, ny, res, x0, y0, h } = terrain;
    const pos = new Float32Array(nx * ny * 3), col = new Float32Array(nx * ny * 3);
    const light = [0.62, 0.66, 0.7], dark = [0.5, 0.54, 0.59];
    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        const i = iy * nx + ix, x = x0 + ix * res, y = y0 + iy * res;
        pos.set([x, y, h[i]], 3 * i);
        const checker = (Math.floor(x / 0.5) + Math.floor(y / 0.5)) & 1;
        const shade = Math.min(Math.max(1 + 0.2 * h[i], 0.6), 1.3);
        const c = checker ? light : dark;
        col.set([c[0] * shade, c[1] * shade, c[2] * shade], 3 * i);
      }
    }
    const index = new Uint32Array((nx - 1) * (ny - 1) * 6);
    let k = 0;
    for (let iy = 0; iy < ny - 1; iy++) {
      for (let ix = 0; ix < nx - 1; ix++) {
        const a = iy * nx + ix, b = a + 1, c = a + nx, d = c + 1;
        index.set([a, b, d, a, d, c], k);
        k += 6;
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    geo.setIndex(new THREE.BufferAttribute(index, 1));
    geo.computeVertexNormals();
    this.terrainMesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.9 }));
    this.terrainMesh.receiveShadow = true;
    this.scene.add(this.terrainMesh);
  }

  buildScan() {
    if (this.scan) {
      this.scene.remove(this.scan);
      this.scan.geometry.dispose();
      this.scan.material.dispose();
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(this.sim.scanPoints, 3));
    this.scan = new THREE.Points(geo, new THREE.PointsMaterial({ color: 0xff7a00, size: 0.045, depthTest: false }));
    this.scan.renderOrder = 1;
    this.scan.frustumCulled = false;
    this.scene.add(this.scan);
  }

  render() {
    const d = this.sim.data;
    for (const [g, mesh] of this.geoms) {
      const p = d.geom_xpos, R = d.geom_xmat, r = 9 * g;
      mesh.matrix.set(R[r], R[r + 1], R[r + 2], p[3 * g], R[r + 3], R[r + 4], R[r + 5], p[3 * g + 1],
        R[r + 6], R[r + 7], R[r + 8], p[3 * g + 2], 0, 0, 0, 1);
    }
    this.scan.geometry.attributes.position.needsUpdate = true;
    const base = new THREE.Vector3(d.qpos[0], d.qpos[1], d.qpos[2]);
    if (this.follow) {
      const delta = base.clone().sub(this.controls.target).multiplyScalar(0.1);
      delta.z *= 0.5;
      this.controls.target.add(delta);
      this.camera.position.add(delta);
    }
    this.sun.position.set(base.x - 3, base.y - 2, base.z + 6);
    this.sun.target.position.copy(base);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}

function meshGeometry(m, id) {
  const va = m.mesh_vertadr[id], vn = m.mesh_vertnum[id], fa = m.mesh_faceadr[id], fn = m.mesh_facenum[id];
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(m.mesh_vert.slice(3 * va, 3 * (va + vn)), 3));
  geo.setIndex(new THREE.BufferAttribute(Uint32Array.from(m.mesh_face.subarray(3 * fa, 3 * (fa + fn))), 1));
  const flat = geo.toNonIndexed();
  flat.computeVertexNormals();
  return flat;
}
