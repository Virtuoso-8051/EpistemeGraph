// Hero: thousands of strands of a strange attractor, drawn like an engraving on paper, that MORPH on the GPU
// between three attractors (Lorenz -> Aizawa -> Halvorsen) as the visitor scrolls through the opening
// chapters. The canvas is fixed behind the first sections; the camera orbits, dollies and drifts with the
// scroll position and the pointer. Entrance: the strands are drawn in while the camera flies in from deep space.
import * as THREE from "../vendor/three.module.min.js";

// ------------------------------------------------------------------------------------------ attractors
const SYSTEMS = {
  lorenz: { dt: 0.0055, stride: 2, warm: 1800, f(x, y, z) { return [10 * (y - x), x * (28 - z) - y, x * y - (8 / 3) * z]; }, seed: [0.1, 0, 20] },
  aizawa: { dt: 0.012, stride: 2, warm: 2500, f(x, y, z) {
    const a = 0.95, b = 0.7, c = 0.6, d = 3.5, e = 0.25, f = 0.1;
    return [(z - b) * x - d * y, d * x + (z - b) * y, c + a * z - (z * z * z) / 3 - (x * x + y * y) * (1 + e * z) + f * z * x * x * x];
  }, seed: [0.1, 0, 0] },
  halvorsen: { dt: 0.0045, stride: 2, warm: 2500, f(x, y, z) {
    const a = 1.89;
    return [-a * x - 4 * y - 4 * z - y * y, -a * y - 4 * z - 4 * x - z * z, -a * z - 4 * x - 4 * y - x * x];
  }, seed: [-1.48, -1.51, 2.04] },
};

function rk2(sys, p) {
  const h = sys.dt, [x, y, z] = p;
  const k1 = sys.f(x, y, z);
  const k2 = sys.f(x + h * k1[0] * 0.5, y + h * k1[1] * 0.5, z + h * k1[2] * 0.5);
  return [x + h * k2[0], y + h * k2[1], z + h * k2[2]];
}

/** S strands x M points of one attractor, centred and scaled to radius ~1.6 (same layout for every system). */
function strands(name, S, M, rng) {
  const sys = SYSTEMS[name];
  let p = sys.seed.slice();
  for (let i = 0; i < sys.warm; i++) p = rk2(sys, p);
  const out = new Float32Array(S * M * 3);
  for (let s = 0; s < S; s++) {
    let q = [p[0] + (rng() - 0.5) * 0.02, p[1] + (rng() - 0.5) * 0.02, p[2] + (rng() - 0.5) * 0.02];
    for (let i = 0; i < 260 + s * 37; i++) q = rk2(sys, q);    // spread the strands along the attractor
    for (let j = 0; j < M; j++) {
      for (let k = 0; k < sys.stride; k++) q = rk2(sys, q);
      if (!isFinite(q[0])) q = p.slice();
      out.set(q, (s * M + j) * 3);
    }
  }
  // normalise: centre on the mean, scale so the 97th-percentile radius is 1.6
  const n = S * M, c = [0, 0, 0];
  for (let i = 0; i < n; i++) { c[0] += out[i * 3]; c[1] += out[i * 3 + 1]; c[2] += out[i * 3 + 2]; }
  c[0] /= n; c[1] /= n; c[2] /= n;
  const r = [];
  for (let i = 0; i < n; i += 7) r.push(Math.hypot(out[i * 3] - c[0], out[i * 3 + 1] - c[1], out[i * 3 + 2] - c[2]));
  r.sort((a, b) => a - b);
  const sc = 1.6 / (r[Math.floor(r.length * 0.97)] || 1);
  for (let i = 0; i < n; i++) { out[i * 3] = (out[i * 3] - c[0]) * sc; out[i * 3 + 1] = (out[i * 3 + 1] - c[1]) * sc; out[i * 3 + 2] = (out[i * 3 + 2] - c[2]) * sc; }
  // orientation: lorenz/aizawa have z as the "up" axis
  if (name !== "halvorsen") for (let i = 0; i < n; i++) { const y = out[i * 3 + 1]; out[i * 3 + 1] = out[i * 3 + 2]; out[i * 3 + 2] = y; }
  return out;
}

const VERT = /* glsl */`
  attribute vec3 pB; attribute vec3 pC; attribute float aT; attribute vec3 aCol;
  uniform float uMorph, uTime, uReveal, uPix;
  varying float vA; varying vec3 vCol;
  void main() {
    float m = uMorph;
    vec3 p = m < 1.0 ? mix(position, pB, smoothstep(0.0, 1.0, m)) : mix(pB, pC, smoothstep(0.0, 1.0, m - 1.0));
    float ph = aT * 6.2831 * 9.0;
    p += 0.012 * vec3(sin(uTime * 0.6 + ph), cos(uTime * 0.5 + ph * 1.3), sin(uTime * 0.4 + ph * 0.7));
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    gl_Position = projectionMatrix * mv;
    float depth = clamp((-mv.z - 2.5) / 6.0, 0.0, 1.0);
    vA = (1.0 - smoothstep(uReveal - 0.04, uReveal, aT)) * mix(1.0, 0.35, depth);
    vCol = aCol;
  }`;
const FRAG = /* glsl */`
  uniform float uOpacity;
  varying float vA; varying vec3 vCol;
  void main() { gl_FragColor = vec4(vCol, vA * uOpacity); }`;

export function initHero(canvas) {
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches || document.documentElement.classList.contains("static");
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: "high-performance" });
  } catch (_) { return { setProgress() {}, enter() {}, ready: Promise.resolve() }; } // no WebGL: page still works
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setClearColor(0x000000, 0);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(36, 1, 0.1, 100);
  const group = new THREE.Group();
  scene.add(group);

  // seeded RNG -> identical figure on every visit
  let s0 = 7;
  const rng = () => ((s0 = (s0 * 16807) % 2147483647) / 2147483647);
  const S = innerWidth < 700 ? 48 : 84, M = innerWidth < 700 ? 900 : 1400;
  const A = strands("lorenz", S, M, rng), B = strands("aizawa", S, M, rng), Cc = strands("halvorsen", S, M, rng);

  const T = new Float32Array(S * M), COL = new Float32Array(S * M * 3);
  const ink = [0.09, 0.085, 0.07], gold = [0.69, 0.5, 0.15], blue = [0.165, 0.47, 0.84];
  for (let s = 0; s < S; s++) {
    const col = s % 9 === 0 ? blue : s % 3 === 0 ? gold : ink;
    for (let j = 0; j < M; j++) { T[s * M + j] = j / (M - 1); COL.set(col, (s * M + j) * 3); }
  }
  const idx = new Uint32Array(S * (M - 1) * 2);
  let w = 0;
  for (let s = 0; s < S; s++) for (let j = 0; j < M - 1; j++) { idx[w++] = s * M + j; idx[w++] = s * M + j + 1; }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(A, 3));
  geo.setAttribute("pB", new THREE.BufferAttribute(B, 3));
  geo.setAttribute("pC", new THREE.BufferAttribute(Cc, 3));
  geo.setAttribute("aT", new THREE.BufferAttribute(T, 1));
  geo.setAttribute("aCol", new THREE.BufferAttribute(COL, 3));
  geo.setIndex(new THREE.BufferAttribute(idx, 1));
  geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 4);
  const uni = { uMorph: { value: 0 }, uTime: { value: 0 }, uReveal: { value: reduce ? 1.05 : 0 }, uOpacity: { value: 0.2 }, uPix: { value: 1 } };
  const mat = new THREE.ShaderMaterial({ vertexShader: VERT, fragmentShader: FRAG, uniforms: uni, transparent: true, depthWrite: false });
  group.add(new THREE.LineSegments(geo, mat));

  // probe: the "verified path" bead that travels along one strand
  const probe = new THREE.Mesh(new THREE.SphereGeometry(0.03, 16, 16), new THREE.MeshBasicMaterial({ color: 0xeb6834 }));
  const halo = new THREE.Mesh(new THREE.RingGeometry(0.055, 0.068, 40), new THREE.MeshBasicMaterial({ color: 0xeb6834, transparent: true, opacity: 0.55, side: THREE.DoubleSide }));
  group.add(probe, halo);

  // scroll-driven state (target) and eased state (cur)
  const tgt = { p: 0, mx: 0, my: 0 }, cur = { p: 0, mx: 0, my: 0, fly: reduce ? 0 : 1 };
  addEventListener("pointermove", (e) => { tgt.mx = e.clientX / innerWidth - 0.5; tgt.my = e.clientY / innerHeight - 0.5; }, { passive: true });

  let W = 1, H = 1;
  function resize() {
    W = canvas.clientWidth || innerWidth; H = canvas.clientHeight || innerHeight;
    renderer.setSize(W, H, false);
    camera.aspect = W / H;
    camera.updateProjectionMatrix();
  }
  resize();
  addEventListener("resize", resize);

  const lerp = (a, b, t) => a + (b - a) * t;
  const wide = () => W > 900;
  // keyframes over the opening chapters: [progress, x-offset, camera distance, yaw, pitch, morph, opacity]
  // the figure holds on the side OPPOSITE the text of each opening block (hero title right, premise 1 left, premise 2 right)
  const KF = [
    [0.00, -1.5, 5.8, 0.0, -0.18, 0.00, 0.13],
    [0.12, -1.5, 5.8, 0.5, -0.10, 0.35, 0.13],
    [0.22, 1.55, 5.6, 1.1, 0.10, 0.85, 0.135],
    [0.34, 1.55, 5.6, 1.5, 0.12, 1.05, 0.135],
    [0.45, -1.55, 5.7, 2.2, -0.05, 1.6, 0.135],
    [0.60, -1.55, 5.7, 2.6, 0.00, 2.0, 0.12],
    [0.74, 0.0, 7.6, 3.2, 0.30, 2.0, 0.0],
    [1.00, 0.0, 8.0, 3.3, 0.30, 2.0, 0.0],
  ];
  function at(p) {
    p = Math.max(0, Math.min(1, p));
    let i = 0;
    while (i < KF.length - 2 && p > KF[i + 1][0]) i++;
    const a = KF[i], b = KF[i + 1], t = (p - a[0]) / (b[0] - a[0] || 1), e = t * t * (3 - 2 * t);
    return a.map((v, k) => lerp(v, b[k], e));
  }

  let raf = 0, last = 0, running = false, t0 = performance.now();
  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000 || 0.016);
    last = now;
    cur.p = lerp(cur.p, tgt.p, 1 - Math.pow(0.0015, dt));
    cur.mx = lerp(cur.mx, tgt.mx, 0.05); cur.my = lerp(cur.my, tgt.my, 0.05);
    cur.fly = Math.max(0, cur.fly - dt * 0.55);
    const [, x, dist, yaw, pitch, morph, op] = at(cur.p);
    const fly = cur.fly * cur.fly;
    group.position.x = wide() ? x : 0;
    group.scale.setScalar(wide() ? 0.84 : 0.7);
    group.rotation.y = yaw + (now - t0) * 0.00004 + cur.mx * 0.5 + fly * 2.4;
    group.rotation.x = pitch + cur.my * 0.3 - fly * 0.6;
    camera.position.set(0, 0.05, dist + fly * 9);
    camera.lookAt(group.position.x * 0.15, 0, 0);
    uni.uMorph.value = morph;
    uni.uOpacity.value = wide() ? op : op * 0.75;
    uni.uTime.value = (now - t0) / 1000;
    if (!reduce && uni.uReveal.value < 1.05) uni.uReveal.value = Math.min(1.05, uni.uReveal.value + dt * (entered ? 0.42 : 0.0));
    // probe along strand #0 (in the currently displayed morph, approximated by attractor A/B/C mix on CPU)
    const k = Math.floor(((now - t0) * 0.09) % (M - 1));
    const src = morph < 1 ? [A, B, morph] : [B, Cc, morph - 1];
    const j = k * 3, m = src[2] * src[2] * (3 - 2 * src[2]);
    probe.position.set(lerp(src[0][j], src[1][j], m), lerp(src[0][j + 1], src[1][j + 1], m), lerp(src[0][j + 2], src[1][j + 2], m));
    halo.position.copy(probe.position);
    halo.quaternion.copy(camera.quaternion);
    halo.scale.setScalar(1 + 0.3 * Math.sin(now * 0.004));
    probe.visible = halo.visible = op > 0.02;
    canvas.style.opacity = op > 0.005 ? "1" : "0";
    renderer.render(scene, camera);
    if (op > 0.005 || Math.abs(cur.p - tgt.p) > 0.001) raf = requestAnimationFrame(frame);
    else running = false;
  }
  function wake() { if (!running && !reduce) { running = true; last = performance.now(); raf = requestAnimationFrame(frame); } }
  document.addEventListener("visibilitychange", () => { if (document.hidden) { cancelAnimationFrame(raf); running = false; } else wake(); });

  let entered = reduce;
  if (reduce) { cur.fly = 0; requestAnimationFrame((n) => { last = n; frame(n); }); }
  else wake();

  return {
    /** 0 at the top of the page, 1 when the opening chapters have scrolled away. */
    setProgress(p) { tgt.p = p; if (reduce) { cur.p = p; frame(performance.now()); } else wake(); },
    /** entrance: fly in from depth and draw the strands. */
    enter() { entered = true; cur.fly = reduce ? 0 : 1; wake(); },
  };
}
