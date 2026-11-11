// "Signal conditions": three ridge-line fields, one per geometric signal the detector measures.
// Each is a stack of horizontal lines displaced by a field that evolves in time and bends towards the pointer.
//   sheaf    - a local disagreement that splits into two peaks (claims that contradict on an overlap)
//   curvature- the whole fabric bends around a grafted claim (Forman-Ricci curvature shift)
//   spectral - banded oscillations whose frequency drifts (Laplacian spectrum moving)

const FIELDS = {
  sheaf(x, y, t, px, py) {
    const d1 = Math.hypot(x - 0.42 - 0.06 * Math.sin(t * 0.7), y - 0.5), d2 = Math.hypot(x - 0.6 + 0.05 * Math.cos(t * 0.6), y - 0.48);
    const p = Math.exp(-(d1 * d1) / 0.012) + 0.8 * Math.exp(-(d2 * d2) / 0.01);
    return p * (0.9 + 0.1 * Math.sin(t * 2 + x * 9)) + 0.6 * Math.exp(-((x - px) ** 2 + (y - py) ** 2) / 0.01);
  },
  curvature(x, y, t, px, py) {
    const cx = 0.5 + 0.12 * Math.sin(t * 0.4), dx = x - cx;
    const bend = 0.55 * Math.exp(-(dx * dx) / 0.05) * Math.sin((y - 0.5) * 3.1 + t * 0.5);
    return bend + 0.5 * Math.exp(-((x - px) ** 2 + (y - py) ** 2) / 0.02) * Math.sign(x - px || 1) * 0.6;
  },
  spectral(x, y, t, px, py) {
    const f = 7 + 4 * Math.sin(t * 0.3 + y * 2);
    const env = Math.exp(-((x - 0.55) ** 2) / 0.06);
    return 0.35 * env * Math.sin(x * f * 6.283 + t * 1.6 + y * 5) + 0.45 * Math.exp(-((x - px) ** 2 + (y - py) ** 2) / 0.012);
  },
};

export function initSignals(root) {
  const still = document.documentElement.classList.contains("static") || matchMedia("(prefers-reduced-motion: reduce)").matches;
  const panels = [...root.querySelectorAll("canvas[data-field]")].map((c) => ({ c, f: FIELDS[c.dataset.field], px: -1, py: -1, tx: -1, ty: -1, vis: false }));
  panels.forEach((P) => {
    P.c.addEventListener("pointermove", (e) => { const r = P.c.getBoundingClientRect(); P.tx = (e.clientX - r.left) / r.width; P.ty = (e.clientY - r.top) / r.height; });
    P.c.addEventListener("pointerleave", () => { P.tx = -1; P.ty = -1; });
    new IntersectionObserver(([e]) => { P.vis = e.isIntersecting; if (P.vis) loop(); }).observe(P.c);
  });
  let raf = 0, run = false;
  function draw(P, t) {
    const c = P.c, dpr = Math.min(2, devicePixelRatio || 1), w = c.clientWidth, h = c.clientHeight;
    if (!w || !h) return;
    if (c.width !== Math.round(w * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    P.px += ((P.tx < 0 ? -1 : P.tx) - P.px) * 0.12; P.py += ((P.ty < 0 ? -1 : P.ty) - P.py) * 0.12;
    const L = 30, step = 3, amp = h / L * 2.6;
    g.lineWidth = 1;
    for (let i = 0; i < L; i++) {
      const y = (i + 0.5) / L, base = 8 + y * (h - 16);
      g.beginPath();
      for (let X = 0; X <= w; X += step) {
        const x = X / w, d = P.f(x, y, t, P.px, P.py);
        const Y = base - d * amp;
        X ? g.lineTo(X, Y) : g.moveTo(X, Y);
      }
      // occlusion: fill under each line with paper so nearer ridges hide farther ones (joy-division look)
      g.lineTo(w, h); g.lineTo(0, h); g.closePath();
      g.fillStyle = "#fcfcfb"; g.fill();
      g.strokeStyle = "#0e0e0c"; g.stroke();
    }
  }
  function loop() {
    if (run) return;
    run = true;
    const t0 = performance.now();
    const f = (now) => {
      const t = (now - t0) / 1000;
      let any = false;
      panels.forEach((P) => { if (P.vis) { any = true; draw(P, t); } });
      if (any && !still) raf = requestAnimationFrame(f); else run = false;
    };
    raf = requestAnimationFrame(f);
  }
  if (still) panels.forEach((P) => draw(P, 2));
  addEventListener("resize", () => panels.forEach((P) => draw(P, 2)));
}
