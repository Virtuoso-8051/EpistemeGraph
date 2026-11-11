// Entrance: brand mark + a progress bar driven by REAL loading steps (fonts, server state, knowledge graph,
// 3D scene). At 100 % a rounded "portal" opens in the paper and expands until it becomes the page.
// It can never trap the visitor: if the server is slow or down, the page is revealed after a few seconds
// and the loader keeps retrying in the background with a plain-language message.

export function createIntro(el) {
  const skip = !el || document.documentElement.classList.contains("static") || matchMedia("(prefers-reduced-motion: reduce)").matches;
  const tasks = new Map();          // name -> [weight, fraction]
  let shown = 0, target = 0, finished = false, onDone = [];
  if (skip) { if (el) el.remove(); return { task() {}, set() {}, msg() {}, finish() { onDone.forEach((f) => f()); onDone = []; }, then(f) { f(); } }; }

  const pctEl = el.querySelector("[data-pct]"), bar = el.querySelector("[data-bar]"), msgEl = el.querySelector("[data-msg]");
  document.documentElement.classList.add("intro-on");

  function recompute() {
    let W = 0, D = 0;
    tasks.forEach(([w, f]) => { W += w; D += w * f; });
    target = W ? D / W : 0;
  }
  function tick() {
    shown += (target - shown) * 0.12 + (target > shown ? 0.002 : 0);
    shown = Math.min(shown, target);
    const p = Math.round(shown * 100);
    pctEl.textContent = String(p).padStart(2, "0");
    bar.style.transform = `scaleX(${shown})`;
    if (shown >= 0.995 && target >= 1) return reveal();
    requestAnimationFrame(tick);
  }
  function reveal() {
    if (finished) return;
    finished = true;
    pctEl.textContent = "100";
    msgEl.textContent = "Ready";
    el.classList.add("opening");                   // portal grows: small window -> full screen
    setTimeout(() => { onDone.forEach((f) => f()); onDone = []; }, 450);
    setTimeout(() => { el.classList.add("gone"); document.documentElement.classList.remove("intro-on"); }, 1900);
    setTimeout(() => el.remove(), 2600);
  }
  requestAnimationFrame(tick);
  // never block the page for long: after 9 s reveal whatever has loaded
  setTimeout(() => { tasks.forEach((v) => (v[1] = 1)); recompute(); }, 9000);

  return {
    task(name, weight) { tasks.set(name, [weight, 0]); recompute(); },
    set(name, f) { const t = tasks.get(name); if (t) { t[1] = Math.max(t[1], Math.min(1, f)); recompute(); } },
    msg(text) { if (!finished) msgEl.textContent = text; },
    finish() { tasks.forEach((v) => (v[1] = 1)); recompute(); },
    then(f) { finished && !onDone.length ? f() : onDone.push(f); },
  };
}
