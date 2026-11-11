// Shared helpers. Everything that originates from an LLM (answers, entity names, quotes) is
// rendered through esc() - never interpolate raw strings into innerHTML.

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

export function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

export const isNum = (x) => x !== null && x !== undefined && x !== "" && !Number.isNaN(Number(x));
export const pct = (x, d = 1) => (isNum(x) ? `${(100 * Number(x)).toFixed(d)}%` : "—");
export const num = (x, d = 3) => (isNum(x) ? Number(x).toFixed(d) : "—");
export const int = (x) => (isNum(x) ? Number(x).toLocaleString() : "—");
export const ago = (iso) => {
  if (!iso) return "";
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return new Date(iso).toLocaleDateString();
};

export const STATUS = {
  VERIFIED: { color: "#0ca30c", icon: "✓", label: "Verified", text: "Every asserted claim is backed by the knowledge graph at the calibrated risk level." },
  ABSTAIN: { color: "#fab219", icon: "◐", label: "Abstain", text: "The evidence is ambiguous at this risk level: route to a human or ask for sources." },
  HALLUCINATED: { color: "#d03b3b", icon: "✕", label: "Hallucinated", text: "The answer makes claims the source papers do not support." },
  UNCALIBRATED: { color: "#898781", icon: "○", label: "Uncalibrated", text: "Features and claim-level evidence are computed; run the benchmark once to train the calibrated decision layer." },
};
export const badge = (d) => `<span class="badge b-${esc(d || "UNCALIBRATED")}">${(STATUS[d] || STATUS.UNCALIBRATED).icon} ${esc((STATUS[d] || STATUS.UNCALIBRATED).label)}</span>`;

// access settings live only in this browser (localStorage), never on the server
export function getSetting(k) { try { return localStorage.getItem(`eg.${k}`) || ""; } catch (_) { return ""; } }
export function setSetting(k, v) { try { v ? localStorage.setItem(`eg.${k}`, v) : localStorage.removeItem(`eg.${k}`); } catch (_) { /* storage blocked */ } }

export async function api(path, opts = {}) {
  const init = { ...opts };
  const headers = {};
  if (opts.json !== undefined) {
    init.method = init.method || "POST";
    headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(opts.json);
    delete init.json;
  }
  if (getSetting("admin")) headers["X-Admin-Token"] = getSetting("admin");
  if (getSetting("openai") && path.startsWith("/api/audit")) headers["X-LLM-Key"] = getSetting("openai");
  init.headers = headers;
  const ms = opts.timeout || (init.method && init.method !== "GET" ? 120000 : 30000);
  delete init.timeout;
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), ms);
  init.signal = ctl.signal;
  let r;
  try { r = await fetch(path, init); }
  catch (e) { throw new Error(e.name === "AbortError" ? `request timed out after ${Math.round(ms / 1000)} s` : `network error: ${e.message}`); }
  finally { clearTimeout(timer); }
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try { const j = await r.json(); msg = j.detail || msg; } catch (_) { /* not json */ }
    throw new Error(msg);
  }
  const ct = r.headers.get("content-type") || "";
  return ct.includes("application/json") ? r.json() : r.text();
}

let toastTimer;
export function toast(msg, err = false) {
  document.querySelectorAll(".toast").forEach((t) => t.remove());
  const t = document.createElement("div");
  t.className = "toast" + (err ? " err" : "");
  t.textContent = msg;
  document.body.appendChild(t);
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.remove(), err ? 6000 : 3500);
}

const tip = () => document.getElementById("tooltip");
export function showTip(html, ev) {
  const t = tip();
  t.innerHTML = html;
  t.style.opacity = 1;
  const pad = 14, w = t.offsetWidth, h = t.offsetHeight;
  let x = ev.clientX + pad, y = ev.clientY + pad;
  if (x + w > innerWidth - 8) x = ev.clientX - w - pad;
  if (y + h > innerHeight - 8) y = ev.clientY - h - pad;
  t.style.left = `${x}px`;
  t.style.top = `${y}px`;
}
export const hideTip = () => { tip().style.opacity = 0; };

export function tex(el, expr, display = true) {
  try { window.katex.render(expr, el, { displayMode: display, throwOnError: false }); }
  catch (_) { el.textContent = expr; }
}

export function countUp(el, to, fmt = (v) => v.toFixed(0), ms = 1100) {
  if (!isNum(to)) { el.textContent = "—"; return; }
  const target = Number(to);
  if (document.documentElement.classList.contains("static") || matchMedia("(prefers-reduced-motion: reduce)").matches) {
    el.textContent = fmt(target); return;
  }
  const t0 = performance.now();
  const step = (t) => {
    const k = Math.min(1, (t - t0) / ms), e = 1 - Math.pow(1 - k, 3);
    el.textContent = fmt(target * e);
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

export function segValue(id) { return document.querySelector(`#${id} button.on`)?.dataset.v; }
export function bindSeg(id, onChange) {
  document.querySelectorAll(`#${id} button`).forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll(`#${id} button`).forEach((x) => x.classList.toggle("on", x === b));
    onChange && onChange(b.dataset.v);
  }));
}
