// Knowledge-graph story: the 3D graph stays pinned while chapters scroll past it (corpus, hubs, bridges,
// evidence, explore). Every chapter is computed from the REAL graph; entering a chapter re-stages the graph
// (colour mode, highlighted set, camera), and hovering an item flies the camera to that entity.
import { esc, int } from "./util.js";

const PDF_COLORS = ["#f2b347", "#7fb2ff", "#7ed3a8", "#ff8a65", "#c9a7ff", "#f07fb5", "#a8d672"];

export function buildStory(root, kg, data, state) {
  if (!root || !kg || !data) return;
  if (root._io) root._io.disconnect();
  const nodes = kg.nodes || [];
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const pdfs = [...new Set(nodes.flatMap((n) => n.pdfs))].sort();
  const pdfColor = new Map(pdfs.map((p, i) => [p, PDF_COLORS[i % PDF_COLORS.length]]));
  const hubs = nodes.slice().sort((a, b) => b.degree - a.degree).slice(0, 7);
  const shared = nodes.filter((n) => n.pdfs.length > 1).sort((a, b) => b.degree - a.degree);
  const ev = (data.links || []).slice().filter((l) => l.quote).sort((a, b) => (b.grounding - a.grounding) || (b.support - a.support)).slice(0, 5);
  const kgs = state.kg || {};

  const set = (sel, html) => { const el = root.querySelector(sel); if (el) el.innerHTML = html; };
  set("[data-ch=corpus] .ch-big", `${pdfs.length}<small> papers</small>`);
  set("[data-ch=corpus] .ch-list", pdfs.map((p, i) => {
    const ents = nodes.filter((n) => n.pdfs.includes(p)).length;
    return `<li data-pdf="${esc(p)}"><span class="num" style="color:${pdfColor.get(p)}">${String(i + 1).padStart(2, "0")}</span>
      <div><b>${esc(p.replace(/\.pdf$/i, ""))}</b><small>${int(ents)} entities in the graph</small></div></li>`;
  }).join(""));
  set("[data-ch=corpus] .ch-note", `${int(kgs.verified_triplets_pushed)} verified triples · ${int(kgs.resolved_entity_count)} canonical entities · every edge carries a verbatim quote from its source chunk.`);

  set("[data-ch=hubs] .ch-list", hubs.map((n) => `<li data-node="${esc(n.id)}"><span class="num">${n.degree}</span>
      <div><b>${esc(n.id)}</b><small>${esc(n.type)} · ${esc(n.pdfs.map((p) => p.replace(/\.pdf$/i, "")).join(" · "))}</small></div></li>`).join(""));

  set("[data-ch=bridges] .ch-big", `${shared.length}<small> bridges</small>`);
  set("[data-ch=bridges] .ch-list", shared.slice(0, 6).map((n, i) => `<li data-node="${esc(n.id)}"><span class="num">${String(i + 1).padStart(2, "0")}</span>
      <div><b>${esc(n.id)}</b><small>appears in ${n.pdfs.length} papers · degree ${n.degree}</small></div></li>`).join("") ||
      `<li><div><b>No entity is shared yet</b><small>add a second paper to the corpus to see bridges</small></div></li>`);

  set("[data-ch=evidence] .ch-list", ev.map((l) => {
    const s = l.source.id || l.source, t = l.target.id || l.target;
    return `<li data-link="${esc(s)}|${esc(t)}"><span class="num">${(+l.grounding).toFixed(2)}</span>
      <div><b>${esc(s)} <i class="rel">${esc(l.rel)}</i> ${esc(t)}</b><q>${esc(String(l.quote).slice(0, 180))}</q></div></li>`;
  }).join(""));

  // ---- staging
  const stage = {
    corpus: () => { kg.setColorMode("pdf", pdfColor); kg.highlightSet(null); kg.overview(); kg.callouts(hubs.slice(0, 3)); },
    hubs: () => { kg.setColorMode("default"); kg.highlightSet(null); if (hubs[0]) kg.flyTo(hubs[0], 1); kg.callouts(hubs.slice(0, 5)); },
    bridges: () => { kg.setColorMode("default"); kg.highlightSet(shared); kg.overview(0.85); kg.callouts(shared.slice(0, 5)); },
    evidence: () => { kg.setColorMode("default"); const l = ev[0]; if (l) focusLink(l); },
    explore: () => { kg.setColorMode("default"); kg.highlightSet(null); kg.callouts([]); kg.overview(); kg.autoRotate(true); },
  };
  function focusLink(l) {
    const a = byId.get(l.source.id || l.source), b = byId.get(l.target.id || l.target);
    if (!a || !b) return;
    kg.highlightSet([a, b]);
    kg.flyTo(a, 1.3);
    kg.callouts([a, b]);
  }
  let active = null;
  const chapters = [...root.querySelectorAll(".chapter[data-ch]")];
  const io = new IntersectionObserver((es) => es.forEach((e) => {
    if (!e.isIntersecting) return;
    const ch = e.target.dataset.ch;
    if (ch === active) return;
    active = ch;
    chapters.forEach((c) => c.classList.toggle("on", c === e.target));
    root.dataset.stage = ch;
    kg.autoRotate(ch === "explore" || ch === "corpus");
    (stage[ch] || stage.explore)();
  }), { rootMargin: "-42% 0px -42% 0px" });
  chapters.forEach((c) => io.observe(c));
  root._io = io;

  // ---- item interaction
  root.querySelectorAll("[data-node]").forEach((li) => {
    const n = byId.get(li.dataset.node);
    if (!n) return;
    li.addEventListener("mouseenter", () => { kg.highlightSet([n, ...n.neighbors]); kg.flyTo(n, 1); kg.callouts([n]); });
    li.addEventListener("click", () => kg.select(n));
  });
  root.querySelectorAll("[data-pdf]").forEach((li) => {
    li.addEventListener("mouseenter", () => kg.highlightSet(nodes.filter((n) => n.pdfs.includes(li.dataset.pdf))));
    li.addEventListener("mouseleave", () => kg.highlightSet(null));
  });
  root.querySelectorAll("[data-link]").forEach((li, i) => li.addEventListener("mouseenter", () => focusLink(ev[i])));
}
