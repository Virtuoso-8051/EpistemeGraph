// Knowledge-graph explorer (corpus) and claim-graph view (audit), built on force-graph (canvas).
// Every label/tooltip is escaped: entity names and quotes come from LLM extraction.
import { esc } from "./util.js";

const FG = window.ForceGraph;

/** Evidence drawer of an entity: aliases, papers and every evidence-backed relation with its verbatim quote. */
function renderDrawer(drawer, n, onClose) {
  const rels = n.links.slice().sort((a, b) => b.grounding - a.grounding).map((l) => {
    const out = (l.source.id || l.source) === n.id;
    const other = out ? (l.target.id || l.target) : (l.source.id || l.source);
    return `<div class="rel"><span class="r">${out ? "→" : "←"} ${esc(l.rel)}</span> <b>${esc(other)}</b>
      <span class="muted mono" style="font-size:10.5px"> · g=${l.grounding} · ${esc(l.pdfs.join(", "))}</span>${l.quote ? `<q>${esc(l.quote)}</q>` : ""}</div>`;
  }).join("");
  drawer.innerHTML = `<button class="close" aria-label="close">×</button>
    <div class="kicker">${esc(n.type)} · degree ${n.degree}</div><h3>${esc(n.id)}</h3>
    <div class="chips" style="margin:8px 0">${n.pdfs.map((p) => `<span class="chip">${esc(p)}</span>`).join("")}</div>
    ${n.aliases.length ? `<div class="muted" style="font-size:12.5px;margin-bottom:8px">aka ${n.aliases.map(esc).join(" · ")}</div>` : ""}
    <div class="kicker" style="margin-top:14px">${n.links.length} evidence-backed relations</div>${rels}`;
  drawer.classList.add("open");
  drawer.querySelector(".close").onclick = onClose;
}
const BLUE = "#2a78d6", AQUA = "#1baf7a", INK = "#0e0e0c", MUTED = "#898781", ORANGE = "#eb6834", CRIT = "#d03b3b";

export class KGExplorer {
  constructor(el, drawer, onSelect) {
    this.el = el; this.drawer = drawer; this.onSelect = onSelect;
    this.hoverNode = null; this.focus = null; this.showLabels = false; this.pdfFilter = null;
    this.hlNodes = new Set(); this.hlLinks = new Set();
    this.fg = FG()(el)
      .backgroundColor("#fcfcfb")
      .nodeId("id")
      .nodeRelSize(2.6)
      .nodeVal((n) => 1 + Math.sqrt(n.degree) * 1.6)
      .nodeColor((n) => this.color(n))
      .nodeLabel((n) => `<div style="font:12px Inter,sans-serif"><b>${esc(n.id)}</b><br>${esc(n.type)} · degree ${n.degree}<br>${esc(n.pdfs.join(", "))}</div>`)
      .linkColor((l) => (this.hlLinks.has(l) ? INK : this.dim(l) ? "rgba(14,14,12,0.05)" : "rgba(14,14,12,0.16)"))
      .linkWidth((l) => (this.hlLinks.has(l) ? 1.6 : 0.6))
      .linkDirectionalArrowLength((l) => (this.hlLinks.has(l) ? 4 : 0))
      .linkDirectionalArrowRelPos(1)
      .linkLabel((l) => `<div style="font:12px Inter,sans-serif;max-width:320px"><b>${esc(l.rel)}</b> · grounding ${l.grounding}<br><i>${esc(l.quote)}</i></div>`)
      .nodeCanvasObjectMode(() => "after")
      .nodeCanvasObject((n, ctx, scale) => this.label(n, ctx, scale))
      .onNodeHover((n) => this.hover(n))
      .onNodeClick((n) => this.select(n))
      .onBackgroundClick(() => this.select(null))
      .warmupTicks(80)          // layout is pre-computed before the first frame, so fitting is valid at once
      .cooldownTicks(160)
      .d3VelocityDecay(0.32)
      .onEngineStop(() => { if (this.needFit) { this.needFit = false; this.fg.zoomToFit(500, 40); } });
    this.fg.d3Force("charge").strength(-70).distanceMax(260);
    this.fg.d3Force("link").distance(38);
    // KGs extracted from papers have many small components; a weak centering pull keeps them near the
    // main cluster instead of being flung to the edges (which would make "fit" shrink everything)
    this.fg.d3Force("x", window.d3.forceX(0).strength(0.07));
    this.fg.d3Force("y", window.d3.forceY(0).strength(0.07));
    new ResizeObserver(() => this.fg.width(el.clientWidth).height(el.clientHeight)).observe(el);
    new IntersectionObserver(([e]) => (e.isIntersecting ? this.fg.resumeAnimation() : this.fg.pauseAnimation())).observe(el);
  }

  color(n) {
    const base = n.pdfs.length > 1 ? AQUA : BLUE;
    if (this.pdfFilter && !n.pdfs.includes(this.pdfFilter)) return "rgba(137,135,129,0.25)";
    if (this.hlNodes.size && !this.hlNodes.has(n)) return "rgba(137,135,129,0.28)";
    return base;
  }
  dim(l) { return this.hlNodes.size && !this.hlLinks.has(l); }

  label(n, ctx, scale) {
    const show = this.showLabels ? n.degree >= 2 : n.degree >= 7;
    if (!(show || this.hlNodes.has(n)) || (this.hlNodes.size && !this.hlNodes.has(n))) return;
    const fs = Math.max(9 / scale, 2.4), r = Math.sqrt(1 + Math.sqrt(n.degree) * 1.6) * 3.2;
    ctx.font = `${n === this.hoverNode || n === this.focus ? 600 : 500} ${fs}px Inter, sans-serif`;
    const text = n.id.length > 34 ? n.id.slice(0, 32) + "…" : n.id;
    const w = ctx.measureText(text).width;
    ctx.fillStyle = "rgba(252,252,251,0.85)";
    ctx.fillRect(n.x - w / 2 - 2, n.y + r + 1, w + 4, fs + 3);
    ctx.fillStyle = INK; ctx.textAlign = "center"; ctx.textBaseline = "top";
    ctx.fillText(text, n.x, n.y + r + 2);
  }

  setData(data) {
    const byId = new Map(data.nodes.map((n) => [n.id, { ...n, neighbors: [], links: [] }]));
    const links = data.links.filter((l) => byId.has(l.source) && byId.has(l.target)).map((l) => ({ ...l }));
    links.forEach((l) => {
      const a = byId.get(l.source), b = byId.get(l.target);
      a.neighbors.push(b); b.neighbors.push(a); a.links.push(l); b.links.push(l);
    });
    this.nodes = [...byId.values()];
    this.needFit = true; // fit again when the layout has fully settled
    this.fg.graphData({ nodes: this.nodes, links });
    setTimeout(() => this.fg.zoomToFit(0, 40), 60);
  }

  refresh() { this.fg.nodeColor(this.fg.nodeColor()); this.fg.linkColor(this.fg.linkColor()); }

  highlight(n) {
    this.hlNodes.clear(); this.hlLinks.clear();
    if (n) { this.hlNodes.add(n); n.neighbors.forEach((m) => this.hlNodes.add(m)); n.links.forEach((l) => this.hlLinks.add(l)); }
    this.refresh();
  }
  hover(n) { this.hoverNode = n; this.el.style.cursor = n ? "pointer" : ""; if (!this.focus) this.highlight(n); }

  select(n) {
    this.focus = n;
    this.highlight(n);
    if (!n) { this.drawer.classList.remove("open"); return; }
    this.fg.centerAt(n.x, n.y, 700); this.fg.zoom(3.2, 700);
    renderDrawer(this.drawer, n, () => this.select(null));
    this.onSelect && this.onSelect(n);
  }

  find(q) {
    const s = q.trim().toLowerCase();
    if (!s || !this.nodes) return null;
    return this.nodes.find((n) => n.id.toLowerCase() === s) ||
      this.nodes.find((n) => n.id.toLowerCase().includes(s) || n.aliases.some((a) => a.toLowerCase() === s));
  }
  filterPdf(p) { this.pdfFilter = this.pdfFilter === p ? null : p; this.refresh(); return this.pdfFilter; }
  toggleLabels() { this.showLabels = !this.showLabels; this.refresh(); }
  fit() { this.select(null); this.fg.zoomToFit(600, 40); }
  // story API (2D equivalents)
  setColorMode() {}
  highlightSet(list) { this.hlNodes = new Set(list || []); this.hlLinks.clear(); this.refresh(); }
  flyTo(n) { if (n) { this.fg.centerAt(n.x, n.y, 700); this.fg.zoom(2.6, 700); } }
  overview() { this.fg.zoomToFit(700, 40); }
  autoRotate() {}
  callouts() {}
}

/**
 * 3D knowledge-graph observatory (3d-force-graph / three.js), styled for the dark "observatory" band.
 * Ivory = entity from one paper, gold = entity shared across papers (a bridge). Translucent links; hovered
 * neighbourhoods light up with particles flowing along the evidence edges. The scroll story drives it through
 * setColorMode / highlightSet / flyTo / overview / callouts; click opens the evidence drawer.
 */
const IVORY = "#e6e0d0", GOLD = "#f2b347", DIM = "#2b2a25", LINK = "#9b9483";
export class KG3D {
  constructor(el, drawer, opts = {}) {
    this.el = el; this.drawer = drawer;
    this.hl = new Set(); this.hlLinks = new Set(); this.focus = null; this.pdfFilter = null;
    this.mode = "default"; this.pdfColor = null; this.calloutNodes = [];
    const FG3 = window.ForceGraph3D;
    this.fg = FG3({ controlType: "orbit" })(el)
      .backgroundColor(opts.background || "#0b0b09")
      .showNavInfo(false)
      .nodeId("id")
      .nodeRelSize(3.4)
      .nodeVal((n) => 1 + Math.sqrt(n.degree) * 2.2)
      .nodeResolution(16)
      .nodeOpacity(0.96)
      .nodeColor((n) => this.color(n))
      .nodeLabel((n) => `<div class="kg-tip"><b>${esc(n.id)}</b><br>${esc(n.type)} · degree ${n.degree}<br>${esc(n.pdfs.join(", "))}</div>`)
      .linkColor((l) => (this.hlLinks.has(l) ? GOLD : LINK))
      .linkOpacity(0.22)
      .linkWidth((l) => (this.hlLinks.has(l) ? 1.3 : 0))
      .linkDirectionalParticles((l) => (this.hlLinks.has(l) ? 3 : 0))
      .linkDirectionalParticleWidth(2.4)
      .linkDirectionalParticleSpeed(0.007)
      .linkDirectionalParticleColor(() => GOLD)
      .linkLabel((l) => `<div class="kg-tip"><b>${esc(l.rel)}</b> · grounding ${l.grounding}<br><i>${esc(l.quote)}</i></div>`)
      .onNodeHover((n) => { el.style.cursor = n ? "pointer" : ""; if (!this.focus && !this.storyHold) this.highlight(n); })
      .onNodeClick((n) => this.select(n))
      .onBackgroundClick(() => this.select(null))
      .warmupTicks(80)
      .cooldownTicks(220);
    this.fg.d3Force("charge").strength(-55).distanceMax(220);
    this.fg.d3Force("link").distance(30);
    // weak pull towards the origin on all three axes: paper KGs have many small components which would
    // otherwise drift away and make "fit to view" shrink the main cluster
    const k = 0.09;
    let simNodes = [];
    const center3 = (alpha) => {
      for (const n of simNodes) { n.vx -= (n.x || 0) * k * alpha; n.vy -= (n.y || 0) * k * alpha; n.vz = (n.vz || 0) - (n.z || 0) * k * alpha; }
    };
    center3.initialize = (ns) => { simNodes = ns; };
    this.fg.d3Force("center3", center3);
    const c = this.fg.controls();
    if (c) {
      c.autoRotate = true; c.autoRotateSpeed = 0.5;
      el.addEventListener("pointerdown", () => { c.autoRotate = false; });
    }
    // HTML callout labels that follow their nodes on screen
    this.layer = document.createElement("div");
    this.layer.className = "kg-callouts";
    el.parentElement.appendChild(this.layer);
    this.visible = true;
    new ResizeObserver(() => this.fg.width(el.clientWidth).height(el.clientHeight)).observe(el);
    new IntersectionObserver(([e]) => {
      this.visible = e.isIntersecting;
      if (e.isIntersecting) { this.fg.resumeAnimation(); this.tickCallouts(); } else this.fg.pauseAnimation();
    }).observe(el);
  }

  color(n) {
    if (this.pdfFilter && !n.pdfs.includes(this.pdfFilter)) return DIM;
    if (this.hl.size && !this.hl.has(n)) return DIM;
    if (this.mode === "pdf" && this.pdfColor) return this.pdfColor.get(n.pdfs[0]) || IVORY;
    return n.pdfs.length > 1 ? GOLD : IVORY;
  }

  setData(data) {
    const byId = new Map(data.nodes.map((n) => [n.id, { ...n, neighbors: [], links: [] }]));
    const links = data.links.filter((l) => byId.has(l.source) && byId.has(l.target)).map((l) => ({ ...l }));
    links.forEach((l) => {
      const a = byId.get(l.source), b = byId.get(l.target);
      a.neighbors.push(b); b.neighbors.push(a); a.links.push(l); b.links.push(l);
    });
    this.nodes = [...byId.values()];
    this.fg.graphData({ nodes: this.nodes, links });
    setTimeout(() => this.fg.zoomToFit(900, 30), 400);
  }

  refresh() {
    this.fg.nodeColor(this.fg.nodeColor()).linkColor(this.fg.linkColor()).linkWidth(this.fg.linkWidth())
      .linkDirectionalParticles(this.fg.linkDirectionalParticles());
  }

  highlight(n) {
    this.hl.clear(); this.hlLinks.clear();
    if (n) { this.hl.add(n); n.neighbors.forEach((m) => this.hl.add(m)); n.links.forEach((l) => this.hlLinks.add(l)); }
    this.refresh();
  }

  // ---------------------------------------------------------------- story API
  setColorMode(mode, pdfColor) { this.mode = mode; this.pdfColor = pdfColor || this.pdfColor; this.refresh(); }
  highlightSet(list) {
    this.hl.clear(); this.hlLinks.clear();
    this.storyHold = !!(list && list.length);
    if (list && list.length) {
      list.forEach((n) => this.hl.add(n));
      list.forEach((n) => n.links.forEach((l) => { if (this.hl.has(l.source) && this.hl.has(l.target)) this.hlLinks.add(l); }));
    }
    this.refresh();
  }
  flyTo(n, near = 1) {
    if (!n) return;
    const d = 190 * near + 110, len = Math.hypot(n.x || 1, n.y || 1, n.z || 1);
    const r = 1 + d / len;
    this.fg.cameraPosition({ x: (n.x || 1) * r, y: (n.y || 1) * r, z: (n.z || 1) * r }, { x: n.x || 0, y: n.y || 0, z: n.z || 0 }, 1300);
  }
  overview(pad = 1) { this.fg.zoomToFit(1200, 30 / pad); }
  autoRotate(on) { const c = this.fg.controls(); if (c) c.autoRotate = !!on; }
  callouts(list) {
    this.calloutNodes = (list || []).slice(0, 6);
    this.layer.innerHTML = this.calloutNodes.map((n) => `<div class="kg-callout"><i></i><span>${esc(n.id)}</span><small>deg ${n.degree}</small></div>`).join("");
    this.tickCallouts();
  }
  tickCallouts() {
    if (this._co) return;
    const step = () => {
      this._co = 0;
      if (!this.visible || !this.calloutNodes.length) return;
      const W = this.el.clientWidth, H = this.el.clientHeight;
      [...this.layer.children].forEach((e, i) => {
        const n = this.calloutNodes[i];
        if (!n || n.x === undefined) return;
        const p = this.fg.graph2ScreenCoords(n.x, n.y, n.z);
        const out = p.x < 0 || p.y < 0 || p.x > W || p.y > H;
        e.style.transform = `translate(${p.x}px, ${p.y}px)`;
        e.style.opacity = out ? "0" : "1";
      });
      this._co = requestAnimationFrame(step);
    };
    this._co = requestAnimationFrame(step);
  }

  select(n) {
    this.focus = n;
    this.highlight(n);
    if (!n) { this.drawer.classList.remove("open"); return; }
    this.flyTo(n, 1.1);
    this.autoRotate(false);
    renderDrawer(this.drawer, n, () => this.select(null));
  }

  find(q) { return KGExplorer.prototype.find.call(this, q); }
  filterPdf(p) { this.pdfFilter = this.pdfFilter === p ? null : p; this.refresh(); return this.pdfFilter; }
  toggleLabels() { const c = this.fg.controls(); if (c) c.autoRotate = !c.autoRotate; }
  fit() { this.select(null); this.fg.zoomToFit(900, 30); }
}

/** Claim graph of ONE answer: blue = grounded in the KG, orange = only the LLM says so, red = conflicts. */
export function claimGraph(el, details) {
  el.innerHTML = "";
  if (!details || !details.length) { el.innerHTML = `<div class="empty" style="margin:16px">No claims extracted.</div>`; return null; }
  const nodes = new Map();
  const add = (name, aligned) => {
    const n = nodes.get(name) || { id: name, aligned: false };
    n.aligned = n.aligned || aligned; nodes.set(name, n);
  };
  const links = details.map((d) => {
    add(d.source, d.source_aligned); add(d.target, d.target_aligned);
    const color = d.conflict ? CRIT : d.claim_type === "REFUTED" ? MUTED : d.is_supported ? BLUE : ORANGE;
    return { source: d.source, target: d.target, rel: d.relation, color, d };
  });
  const fg = FG()(el).backgroundColor("#fcfcfb").width(el.clientWidth).height(el.clientHeight)
    .graphData({ nodes: [...nodes.values()], links })
    // nodes and labels are drawn at a constant SCREEN size, so auto-fitting a small graph never balloons them
    .nodeCanvasObjectMode(() => "replace")
    .nodeCanvasObject((n, ctx, scale) => {
      const r = 6 / scale;
      ctx.beginPath(); ctx.arc(n.x, n.y, r, 0, 2 * Math.PI);
      ctx.fillStyle = n.aligned ? INK : "#ffffff"; ctx.fill();
      ctx.strokeStyle = INK; ctx.lineWidth = 1.4 / scale; ctx.stroke();
      const fs = 11.5 / scale;
      ctx.font = `500 ${fs}px Inter, sans-serif`; ctx.textAlign = "center"; ctx.textBaseline = "top";
      const t = n.id.length > 30 ? n.id.slice(0, 28) + "…" : n.id, w = ctx.measureText(t).width;
      ctx.fillStyle = "rgba(252,252,251,0.88)"; ctx.fillRect(n.x - w / 2 - 2 / scale, n.y + r + 2 / scale, w + 4 / scale, fs + 3 / scale);
      ctx.fillStyle = INK; ctx.fillText(t, n.x, n.y + r + 3 / scale);
    })
    .nodePointerAreaPaint((n, color, ctx, scale) => { ctx.fillStyle = color; ctx.beginPath(); ctx.arc(n.x, n.y, 9 / scale, 0, 2 * Math.PI); ctx.fill(); })
    .nodeLabel((n) => `<div style="font:12px Inter,sans-serif"><b>${esc(n.id)}</b><br>${n.aligned ? "aligned to a KG entity" : "not in the knowledge graph"}</div>`)
    .linkColor((l) => l.color).linkWidth(2).linkDirectionalArrowLength(5).linkDirectionalArrowRelPos(0.9)
    .linkLineDash((l) => (l.d.claim_type === "REFUTED" ? [3, 3] : null))
    .linkLabel((l) => `<div style="font:12px Inter,sans-serif;max-width:300px"><b>${esc(l.rel)}</b> · ${esc(l.d.support_tier)}${l.d.conflict ? " · " + esc(l.d.conflict) : ""}${l.d.evidence_quote ? `<br><i>${esc(l.d.evidence_quote)}</i>` : ""}</div>`)
    .warmupTicks(150).cooldownTicks(60);
  fg.d3Force("charge").strength(-260);
  fg.d3Force("link").distance(70);
  const fit = () => { fg.zoomToFit(0, 56); if (fg.zoom() > 3) fg.zoom(3); };
  setTimeout(fit, 60);
  let fitted = false;
  fg.onEngineStop(() => { if (!fitted) { fitted = true; fit(); } });
  new ResizeObserver(() => fg.width(el.clientWidth).height(el.clientHeight)).observe(el);
  return fg;
}
