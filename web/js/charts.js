// D3 charts (light palette). Conventions: thin marks, recessive grid, rounded data ends, hover
// tooltip on every mark, legend whenever there are >= 2 series, text in ink tokens (never series colour).
import { esc, hideTip, num, pct, showTip } from "./util.js";

const d3 = window.d3;
export const C = {
  blue: "#2a78d6", orange: "#eb6834", aqua: "#1baf7a", neg: "#e34948", ink: "#0e0e0c", ink2: "#52514e",
  muted: "#898781", grid: "#e1e0d9", axis: "#c3c2b7", surface: "#fcfcfb", gray: "#c3c2b7",
  good: "#0ca30c", warn: "#fab219", crit: "#d03b3b",
};

// Charts are drawn in real pixels for the container's CURRENT width and redrawn when it changes,
// so text is never stretched or clipped (an SVG drawn narrow and scaled up enlarges its labels).
const RO = new ResizeObserver((entries) => entries.forEach((e) => {
  const el = e.target, w = Math.round(e.contentRect.width);
  if (el.__redraw && el.__w && Math.abs(w - el.__w) > 24) el.__redraw();
}));
function remember(el, redraw) { el.__redraw = redraw; el.__w = Math.round(el.clientWidth); RO.observe(el); }

function frame(el, height, m) {
  el.innerHTML = "";
  const W = Math.max(280, el.clientWidth || 600);
  // drawn at true pixel size (width/height attributes): text is never magnified; CSS max-width only shrinks
  const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${W} ${height}`).attr("width", W).attr("height", height).attr("role", "img");
  return { svg, W, H: height, iw: W - m.l - m.r, ih: height - m.t - m.b, g: svg.append("g").attr("transform", `translate(${m.l},${m.t})`) };
}

export function legend(el, items) {
  const d = document.createElement("div");
  d.className = "legend";
  d.innerHTML = items.map((i) => `<span><i class="${i.sq ? "sq" : ""}" style="background:${i.color}"></i>${esc(i.label)}</span>`).join("");
  el.prepend(d);
}

/** Horizontal bars: data [{label, value, color?, tip?}] */
export function hbar(el, data, opts = {}) {
  remember(el, () => hbar(el, data, opts));
  const { fmt = (v) => num(v, 3), max = null, height = null, highlight = null } = opts;
  const m = { t: 6, r: 56, b: 10, l: Math.min(200, 12 + 7 * d3.max(data, (d) => d.label.length)) };
  const h = height || data.length * 30 + m.t + m.b;
  const { g, iw, ih } = frame(el, h, m);
  const x = d3.scaleLinear().domain([0, max ?? ((d3.max(data, (d) => d.value) * 1.05) || 1)]).range([0, iw]);
  const y = d3.scaleBand().domain(data.map((d) => d.label)).range([0, ih]).padding(0.35);
  g.selectAll("text.lbl").data(data).join("text").attr("class", "lbl").attr("x", -10).attr("y", (d) => y(d.label) + y.bandwidth() / 2)
    .attr("dy", "0.35em").attr("text-anchor", "end").text((d) => d.label);
  g.selectAll("rect").data(data).join("rect").attr("x", 0).attr("y", (d) => y(d.label)).attr("height", y.bandwidth()).attr("rx", 3)
    .attr("fill", (d) => d.color || (highlight && d.label === highlight ? C.blue : C.gray))
    .attr("width", (d) => Math.max(2, x(d.value))).attr("class", "grow-l")
    .on("mousemove", (ev, d) => showTip(d.tip || `<b>${esc(d.label)}</b><br>${fmt(d.value)}`, ev)).on("mouseleave", hideTip);
  g.selectAll("text.val").data(data).join("text").attr("class", "val").attr("x", (d) => x(d.value) + 6)
    .attr("y", (d) => y(d.label) + y.bandwidth() / 2).attr("dy", "0.35em").text((d) => fmt(d.value));
}

/** Diverging bars around 0: positive = blue (toward supported), negative = red. */
export function diverging(el, data, opts = {}) {
  remember(el, () => diverging(el, data, opts));
  const { fmt = (v) => (v >= 0 ? "+" : "") + num(v, 2), leftLabel = "", rightLabel = "" } = opts;
  const m = { t: 6, r: 46, b: 8, l: Math.min(190, 12 + 6.8 * d3.max(data, (d) => d.label.length)) };
  const { g, iw, ih } = frame(el, data.length * 30 + m.t + m.b, m);
  const ext = d3.max(data, (d) => Math.abs(d.value)) || 1;
  const x = d3.scaleLinear().domain([-ext * 1.1, ext * 1.1]).range([0, iw]);
  const y = d3.scaleBand().domain(data.map((d) => d.label)).range([0, ih]).padding(0.32);
  if (leftLabel || rightLabel) {
    const lab = document.createElement("div");
    lab.className = "legend";
    lab.style.cssText = `justify-content:space-between;margin-left:${m.l}px;margin-right:${m.r}px;font-family:var(--mono);font-size:11px`;
    lab.innerHTML = `<span style="color:${C.neg}">${esc(leftLabel)}</span><span style="color:${C.blue}">${esc(rightLabel)}</span>`;
    el.prepend(lab);
  }
  g.append("line").attr("x1", x(0)).attr("x2", x(0)).attr("y1", -4).attr("y2", ih).attr("stroke", C.axis);
  g.selectAll("text.lbl").data(data).join("text").attr("class", "lbl").attr("x", -10).attr("y", (d) => y(d.label) + y.bandwidth() / 2)
    .attr("dy", "0.35em").attr("text-anchor", "end").text((d) => d.label);
  g.selectAll("rect").data(data).join("rect").attr("y", (d) => y(d.label)).attr("height", y.bandwidth()).attr("rx", 3)
    .attr("fill", (d) => (d.value >= 0 ? C.blue : C.neg))
    .attr("x", (d) => (d.value >= 0 ? x(0) : x(d.value))).attr("width", (d) => Math.max(1, Math.abs(x(d.value) - x(0))))
    .attr("class", (d) => (d.value >= 0 ? "grow-l" : "grow-r"))
    .on("mousemove", (ev, d) => showTip(d.tip || `<b>${esc(d.label)}</b><br>${fmt(d.value)}`, ev)).on("mouseleave", hideTip);
  // value labels: outside short bars; INSIDE long bars (so negative labels never collide with category names)
  const inside = (d) => Math.abs(x(d.value) - x(0)) > 48;
  g.selectAll("text.val.v").data(data).join("text").attr("class", "val v").attr("y", (d) => y(d.label) + y.bandwidth() / 2).attr("dy", "0.35em")
    .attr("x", (d) => (d.value >= 0 ? (inside(d) ? x(d.value) - 5 : x(d.value) + 5) : (inside(d) ? x(d.value) + 5 : x(d.value) - 5)))
    .attr("text-anchor", (d) => ((d.value >= 0) !== inside(d) ? "start" : "end"))
    .style("fill", (d) => (inside(d) ? "#ffffff" : null)).text((d) => fmt(d.value));
}

function xyAxes(g, x, y, iw, ih, xl, yl) {
  g.append("g").attr("class", "grid").selectAll("line").data(y.ticks(5)).join("line").attr("class", "gridline")
    .attr("x1", 0).attr("x2", iw).attr("y1", (d) => y(d)).attr("y2", (d) => y(d));
  g.append("g").attr("class", "axis").attr("transform", `translate(0,${ih})`).call(d3.axisBottom(x).ticks(5).tickSizeOuter(0));
  g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5).tickSizeOuter(0));
  g.append("text").attr("class", "val").attr("x", iw).attr("y", ih + 34).attr("text-anchor", "end").text(xl);
  g.append("text").attr("class", "val").attr("x", 0).attr("y", -10).text(yl);
}

/** Line chart with crosshair tooltips. series [{name, color, points:[{x,y,n?}]}] */
export function lines(el, series, opts = {}) {
  remember(el, () => lines(el, series, opts));
  const { xLabel = "", yLabel = "", diagonal = true, height = 300, markers = false } = opts;
  const m = { t: 22, r: 16, b: 42, l: 42 };
  const { g, iw, ih, svg } = frame(el, height, m);
  const x = d3.scaleLinear().domain([0, 1]).range([0, iw]);
  const y = d3.scaleLinear().domain([0, 1]).range([ih, 0]);
  xyAxes(g, x, y, iw, ih, xLabel, yLabel);
  if (diagonal) g.append("line").attr("x1", x(0)).attr("y1", y(0)).attr("x2", x(1)).attr("y2", y(1)).attr("stroke", C.axis).attr("stroke-dasharray", "3 4");
  const ln = d3.line().x((p) => x(p.x)).y((p) => y(p.y));
  series.forEach((s) => {
    const path = g.append("path").datum(s.points).attr("fill", "none").attr("stroke", s.color).attr("stroke-width", 2).attr("d", ln);
    const L = path.node().getTotalLength();
    path.attr("class", "draw").style("--len", L);
    if (markers) g.selectAll(null).data(s.points).join("circle").attr("cx", (p) => x(p.x)).attr("cy", (p) => y(p.y)).attr("r", 4.5)
      .attr("fill", s.color).attr("stroke", C.surface).attr("stroke-width", 2);
  });
  const cross = g.append("line").attr("y1", 0).attr("y2", ih).attr("stroke", C.ink2).attr("stroke-width", 1).attr("opacity", 0);
  const dots = series.map((s) => g.append("circle").attr("r", 4.5).attr("fill", s.color).attr("stroke", C.surface).attr("stroke-width", 2).attr("opacity", 0));
  svg.append("rect").attr("x", m.l).attr("y", m.t).attr("width", iw).attr("height", ih).attr("fill", "transparent")
    .on("mousemove", (ev) => {
      const [px] = d3.pointer(ev, g.node());
      const xv = x.invert(px);
      cross.attr("x1", px).attr("x2", px).attr("opacity", 0.4);
      const rows = series.map((s, i) => {
        const p = s.points.reduce((a, b) => (Math.abs(b.x - xv) < Math.abs(a.x - xv) ? b : a), s.points[0]);
        dots[i].attr("cx", x(p.x)).attr("cy", y(p.y)).attr("opacity", 1);
        return `<span style="color:${s.color}">●</span> ${esc(s.name)}: ${num(p.y, 2)}${p.n ? ` <span style="opacity:.7">(n=${p.n})</span>` : ""}`;
      });
      showTip(`<b>${esc(xLabel)} ${num(xv, 2)}</b><br>${rows.join("<br>")}`, ev);
    })
    .on("mouseleave", () => { hideTip(); cross.attr("opacity", 0); dots.forEach((d) => d.attr("opacity", 0)); });
  if (series.length > 1) legend(el, series.map((s) => ({ label: s.name, color: s.color })));
}

export function rocPoints(labels, scores) {
  const pairs = labels.map((l, i) => [scores[i], l]).filter((p) => p[0] !== null && !Number.isNaN(p[0])).sort((a, b) => b[0] - a[0]);
  const P = pairs.filter((p) => p[1] === 1).length, N = pairs.length - P;
  if (!P || !N) return { points: [], auc: NaN };
  let tp = 0, fp = 0, auc = 0, prevF = 0, prevT = 0;
  const pts = [{ x: 0, y: 0 }];
  pairs.forEach(([, l], i) => {
    if (l === 1) tp++; else fp++;
    if (i === pairs.length - 1 || pairs[i + 1][0] !== pairs[i][0]) {
      const f = fp / N, t = tp / P;
      auc += (f - prevF) * (t + prevT) / 2;
      prevF = f; prevT = t;
      pts.push({ x: f, y: t });
    }
  });
  return { points: pts, auc };
}

export function reliabilityPoints(labels, scores, bins = 10) {
  const acc = Array.from({ length: bins }, () => ({ s: 0, y: 0, n: 0 }));
  scores.forEach((p, i) => {
    if (p === null || Number.isNaN(p)) return;
    const b = Math.min(bins - 1, Math.max(0, Math.floor(p * bins)));
    acc[b].s += p; acc[b].y += labels[i]; acc[b].n++;
  });
  return acc.filter((a) => a.n).map((a) => ({ x: a.s / a.n, y: a.y / a.n, n: a.n }));
}

/** Horizontal stacked bars. rows [{label, parts:{key:count}}], keys [{key,label,color}] */
export function stacked(el, rows, keys) {
  remember(el, () => stacked(el, rows, keys));
  const m = { t: 8, r: 40, b: 10, l: Math.min(170, 12 + 7.2 * d3.max(rows, (d) => d.label.length)) };
  const { g, iw, ih } = frame(el, rows.length * 34 + m.t + m.b, m);
  const totals = rows.map((r) => keys.reduce((s, k) => s + (r.parts[k.key] || 0), 0));
  const x = d3.scaleLinear().domain([0, d3.max(totals) || 1]).range([0, iw]);
  const y = d3.scaleBand().domain(rows.map((r) => r.label)).range([0, ih]).padding(0.3);
  rows.forEach((r, i) => {
    let acc = 0;
    keys.forEach((k) => {
      const v = r.parts[k.key] || 0;
      if (!v) return;
      g.append("rect").attr("x", x(acc) + 1).attr("y", y(r.label)).attr("height", y.bandwidth()).attr("width", Math.max(1, x(v) - x(0) - 2)).attr("rx", 3)
        .attr("fill", k.color).on("mousemove", (ev) => showTip(`<b>${esc(r.label)}</b><br>${esc(k.label)}: ${v} of ${totals[i]} (${pct(v / totals[i], 0)})`, ev)).on("mouseleave", hideTip);
      acc += v;
    });
    g.append("text").attr("class", "val").attr("x", x(totals[i]) + 6).attr("y", y(r.label) + y.bandwidth() / 2).attr("dy", "0.35em").text(totals[i]);
  });
  g.selectAll("text.lbl").data(rows).join("text").attr("class", "lbl").attr("x", -10).attr("y", (d) => y(d.label) + y.bandwidth() / 2)
    .attr("dy", "0.35em").attr("text-anchor", "end").text((d) => d.label);
  legend(el, keys.map((k) => ({ label: k.label, color: k.color, sq: true })));
}

/** Horizontal box plots: groups [{name, color, values}] */
export function boxes(el, groups, opts = {}) {
  remember(el, () => boxes(el, groups, opts));
  const { domain = null, fmt = (v) => num(v, 3) } = opts;
  const m = { t: 8, r: 18, b: 30, l: 110 };
  const { g, iw, ih } = frame(el, groups.length * 52 + m.t + m.b, m);
  const all = groups.flatMap((gr) => gr.values);
  const x = d3.scaleLinear().domain(domain || d3.extent(all)).nice().range([0, iw]);
  const y = d3.scaleBand().domain(groups.map((gr) => gr.name)).range([0, ih]).padding(0.35);
  g.append("g").attr("class", "axis").attr("transform", `translate(0,${ih})`).call(d3.axisBottom(x).ticks(5).tickSizeOuter(0));
  groups.forEach((gr) => {
    const v = gr.values.slice().sort(d3.ascending);
    if (!v.length) return;
    const q1 = d3.quantile(v, 0.25), q2 = d3.quantile(v, 0.5), q3 = d3.quantile(v, 0.75), iqr = q3 - q1;
    const lo = d3.min(v.filter((t) => t >= q1 - 1.5 * iqr)), hi = d3.max(v.filter((t) => t <= q3 + 1.5 * iqr));
    const cy = y(gr.name) + y.bandwidth() / 2, bh = y.bandwidth();
    const tipH = `<b>${esc(gr.name)}</b> (n=${v.length})<br>median ${fmt(q2)} · IQR ${fmt(q1)}–${fmt(q3)}`;
    g.append("line").attr("x1", x(lo)).attr("x2", x(hi)).attr("y1", cy).attr("y2", cy).attr("stroke", C.ink2);
    g.append("rect").attr("x", x(q1)).attr("width", Math.max(1, x(q3) - x(q1))).attr("y", cy - bh / 2).attr("height", bh).attr("rx", 4)
      .attr("fill", gr.color).attr("fill-opacity", 0.22).attr("stroke", gr.color).attr("stroke-width", 1.5)
      .on("mousemove", (ev) => showTip(tipH, ev)).on("mouseleave", hideTip);
    g.append("line").attr("x1", x(q2)).attr("x2", x(q2)).attr("y1", cy - bh / 2).attr("y2", cy + bh / 2).attr("stroke", gr.color).attr("stroke-width", 2.5);
    g.selectAll(null).data(v.filter((t) => t < lo || t > hi)).join("circle").attr("cx", (t) => x(t)).attr("cy", cy).attr("r", 2.5).attr("fill", gr.color);
    g.append("text").attr("class", "lbl").attr("x", -10).attr("y", cy).attr("dy", "0.35em").attr("text-anchor", "end").text(gr.name);
  });
}

/** Forest plot: rows [{label, value, lo?, hi?, color?, tip?}] with an optional reference line. */
export function forest(el, rows, opts = {}) {
  remember(el, () => forest(el, rows, opts));
  const { ref = null, refLabel = "", fmt = (v) => num(v, 3), domain = null } = opts;
  const m = { t: 14, r: 60, b: 30, l: Math.min(230, 14 + 6.8 * d3.max(rows, (d) => d.label.length)) };
  const { g, iw, ih } = frame(el, rows.length * 28 + m.t + m.b, m);
  const vals = rows.flatMap((r) => [r.value, r.lo ?? r.value, r.hi ?? r.value]).concat(ref ?? []);
  const ext = domain || d3.extent(vals);
  const pad = (ext[1] - ext[0]) * 0.08 || 0.05;
  const x = d3.scaleLinear().domain([ext[0] - pad, ext[1] + pad]).nice().range([0, iw]);
  const y = d3.scaleBand().domain(rows.map((r) => r.label)).range([0, ih]).padding(0.4);
  g.append("g").attr("class", "axis").attr("transform", `translate(0,${ih})`).call(d3.axisBottom(x).ticks(6).tickSizeOuter(0));
  g.append("g").selectAll("line").data(x.ticks(6)).join("line").attr("class", "gridline").attr("x1", (d) => x(d)).attr("x2", (d) => x(d)).attr("y1", 0).attr("y2", ih);
  if (ref !== null) {
    g.append("line").attr("x1", x(ref)).attr("x2", x(ref)).attr("y1", -6).attr("y2", ih).attr("stroke", C.ink2).attr("stroke-dasharray", "3 3");
    if (refLabel) g.append("text").attr("class", "val").attr("x", x(ref) + 4).attr("y", -2).text(refLabel);
  }
  rows.forEach((r) => {
    const cy = y(r.label) + y.bandwidth() / 2, col = r.color || C.blue;
    if (r.lo !== undefined && r.hi !== undefined && r.lo !== null) {
      g.append("line").attr("x1", x(r.lo)).attr("x2", x(r.hi)).attr("y1", cy).attr("y2", cy).attr("stroke", col).attr("stroke-width", 2);
      [r.lo, r.hi].forEach((v) => g.append("line").attr("x1", x(v)).attr("x2", x(v)).attr("y1", cy - 5).attr("y2", cy + 5).attr("stroke", col).attr("stroke-width", 2));
    }
    g.append("circle").attr("cx", x(r.value)).attr("cy", cy).attr("r", 5.5).attr("fill", col).attr("stroke", C.surface).attr("stroke-width", 2)
      .on("mousemove", (ev) => showTip(r.tip || `<b>${esc(r.label)}</b><br>${fmt(r.value)}${r.lo !== undefined && r.lo !== null ? ` [${fmt(r.lo)}, ${fmt(r.hi)}]` : ""}`, ev))
      .on("mouseleave", hideTip);
    g.append("text").attr("class", "val").attr("x", iw + 6).attr("y", cy).attr("dy", "0.35em").text(fmt(r.value));
  });
  g.selectAll("text.lbl").data(rows).join("text").attr("class", "lbl").attr("x", -10).attr("y", (d) => y(d.label) + y.bandwidth() / 2)
    .attr("dy", "0.35em").attr("text-anchor", "end").text((d) => d.label);
}

/** Semicircle gauge 0..1 */
export function gauge(el, value, { color = C.crit, label = "", sub = "" } = {}) {
  el.innerHTML = "";
  const W = 240, H = 140, r = 96, cx = W / 2, cy = 116;
  const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${W} ${H}`);
  const arc = d3.arc().innerRadius(r - 14).outerRadius(r).cornerRadius(7);
  const g = svg.append("g").attr("transform", `translate(${cx},${cy})`);
  g.append("path").attr("d", arc({ startAngle: -Math.PI / 2, endAngle: Math.PI / 2 })).attr("fill", "#ebe8df");
  const v = value === null || value === undefined || Number.isNaN(value) ? null : Math.max(0, Math.min(1, value));
  if (v !== null) {
    g.append("path").attr("fill", color).attr("class", "fade-in").attr("d", arc({ startAngle: -Math.PI / 2, endAngle: -Math.PI / 2 + Math.PI * v }));
  }
  g.append("text").attr("text-anchor", "middle").attr("y", -18).attr("font-family", "Space Grotesk, sans-serif").attr("font-weight", 700)
    .attr("font-size", 38).attr("fill", C.ink).text(v === null ? "—" : `${Math.round(100 * v)}%`);
  g.append("text").attr("text-anchor", "middle").attr("y", 4).attr("font-family", "JetBrains Mono, monospace").attr("font-size", 9.5)
    .attr("fill", C.muted).text(label.toUpperCase());
  if (sub) g.append("text").attr("text-anchor", "middle").attr("y", 18).attr("font-size", 10).attr("fill", C.ink2).text(sub);
}

/** Ring 0..1 for the S_KG pillars */
export function ring(el, value, label, color = C.ink) {
  el.innerHTML = "";
  const S = 128, r = 52;
  const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${S} ${S}`);
  const g = svg.append("g").attr("transform", `translate(${S / 2},${S / 2})`);
  const arc = d3.arc().innerRadius(r - 7).outerRadius(r).cornerRadius(4).startAngle(0);
  g.append("path").attr("d", arc({ endAngle: 2 * Math.PI })).attr("fill", "#ebe8df");
  g.append("path").attr("fill", color).attr("class", "fade-in").attr("d", arc({ endAngle: 2 * Math.PI * Math.max(0, Math.min(1, value || 0)) }));
  g.append("text").attr("text-anchor", "middle").attr("dy", "0.35em").attr("font-family", "Space Grotesk, sans-serif").attr("font-weight", 600)
    .attr("font-size", 22).attr("fill", C.ink).text(pct(value, 1));
  const l = document.createElement("div");
  l.className = "lbl";
  l.innerHTML = label;
  el.appendChild(l);
}
