// EpistemeGraph - website controller.
// On load it restores the last cached state of the active corpus workspace (KG, audits, batch,
// benchmark, brain) and re-attaches to any background job that is still running.
import { initHero } from "./hero.js";
import { createIntro } from "./intro.js";
import { initSignals } from "./signals.js";
import { buildStory } from "./story.js";
import { C, boxes, diverging, forest, gauge, hbar, lines, reliabilityPoints, ring, rocPoints, stacked } from "./charts.js";
import { KG3D, KGExplorer, claimGraph } from "./kg.js";
import { $, $$, STATUS, ago, api, badge, bindSeg, countUp, esc, getSetting, int, isNum, num, pct, segValue, setSetting, tex, toast } from "./util.js";

const S = { state: null, brain: null, bench: null, batch: null, jobs: new Map(), kg: null, uploads: [], batchFile: null, rec: null };
const FEAT = {};

// ============================================================================ boot
// capture mode for screenshots/print: ?static=1 (no animations) and optionally &only=<section id>
const QS = new URLSearchParams(location.search);
if (QS.has("static")) document.documentElement.classList.add("static");
if (QS.get("only")) $$("main section").forEach((s) => { if (s.id !== QS.get("only") && !s.querySelector(`#${QS.get("only")}`)) s.hidden = true; });
// entrance: real loading steps drive the progress bar; the page can never stay stuck behind it
const intro = createIntro($("#intro"));
intro.task("fonts", 1); intro.task("scene", 1.5); intro.task("state", 3); intro.task("graph", 3); intro.task("brain", 1);
(document.fonts ? document.fonts.ready : Promise.resolve()).then(() => intro.set("fonts", 1));
setTimeout(() => intro.set("fonts", 1), 2500);
const hero = initHero($("#hero-canvas"));
intro.set("scene", 1);
intro.then(() => hero.enter());
initSignals($("#signals"));
initScrub();
bindOpening();
bindChrome();
bindCorpus();
bindAudit();
bindBatch();
bindBench();
renderMethod();
loadAll();
setInterval(pollJobs, 2000);

async function loadAll(attempt = 0) {
  try {
    S.state = await api("/api/state", { timeout: 15000 });
  } catch (e) {
    // the server may still be starting (start.bat) - keep trying instead of leaving "loading…" forever
    intro.msg(attempt < 2 ? "Waiting for the server" : `Server not answering yet, retrying (${attempt})`);
    if (attempt === 5) toast("The server is not answering yet. The page fills in automatically as soon as it does.", true);
    $("#ws-name").textContent = "connecting…";
    setTimeout(() => loadAll(attempt + 1), Math.min(5000, 700 + attempt * 500));
    return;
  }
  intro.set("state", 1);
  (S.state.features || []).forEach((f) => (FEAT[f.key] = f));
  (S.state.active_jobs || []).forEach((j) => S.jobs.set(j.id, j));
  renderHeader();
  renderCorpus();
  renderHistory();
  if (S.state.latest_audit && !S.rec) renderAudit(S.state.latest_audit);
  else if (!S.state.latest_audit) $("#audit-result").innerHTML = "";
  loadGraph().catch(() => {}).finally(() => intro.set("graph", 1));
  loadBrain().catch(() => {}).finally(() => intro.set("brain", 1));
  loadBatch();
  loadBench();
  renderTray();
  setTimeout(motion, 300);
}

// ============================================================================ motion (GSAP + tilt)
let motionDone = false;
function motion() {
  const still = document.documentElement.classList.contains("static") || matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (window.VanillaTilt && !still) {
    const els = $$(".math-card, #kg-tiles .card").filter((e) => !e.vanillaTilt);
    els.forEach((e) => e.classList.add("tilt"));
    window.VanillaTilt.init(els, { max: 7, speed: 700, glare: true, "max-glare": 0.14, scale: 1.015, perspective: 900 });
  }
  if (motionDone || still || !window.gsap || !window.ScrollTrigger) return;
  motionDone = true;
  gsap.registerPlugin(ScrollTrigger);
  $$(".section").forEach((sec) => {
    const cards = $$(":scope > .grid > .card, :scope > .grid > div > .card, :scope > .card", sec);
    // .reveal sets opacity:0 in CSS, and gsap.from() animates TO the current computed value, i.e. to 0 (invisible
    // sections). So: drop the CSS gate on these cards and animate explicitly to opacity 1.
    cards.forEach((c) => c.classList.add("in"));
    if (cards.length) gsap.fromTo(cards, { y: 46 }, { y: 0, duration: 0.9, ease: "power3.out", stagger: 0.07,
      scrollTrigger: { trigger: sec, start: "top 85%", once: true }, clearProps: "transform" });
    const t = sec.querySelector(".sec-title");
    if (t) gsap.fromTo(t, { yPercent: 18 }, { yPercent: -10, ease: "none", scrollTrigger: { trigger: sec, start: "top bottom", end: "bottom top", scrub: true } });
  });
  gsap.to(".hero-title", { yPercent: -18, opacity: 0.25, ease: "none", scrollTrigger: { trigger: ".hero", start: "top top", end: "bottom top", scrub: true } });
}

// ============================================================================ chrome
function bindChrome() {
  addEventListener("scroll", () => $("#topbar").classList.toggle("scrolled", scrollY > 10), { passive: true });
  const tick = () => {
    const d = new Date();
    $("#t-clock").textContent = d.toLocaleTimeString([], { hour12: false }) + " " +
      (Intl.DateTimeFormat([], { timeZoneName: "short" }).formatToParts(d).find((p) => p.type === "timeZoneName")?.value || "");
  };
  tick(); setInterval(tick, 1000);
  const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) e.target.classList.add("in"); }), { threshold: 0.08 });
  $$(".reveal").forEach((el) => io.observe(el));
  // safety net: content must never stay invisible if an observer / animation does not fire
  setTimeout(() => $$(".reveal").forEach((el) => el.classList.add("in")), 2000);
  const secIO = new IntersectionObserver((es) => es.forEach((e) => {
    if (!e.isIntersecting) return;
    const id = e.target.dataset.sec;
    $$("#rail a").forEach((a) => a.classList.toggle("active", a.dataset.sec === id));
    document.body.classList.toggle("on-dark", id === "graph");
    $$("#nav a").forEach((a) => a.classList.toggle("active", a.getAttribute("href") === `#${id}`));
  }), { rootMargin: "-45% 0px -50% 0px" });
  $$("[data-sec]").forEach((s) => s.tagName !== "A" && secIO.observe(s));
  $("#ws-btn").addEventListener("click", (e) => { e.stopPropagation(); $("#ws-menu").hidden = !$("#ws-menu").hidden; });
  document.addEventListener("click", (e) => { if (!e.target.closest(".ws-switch")) $("#ws-menu").hidden = true; });
}

function renderHeader() {
  const st = S.state, ws = st.workspace || {}, kg = st.kg || {}, br = st.brain || {};
  const kgState = ws.status === "building" || ws.status === "queued" ? "BUILDING" : kg.overall_kg_accuracy_score ? "ACTIVE" : "EMPTY";
  $("#t-state").textContent = kgState;
  $("#t-skg").textContent = pct(kg.overall_kg_accuracy_score, 2);
  $("#t-brain").textContent = (br.status || "uncalibrated").toUpperCase();
  $("#ws-name").textContent = ws.name || ws.id || "corpus";
  $(".ws-dot").style.background = kgState === "ACTIVE" ? "#0ca30c" : kgState === "BUILDING" ? "#fab219" : "#898781";
  $("#h-corpus").textContent = `${(ws.documents || []).length || kg.documents_in_corpus || 0} PDF · ${ws.name || ""}`.slice(0, 48);
  $("#h-ref").textContent = kg.verified_triplets_pushed ? `${kg.verified_triplets_pushed} VERIFIED TRIPLES` : "NO GRAPH YET";
  $("#h-brain").textContent = br.status === "calibrated" ? `${br.champion} · α=${num(br.alpha, 2)}` : br.status === "transferred" ? "TRANSFERRED" : "UNCALIBRATED";
  countUp($("#hs-skg"), kg.overall_kg_accuracy_score, (v) => `${(100 * v).toFixed(1)}%`);
  countUp($("#hs-tri"), kg.verified_triplets_pushed);
  countUp($("#hs-ent"), kg.resolved_entity_count);
  countUp($("#hs-aud"), st.history_count || 0);
  $("#foot-sys").textContent = `LLM ${st.system.llm_model} · JUDGE ${st.system.judge_model} · NEO4J ${st.system.neo4j ? "CONNECTED" : "LOCAL"} · ${st.system.budget_usd ? `SPENT ~$${st.system.spend_usd} / $${st.system.budget_usd}` : `SPENT ~$${st.system.spend_usd}`} · ${st.system.api_key ? "API KEY OK" : "NO API KEY"}`;
  $("#ws-menu").innerHTML = `<div class="kicker" style="padding:6px 12px">Corpora (each has its own graph)</div>` + (st.workspaces || []).map((w) => `
    <div class="ws-item ${w.id === ws.id ? "active" : ""}" data-id="${esc(w.id)}">
      <b>${esc(w.name || w.id)}</b><small>${pct(w.s_kg, 1)}</small>
      <small>${(w.documents || []).length} docs · ${int(w.triples)} triples · ${w.has_brain ? "calibrated" : "no brain"}</small>
      <small>${esc(w.status || "")} · ${ago(w.created_at)}</small></div>`).join("");
  $$("#ws-menu .ws-item").forEach((it) => it.addEventListener("click", async () => {
    $("#ws-menu").hidden = true;
    if (it.dataset.id === ws.id) return;
    await api(`/api/workspaces/${encodeURIComponent(it.dataset.id)}/activate`, { method: "POST" });
    S.rec = null; S.jobs.clear();
    toast("Switched corpus");
    loadAll();
  }));
}

// ============================================================================ jobs
const JOB_LABEL = { build_kg: "Building knowledge graph", batch: "Batch audit", bench_gen: "Generating benchmark",
  benchmark: "Research benchmark", retrain: "Retraining brain", kg_precision: "KG precision audit" };

function trackJob(job) { if (job) { S.jobs.set(job.id, job); renderTray(); } }

async function pollJobs() {
  const live = [...S.jobs.values()].filter((j) => ["queued", "running"].includes(j.state));
  for (const j of live) {
    let cur;
    try { cur = await api(`/api/jobs/${encodeURIComponent(j.id)}`); } catch (_) { continue; }
    S.jobs.set(j.id, cur);
    if (cur.type === "batch") loadBatch(true);
    if (cur.type === "benchmark" || cur.type === "bench_gen") renderBenchStatus(cur);
    if (cur.type === "build_kg") renderBuildStatus(cur);
    if (!["queued", "running"].includes(cur.state)) onJobDone(cur);
  }
  renderTray();
}

function onJobDone(j) {
  const ok = j.state === "completed";
  toast(`${JOB_LABEL[j.type] || j.type}: ${ok ? (j.warning ? "finished with a note" : "done") : j.state}${j.error ? " - " + j.error.split(String.fromCharCode(10))[0] : ""}`, !ok || !!j.warning);
  setTimeout(() => { S.jobs.delete(j.id); renderTray(); }, 8000);
  if (j.type === "build_kg" || j.type === "kg_precision") loadAll();
  if (j.type === "batch") loadBatch();
  if (j.type === "bench_gen") loadBench();
  if (j.type === "benchmark" || j.type === "retrain") { loadBench(); loadBrain(); api("/api/state").then((s) => { S.state = s; renderHeader(); }); }
}

function renderTray() {
  $("#tray").innerHTML = [...S.jobs.values()].map((j) => {
    const p = j.total ? j.done / j.total : 0, live = ["queued", "running"].includes(j.state);
    return `<div class="tray-item"><div class="top"><span>${esc(JOB_LABEL[j.type] || j.type)}</span><span>${esc(j.state)}</span></div>
      <div class="progress ${live && !j.total ? "indet" : ""}"><i style="width:${Math.round(100 * (live ? p : 1))}%"></i></div>
      <div class="muted mono" style="font-size:10.5px;margin-top:5px">${esc(j.stage || "")} ${j.total ? `${j.done}/${j.total}` : ""} ${esc((j.note || "").slice(0, 60))}</div></div>`;
  }).join("");
}

function stepper(stages, current, state) {
  const idx = stages.indexOf(current);
  return `<div class="stepper">${stages.map((s, i) => `<span class="${state === "completed" || i < idx ? "done" : i === idx && state === "running" ? "on" : ""}">${i + 1} · ${esc(s)}</span>`).join("")}</div>`;
}

function jobPanel(j, stages) {
  if (!j) return `<div class="empty">Not started.</div>`;
  const live = ["queued", "running"].includes(j.state), p = j.total ? j.done / j.total : 0;
  return `${stepper(stages, j.stage, j.state)}
    <div class="progress ${live && !j.total ? "indet" : ""}"><i style="width:${Math.round(100 * (j.state === "completed" ? 1 : p))}%"></i></div>
    <div style="display:flex;justify-content:space-between;gap:10px;margin-top:8px;font-size:13px;flex-wrap:wrap">
      <span>${badge(j.state === "completed" ? "VERIFIED" : j.state === "failed" ? "HALLUCINATED" : "ABSTAIN").replace(/Verified|Hallucinated|Abstain/, esc(j.state))}
        <span class="mono muted" style="font-size:11.5px">${esc(j.stage || "")} ${j.total ? `${j.done}/${j.total}` : ""} ${esc(j.note || "")}</span></span>
      ${live ? `<button class="btn small ghost" data-cancel="${esc(j.id)}">Stop (resumable)</button>` : `<span class="muted mono" style="font-size:11px">${ago(j.updated_at)}</span>`}
    </div>
    ${j.error ? `<div class="log" style="white-space:pre-wrap">${esc(j.error)}</div>` : ""}
    ${j.warning ? `<div class="log" style="white-space:pre-wrap;border-left:3px solid #c98a00;background:#fff8e6;color:#6b4a00">${esc(j.warning)}</div>` : ""}
    ${j.log_tail && (live || j.state === "failed") ? `<details ${j.state === "failed" ? "open" : ""}><summary class="mono muted" style="font-size:11px;cursor:pointer;margin-top:8px">log</summary><div class="log">${esc(j.log_tail)}</div></details>` : ""}`;
}

document.addEventListener("click", async (e) => {
  const b = e.target.closest("[data-cancel]");
  if (!b) return;
  await api(`/api/jobs/${encodeURIComponent(b.dataset.cancel)}/cancel`, { method: "POST" });
  toast("Stopped. Re-run later to resume from the last checkpoint.");
  pollJobs();
});

// ============================================================================ 001 corpus
function bindCorpus() {
  const dz = $("#dropzone"), fi = $("#file-input");
  dz.addEventListener("click", () => fi.click());
  dz.addEventListener("keydown", (e) => (e.key === "Enter" || e.key === " ") && fi.click());
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => addUploads(e.dataTransfer.files));
  fi.addEventListener("change", () => addUploads(fi.files));
  bindSeg("upload-mode", (v) => {
    $("#upload-note").textContent = v === "new"
      ? "A new corpus starts a fresh graph from scratch in its own Neo4j namespace; earlier graphs are preserved and you can switch back any time."
      : "These PDFs are merged into the active corpus: new mentions are resolved against its existing entities; already-ingested files are skipped.";
    $("#corpus-name").hidden = v !== "new";
  });
  $("#upload-btn").addEventListener("click", upload);
  $("#kgprec-btn").addEventListener("click", async () => {
    try { const r = await api("/api/kg/precision-audit", { json: { n: Number($("#kgprec-n").value) || 100 } }); trackJob(r.job); toast("KG precision audit started"); }
    catch (e) { toast(e.message, true); }
  });
  $("#kg-search").addEventListener("keydown", (e) => {
    if (e.key !== "Enter" || !S.kg) return;
    const n = S.kg.find(e.target.value);
    n ? S.kg.select(n) : toast("No entity matches that name");
  });
  $("#kg-fit").addEventListener("click", () => S.kg && S.kg.fit());
  $("#kg-3d").addEventListener("click", () => S.kg && S.kg.toggleLabels());
  $("#kg-sync").addEventListener("click", async () => {
    try { const r = await api("/api/kg/sync-neo4j", { method: "POST" }); trackJob(r.job); toast("Pushing the whole graph to Neo4j…"); }
    catch (e) { toast(e.message, true); }
  });
  bindSettings();
}

function addUploads(files) {
  for (const f of files) if (/\.(pdf|txt)$/i.test(f.name) && !S.uploads.some((u) => u.name === f.name)) S.uploads.push(f);
  $("#file-list").innerHTML = S.uploads.map((f, i) => `<div class="file-row"><span>${esc(f.name)}</span><span class="muted">${(f.size / 1048576).toFixed(2)} MB <a href="#" data-rm="${i}">remove</a></span></div>`).join("");
  $$("#file-list [data-rm]").forEach((a) => a.addEventListener("click", (e) => { e.preventDefault(); S.uploads.splice(+a.dataset.rm, 1); addUploads([]); }));
  $("#upload-btn").disabled = !S.uploads.length;
}

async function upload() {
  const fd = new FormData();
  S.uploads.forEach((f) => fd.append("files", f));
  fd.append("mode", segValue("upload-mode"));
  fd.append("name", $("#corpus-name").value);
  $("#upload-btn").disabled = true;
  try {
    const r = await api("/api/kg/upload", { method: "POST", body: fd });
    trackJob(r.job);
    S.uploads = []; addUploads([]);
    toast(segValue("upload-mode") === "new" ? "Fresh corpus created: building its graph…" : "Extending the current corpus…");
    S.state = await api("/api/state"); renderHeader(); renderCorpus();
  } catch (e) { toast(e.message, true); $("#upload-btn").disabled = false; }
}

function renderBuildStatus(j) {
  const live = ["queued", "running"].includes(j.state);
  $("#upload-note").innerHTML = live
    ? `<b>Building:</b> ${esc(j.stage)} ${j.total ? `${j.done}/${j.total} chunks` : ""} <span class="muted">${esc(j.note || "")}</span>`
    : j.state === "failed" ? `<span style="color:#a42424">Build failed: ${esc(j.error || "see log")}</span>` : "Build complete.";
}

function renderCorpus() {
  const kg = S.state.kg, diag = S.state.kg_diagnostics || {};
  if (!kg) {
    $("#skg-v").textContent = "—"; $("#skg-sub").textContent = "no graph yet: drop PDFs to build one";
    $("#rings").innerHTML = ""; $("#skg-formula").innerHTML = ""; $("#kg-tiles").innerHTML = "";
    $("#doc-table").innerHTML = `<div class="empty">No documents.</div>`; $("#shared").innerHTML = "";
  } else {
    countUp($("#skg-v"), kg.overall_kg_accuracy_score, (v) => `${(100 * v).toFixed(2)}%`);
    $("#skg-sub").innerHTML = `S<sub>KG</sub> · ${int(kg.documents_in_corpus)} docs · ${int(kg.total_chunks_processed)} chunks`;
    const P = [kg.provenance_grounding_score, kg.semantic_coverage_score, kg.topological_coherence_score, kg.internal_consistency_score];
    tex($("#skg-formula"), `S_{\\mathrm{KG}}=\\big(\\Phi_{\\mathrm{prov}}\\,\\Phi_{\\mathrm{cov}}\\,\\Phi_{\\mathrm{topo}}\\,\\Phi_{\\mathrm{cons}}\\big)^{1/4}=(${P.map((p) => num(p, 3)).join("\\cdot")})^{1/4}=${num(kg.overall_kg_accuracy_score, 4)}`);
    $("#rings").innerHTML = "<div class='ring'></div>".repeat(4);
    const lab = ["Φ<sub>prov</sub> precision", "Φ<sub>cov</sub> coverage", "Φ<sub>topo</sub> topology", "Φ<sub>cons</sub> consistency"];
    $$("#rings .ring").forEach((el, i) => ring(el, P[i], lab[i], i === 0 ? C.blue : C.ink));
    const tiles = [
      ["Verified triples", int(kg.verified_triplets_pushed), `of ${int(kg.raw_triplets_extracted)} extracted · ${pct(diag.claim_acceptance_rate)} accepted`],
      ["Canonical entities", int(kg.resolved_entity_count), `${int(kg.cross_pdf_shared_entities_count)} shared across papers`],
      ["Rejected claims", int((kg.raw_triplets_extracted || 0) - (kg.verified_triplets_pushed || 0)), Object.entries(diag.rejected_by_reason || {}).map(([k, v]) => `${k.replace(/_/g, " ")}: ${v}`).join(" · ") || "evidence quote not found"],
      ["Neo4j namespace", esc(S.state.workspace.kg_id || "—"), `${esc(S.state.workspace.status || "")} · ${S.state.system.neo4j ? "AuraDB + local mirror" : "local mirror"}`],
    ];
    $("#kg-tiles").innerHTML = tiles.map(([k, v, s]) => `<div class="card tile"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${esc(s)}</div></div>`).join("");
    $("#doc-table").innerHTML = (S.state.documents || []).length ? `<div class="table-wrap" style="max-height:260px"><table><thead><tr><th>Document</th><th>Chunks</th><th>Ingested</th><th>Hash</th></tr></thead><tbody>${
      S.state.documents.map((d) => `<tr><td>${esc(d.name)}</td><td>${int(d.chunks)}</td><td>${ago(d.ingested_at)}</td><td class="mono muted">${esc(d.hash)}</td></tr>`).join("")}</tbody></table></div>` : `<div class="empty">No documents.</div>`;
    $("#shared").innerHTML = (kg.cross_pdf_shared_entities || []).map((s) => `<span class="chip" style="cursor:pointer" data-ent="${esc(s)}">${esc(s)}</span>`).join("") || `<span class="muted">None yet.</span>`;
    $$("#shared [data-ent]").forEach((c) => c.addEventListener("click", () => { const n = S.kg && S.kg.find(c.dataset.ent); if (n) { $("#kg-wrap").scrollIntoView({ behavior: "smooth", block: "center" }); S.kg.select(n); } }));
  }
  const kp = S.state.kg_precision;
  const o = kp && kp.overall ? kp.overall.strict : null;
  $("#kgprec").innerHTML = o ? `<div class="skg-hero"><div class="v" style="font-size:52px">${pct(o.p, 1)}</div>
      <div class="mono muted" style="font-size:12px">95% CI [${pct(o.lo, 1)}, ${pct(o.hi, 1)}]<br>n = ${o.n} · auditor: ${kp.human ? "human" : esc(kp.auditor)}</div></div>
      <div class="muted" style="font-size:12.5px">Strict precision of a uniform random sample of triples, each checked against its source passage (Wilson interval). Lenient: ${pct(kp.overall.lenient.p, 1)}.</div>`
    : `<div class="muted" style="font-size:13.5px">S<sub>KG</sub> is intrinsic. To report the graph's <b>accuracy</b> as a measured number, audit a random sample of triples against their source passages; the result comes with a 95% confidence interval.</div>`;
  const ma = kp && kp.model_assisted;
  if (ma && ma.available) $("#kgprec").insertAdjacentHTML("beforeend", `<div class="divider"></div><div class="kicker">Learned accuracy · model-assisted estimator</div>
    <div class="skg-hero"><div class="v" style="font-size:40px">${pct(ma.estimate, 1)}</div><div class="mono muted" style="font-size:12px">95% CI [${pct(ma.lo, 1)}, ${pct(ma.hi, 1)}]<br>n=${ma.n} audited of N=${ma.N}</div></div>
    <div class="chart" id="ch-tw"></div>`);
  if (ma && ma.available) diverging($("#ch-tw"), Object.entries(ma.learned_weights).map(([k, v]) => ({ label: k.replace(/_/g, " "), value: v })).sort((a, b) => a.value - b.value), { leftLabel: "\u2190 less likely correct", rightLabel: "more likely correct \u2192" });
  const sens = S.state.sensitivity;
  $("#sens-card").hidden = !sens;
  if (sens) {
    forest($("#ch-sens"), sens.rows.map((r) => ({ label: r.parameter === "default" ? "all defaults" : `${r.meaning} = ${r.value}`, value: r.S_KG,
      color: r.parameter === "default" ? C.blue : C.ink, tip: `<b>${esc(r.meaning)}</b> ${r.value ?? ""}<br>S_KG ${num(r.S_KG, 4)} · triples ${r.verified} · entities ${r.entities}` })),
      { ref: sens.S_KG_default, refLabel: "default", fmt: (v) => num(v, 3) });
    $("#sens-note").textContent = `Max |change| of S_KG over all perturbations: ${num(sens.max_abs_deviation, 4)} (range ${num(sens.S_KG_range, 4)}). Re-run: python scripts/sensitivity_phase1.py`;
  }
}

async function loadGraph() {
  if (!S.kg) {
    try { S.kg = window.ForceGraph3D ? new KG3D($("#kg-canvas"), $("#kg-drawer")) : new KGExplorer($("#kg-canvas"), $("#kg-drawer")); }
    catch (_) { S.kg = new KGExplorer($("#kg-canvas"), $("#kg-drawer")); }   // no WebGL: 2D canvas fallback
  }
  const data = await api("/api/kg/graph", { timeout: 60000 }).catch(() => ({ nodes: [], links: [] }));
  S.kg.setData(data);
  try { buildStory($("#graph"), S.kg, data, S.state); } catch (e) { console.warn("story", e); }
  $("#kg-names").innerHTML = data.nodes.slice(0, 2000).map((n) => `<option value="${esc(n.id)}">`).join("");
  const pdfs = [...new Set(data.nodes.flatMap((n) => n.pdfs))].sort();
  $("#kg-pdfs").innerHTML = pdfs.map((p) => `<button class="chip" data-pdf="${esc(p)}" style="cursor:pointer">${esc(p)}</button>`).join("");
  $$("#kg-pdfs [data-pdf]").forEach((b) => b.addEventListener("click", () => {
    const on = S.kg.filterPdf(b.dataset.pdf);
    $$("#kg-pdfs [data-pdf]").forEach((x) => (x.style.background = x.dataset.pdf === on ? "#0e0e0c" : "", x.style.color = x.dataset.pdf === on ? "#f3f1ea" : ""));
  }));
}

// ============================================================================ 002 audit
function bindAudit() {
  bindSeg("audit-mode");
  $("#audit-btn").addEventListener("click", runAudit);
  $("#q").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) runAudit(); });
  $("#hist-filter").addEventListener("input", renderHistory);
}

function renderExamples(items) {
  const pick = [];
  for (const cat of ["FACTUAL", "CROSS_DOC", "BORDERLINE", "FALSE_PREMISE"]) {
    const it = (items || []).find((x) => x.category === cat);
    if (it) pick.push([cat.replace("_", " ").toLowerCase(), it.question]);
  }
  if (!pick.length) pick.push(
    ["factual", "How does node2vec's biased random walk interpolate between breadth-first and depth-first search?"],
    ["borderline", "What GPU memory footprint does LINE report when embedding the full Wikipedia network?"],
    ["false premise", "Why does DeepWalk's genetic crossover operator make its embeddings robust to noisy edges?"]);
  $("#examples").innerHTML = pick.map(([c, q]) => `<button class="example" data-q="${esc(q)}" title="${esc(q)}"><b>${esc(c)}</b>${esc(q)}</button>`).join("");
  $$("#examples .example").forEach((b) => b.addEventListener("click", () => { $("#q").value = b.dataset.q; $("#q").focus(); }));
}

let pipeTimer;
function animatePipeline(on) {
  const steps = $$("#pipeline div");
  $("#pipeline").hidden = false;
  clearInterval(pipeTimer);
  steps.forEach((s) => s.classList.remove("on", "done"));
  if (!on) { steps.forEach((s) => s.classList.add("done")); return; }
  let i = 0;
  steps[0].classList.add("on");
  const durations = [9000, 7000, 1500, 2500, 800];
  const next = () => {
    if (i >= steps.length - 1) return;
    steps[i].classList.replace("on", "done"); i++; steps[i].classList.add("on");
    pipeTimer = setTimeout(next, durations[i]);
  };
  pipeTimer = setTimeout(next, durations[0]);
}

async function runAudit() {
  const q = $("#q").value.trim();
  if (!q) { toast("Type a question first"); return; }
  $("#audit-btn").disabled = true;
  animatePipeline(true);
  const t0 = performance.now();
  try {
    const rec = await api("/api/audit", { json: { query: q, mode: segValue("audit-mode") } });
    clearTimeout(pipeTimer); animatePipeline(false);
    renderAudit(rec);
    toast(`Audited in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
    S.state = await api("/api/state"); renderHeader(); renderHistory();
  } catch (e) {
    clearTimeout(pipeTimer); $("#pipeline").hidden = true;
    toast(e.message, true);
  } finally { $("#audit-btn").disabled = false; }
}

function renderHistory() {
  const f = ($("#hist-filter").value || "").toLowerCase();
  const items = (S.state?.history || []).filter((h) => !f || (h.query || "").toLowerCase().includes(f));
  $("#hist-count").textContent = `· ${S.state?.history_count || 0}`;
  $("#history").innerHTML = items.length ? items.map((h) => `<div class="h-item ${S.rec && S.rec.id === h.id ? "active" : ""}" data-id="${esc(h.id)}">
      <div class="q">${esc(h.query)}</div>
      <div class="m"><span>${badge(h.decision)}</span><span>${isNum(h.hallucination_probability) ? `P(h) ${pct(h.hallucination_probability, 0)}` : `extent ${pct(h.hallucination_extent, 0)}`} · ${ago(h.timestamp)}</span></div></div>`).join("")
    : `<div class="empty">Every audit is saved here and restored when you come back.</div>`;
  $$("#history .h-item").forEach((it) => it.addEventListener("click", async () => {
    try { renderAudit(await api(`/api/history/${encodeURIComponent(it.dataset.id)}`)); $("#audit-result").scrollIntoView({ behavior: "smooth", block: "start" }); }
    catch (e) { toast(e.message, true); }
  }));
}

function renderAudit(rec) {
  S.rec = rec;
  $$("#history .h-item").forEach((h) => h.classList.toggle("active", h.dataset.id === rec.id));
  const st = STATUS[rec.decision] || STATUS.UNCALIBRATED;
  const ph = rec.hallucination_probability, ext = rec.hallucination_extent ?? 0;
  const nAs = rec.asserted_claims ?? (rec.edge_audit_details || []).filter((d) => d.claim_type !== "REFUTED").length;
  const nUn = rec.unsupported_claims ?? 0;
  const brain = rec.brain || {};
  const pset = (rec.prediction_set || []).map((c) => (c === 1 ? "supported" : "hallucinated")).join(", ");
  const gap = isNum(rec.S_final) ? rec.S_final - rec.llm_confidence : null;

  $("#audit-result").innerHTML = `
  <div class="verdict" style="--vc:${st.color}">
    <div class="badge-xl" style="${rec.decision === "ABSTAIN" ? "color:#0e0e0c" : ""}">${st.icon}</div>
    <div><div class="t">${esc(st.label)}</div><div class="d">${esc(st.text)}</div>
      <div class="d mono" style="font-size:11.5px">conformal set {${esc(pset || "∅")}} · ${esc((rec.mode || "closed_book").replace("_", "-"))} · ${ago(rec.timestamp)}
      ${brain.transferred ? ` · <span style="color:#a06a00">brain transferred from another corpus (guarantee not certified here)</span>` : ""}</div></div>
    <div class="r">${isNum(ph) ? "P(hallucination)" : "unsupported claims"}<b>${isNum(ph) ? pct(ph, 0) : pct(ext, 0)}</b></div>
  </div>
  <div class="grid g2" style="margin-top:16px">
    <div class="card path-llm"><h4><span class="dot" style="background:${C.orange}"></span>Path 1 · what the LLM said</h4>
      <div class="kicker" style="color:var(--ink-2);text-transform:none;font-family:var(--sans);font-size:13px;letter-spacing:0">${esc(rec.query)}</div>
      <div class="answer">${esc(rec.answer)}</div>
      <div class="divider"></div>
      <div class="grid g2"><div class="tile"><div class="k">LLM self-confidence</div><div class="v">${pct(rec.llm_confidence, 0)}</div><div class="s">${esc(rec.llm_confidence_reasoning || "")}</div></div>
      <div class="tile"><div class="k">Calibration gap</div><div class="v">${gap === null ? "—" : `${gap >= 0 ? "+" : ""}${(100 * gap).toFixed(0)} pts`}</div><div class="s">verified support − LLM confidence</div></div></div>
    </div>
    <div class="card path-kg"><h4><span class="dot" style="background:${C.blue}"></span>Path 2 · what the knowledge graph supports</h4>
      <div class="grid g2" style="align-items:center">
        <div id="gauge"></div>
        <div>
          <div class="tile"><div class="k">How much is unsupported</div><div class="v">${nUn} / ${nAs}</div><div class="s">asserted claims without a grounded KG path (or in conflict)</div></div>
          <div class="meter" style="margin-top:10px"><i style="width:${Math.round(100 * ext)}%;background:${ext > 0.5 ? C.crit : ext > 0 ? C.orange : C.good}"></i></div>
        </div>
      </div>
      <div class="grid g3" style="margin-top:14px">
        <div class="tile"><div class="k">S<sub>query</sub></div><div class="v" style="font-size:24px">${pct(rec.S_query, 0)}</div><div class="s">calibrated P(supported | x)</div></div>
        <div class="tile"><div class="k">S<sub>KG</sub>(q)</div><div class="v" style="font-size:24px">${pct(rec.S_KG_local, 0)}</div><div class="s">local graph fidelity</div></div>
        <div class="tile"><div class="k">GT subgraph</div><div class="v" style="font-size:24px">${int(rec.gt_nodes)}</div><div class="s">nodes · ${int(rec.gt_edges)} edges</div></div>
      </div>
    </div>
  </div>
  <div class="grid g-wide" style="margin-top:16px">
    <div class="card"><h4>Why · the 13-D evidence vector</h4><div id="why" class="chart"></div></div>
    <div class="card"><h4>Claim graph G′</h4><div class="legend"><span><i style="background:${C.blue}"></i>grounded in KG</span><span><i style="background:${C.orange}"></i>only the LLM says so</span><span><i style="background:${C.crit}"></i>conflict</span><span><i style="background:${C.muted}"></i>refuted by the answer</span></div><div class="mini-graph" id="mini-graph"></div></div>
  </div>
  <div class="card" style="margin-top:16px"><h4>Claim ledger · ${(rec.edge_audit_details || []).length} claims</h4><div class="claims" id="claims"></div></div>`;

  gauge($("#gauge"), isNum(ph) ? ph : null, { color: !isNum(ph) ? C.gray : ph > 0.5 ? C.crit : ph > 0.2 ? C.orange : C.good,
    label: "P(hallucination)", sub: isNum(ph) ? "calibrated" : "train the brain to calibrate" });
  renderWhy($("#why"), rec.features || {});
  claimGraph($("#mini-graph"), rec.edge_audit_details);
  $("#claims").innerHTML = (rec.edge_audit_details || []).map((d) => {
    const kind = d.conflict ? ["✕", C.crit, d.conflict.replace(/_/g, " ")] : d.claim_type === "REFUTED" ? ["–", C.muted, "refuted by answer"]
      : d.is_supported ? ["✓", C.blue, d.support_tier.replace(/_/g, " ").toLowerCase()] : ["?", C.orange, "no grounded path"];
    return `<div class="claim"><div class="ic" style="background:${kind[1]}">${kind[0]}</div>
      <div><div class="tr"><b>${esc(d.source)}</b><span class="rel-t">${esc(d.relation)}</span><b>${esc(d.target)}</b></div>
        ${d.evidence_quote ? `<div class="ev">KG evidence: “${esc(d.evidence_quote)}”</div>` : d.nearest_corpus_sentence ? `<div class="ev">nearest corpus sentence: “${esc(d.nearest_corpus_sentence)}”</div>` : ""}</div>
      <div class="meta">${esc(kind[2])}<br>${esc(d.claim_type.toLowerCase())} · s=${num(d.provenance_strength, 2)}${isNum(d.nli_consistency) ? `<br>NLI consistency ${num(d.nli_consistency, 2)}` : ""}${isNum(d.learned_hallucination_score) ? `<br><b style="color:${d.learned_hallucination_score > 0.5 ? C.crit : C.ink2}">learned P(unsupported) ${pct(d.learned_hallucination_score, 0)}</b>` : ""}</div></div>`;
  }).join("") || `<div class="empty">The extractor found no factual claims in this answer.</div>`;
}

function renderWhy(el, f) {
  const w = S.brain?.metrics?.learned_feature_weights;
  const keys = Object.keys(FEAT).length ? Object.keys(FEAT) : Object.keys(f);
  if (w && w.standardised_coefficients) {
    const data = keys.map((k) => {
      const z = (f[k] - w.feature_means[k]) / (w.feature_scales[k] || 1), c = w.standardised_coefficients[k] * z;
      return { label: FEAT[k]?.short || k, value: c, tip: `<b>${esc(FEAT[k]?.name || k)}</b><br>value ${num(f[k], 3)} (train mean ${num(w.feature_means[k], 3)})<br>learned weight ${num(w.standardised_coefficients[k], 2)} → contribution ${num(c, 2)}` };
    }).sort((a, b) => a.value - b.value);
    diverging(el, data, { leftLabel: "← toward hallucinated", rightLabel: "toward supported →" });
  } else {
    el.innerHTML = `<div class="grid g4" style="gap:10px">${keys.map((k) => `<div class="tile" style="border:1px solid var(--ring);border-radius:10px;padding:10px">
      <div class="k">${esc(FEAT[k]?.name || k)}</div><div class="v" style="font-size:22px">${num(f[k], 3)}</div>
      <div class="s">${esc(FEAT[k]?.family || "")} · ${FEAT[k]?.sign > 0 ? "higher = better" : FEAT[k]?.sign < 0 ? "lower = better" : "learned"}</div></div>`).join("")}</div>
      <p class="muted" style="font-size:12.5px;margin:10px 0 0">Once the brain is trained, this panel shows each feature's learned contribution to the verdict.</p>`;
  }
}

// ============================================================================ 003 batch
function bindBatch() {
  const dz = $("#batch-drop"), fi = $("#batch-file");
  dz.addEventListener("click", () => fi.click());
  dz.addEventListener("keydown", (e) => (e.key === "Enter" || e.key === " ") && fi.click());
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  const pick = async (f) => {
    if (!f) return;
    S.batchFile = f;
    const n = (await f.text()).split(/\r?\n/).filter((l) => l.trim() && !l.trim().startsWith("#")).length;
    $("#batch-file-name").textContent = `${f.name} · ${n} questions`;
  };
  dz.addEventListener("drop", (e) => pick(e.dataTransfer.files[0]));
  fi.addEventListener("change", () => pick(fi.files[0]));
  bindSeg("batch-mode");
  $("#batch-btn").addEventListener("click", async () => {
    const fd = new FormData();
    if (S.batchFile) fd.append("file", S.batchFile); else fd.append("queries", $("#batch-text").value);
    fd.append("mode", segValue("batch-mode"));
    try {
      const r = await api("/api/batch", { method: "POST", body: fd });
      trackJob(r.job); S.batchFile = null; $("#batch-file-name").textContent = ""; $("#batch-text").value = "";
      toast("Batch started: results stream in below"); loadBatch(true);
    } catch (e) { toast(e.message, true); }
  });
}

async function loadBatch(quiet = false) {
  const b = await api("/api/batch/latest").catch(() => null);
  S.batch = b;
  if (!b || !b.batch_id) { $("#batch-status").innerHTML = `<h4>Status</h4><div class="empty">No batch yet. Drop a .txt file with one question per line.</div>`; $("#batch-results").innerHTML = ""; return; }
  if (b.job && ["queued", "running"].includes(b.job.state) && !S.jobs.has(b.job.id)) trackJob(b.job);
  const bjob = b.job || ((b.rows || []).length ? { state: "completed", stage: "completed", done: b.rows.length, total: b.rows.length, updated_at: b.summary?.finished_at } : null);
  $("#batch-status").innerHTML = `<h4>Status · ${esc(b.batch_id)} · ${int(b.config?.n)} questions</h4>${jobPanel(bjob, ["intercept", "evaluate", "completed"])}`;
  const rows = b.rows || [];
  if (!rows.length) { $("#batch-results").innerHTML = ""; return; }
  const dec = {}; rows.forEach((r) => (dec[r.decision] = (dec[r.decision] || 0) + 1));
  const mean = (k) => { const v = rows.map((r) => r[k]).filter(isNum); return v.length ? v.reduce((a, c) => a + Number(c), 0) / v.length : null; };
  $("#batch-results").innerHTML = `
    <div class="grid g5">
      <div class="card tile"><div class="k">Audited</div><div class="v">${rows.length}</div><div class="s">of ${int(b.config?.n)} questions</div></div>
      ${["VERIFIED", "ABSTAIN", "HALLUCINATED", "UNCALIBRATED"].filter((d) => dec[d] || d !== "UNCALIBRATED").slice(0, 3).map((d) =>
        `<div class="card tile"><div class="k">${badge(d)}</div><div class="v">${dec[d] || 0}</div><div class="s">${pct((dec[d] || 0) / rows.length, 0)} of answers</div></div>`).join("")}
      <div class="card tile"><div class="k">Mean unsupported share</div><div class="v">${pct(mean("hallucination_extent"), 0)}</div><div class="s">LLM confidence ${pct(mean("c_llm"), 0)}${isNum(mean("hallucination_probability")) ? ` · P(h) ${pct(mean("hallucination_probability"), 0)}` : ""}</div></div>
    </div>
    <div class="card" style="margin-top:16px"><h4 style="justify-content:space-between">Results <a class="btn small ghost" href="/api/batch/${encodeURIComponent(b.batch_id)}/csv">Download CSV ↓</a></h4>
      <div class="table-wrap"><table><thead><tr><th>#</th><th>Question</th><th>Verdict</th><th>P(hallucination)</th><th>Unsupported</th><th>LLM conf.</th><th>Claims</th></tr></thead><tbody>
      ${rows.map((r, i) => `<tr class="clickable" data-i="${i}"><td class="mono muted">${esc(r.id)}</td><td class="q">${esc(r.question)}</td><td>${badge(r.decision)}</td>
        <td>${bar(r.hallucination_probability, C.crit)}</td><td>${bar(r.hallucination_extent, C.orange)}</td><td>${bar(r.c_llm, C.orange)}</td>
        <td class="mono">${int(r.unsupported_claims)}/${int(r.asserted_claims)}</td></tr>`).join("")}</tbody></table></div></div>`;
  $$("#batch-results tr.clickable").forEach((tr) => tr.addEventListener("click", () => {
    const r = rows[+tr.dataset.i];
    const nxt = tr.nextElementSibling;
    if (nxt && nxt.classList.contains("detail")) { nxt.remove(); return; }
    tr.insertAdjacentHTML("afterend", `<tr class="detail"><td></td><td colspan="6"><div class="answer" style="max-height:260px">${esc(r.answer || "")}</div>
      <div class="chips" style="margin-top:8px">${Object.keys(FEAT).map((k) => `<span class="chip">${esc(FEAT[k].tex ? k : k)} ${num(r[k], 3)}</span>`).join("")}</div></td></tr>`);
  }));
  if (quiet) return;
}

const bar = (v, color) => isNum(v) ? `<div class="bar-cell"><div class="meter"><i style="width:${Math.round(100 * Math.min(1, Math.max(0, v)))}%;background:${color}"></i></div><span class="mono" style="font-size:11.5px">${pct(v, 0)}</span></div>` : `<span class="muted">—</span>`;

// ============================================================================ 004 benchmark
function bindBench() {
  $("#gen-btn").addEventListener("click", async () => {
    try { const r = await api("/api/benchmark/generate", { json: { n: Number($("#gen-n").value) || 500, workers: 6, batch_api: $("#gen-batch").checked } }); trackJob(r.job); renderBenchStatus(r.job); toast("Generating benchmark questions…"); }
    catch (e) { toast(e.message, true); }
  });
  $("#bench-file").addEventListener("change", async (e) => {
    const f = e.target.files[0]; if (!f) return;
    const fd = new FormData(); fd.append("file", f);
    try { const r = await api("/api/benchmark/upload", { method: "POST", body: fd }); toast(`Loaded ${r.n_items} questions`); loadBench(); }
    catch (err) { toast(err.message, true); }
  });
  $("#run-btn").addEventListener("click", async () => {
    try {
      const r = await api("/api/benchmark/run", { json: { limit: Number($("#run-limit").value) || 0, workers: Number($("#run-workers").value) || 6,
        cpu_threads: Number($("#run-cpu").value) || 2, mode: $("#run-mode").value, judge_model: $("#run-judge").value, batch_api: $("#run-batch").checked,
        crosscheck_n: Number($("#run-xn").value) || 0, crosscheck_model: $("#run-xm").value.trim() } });
      trackJob(r.job); renderBenchStatus(r.job); toast("Benchmark running in the background (resumable)");
    } catch (e) { toast(e.message, true); }
  });
}

function renderBenchStatus(j) {
  const stages = j && j.type === "bench_gen" ? ["generate", "validate", "completed"] : ["intercept", "judge", "features", "train", "completed"];
  $("#bench-status").innerHTML = `<h4>${j && j.type === "bench_gen" ? "Generation" : "Run"} status</h4>${jobPanel(j, stages)}`;
}

async function loadBench() {
  const b = await api("/api/benchmark").catch(() => null);
  S.bench = b;
  if (!b) return;
  renderExamples(b.items);
  const byCat = {}; (b.items || []).forEach((i) => (byCat[i.category] = (byCat[i.category] || 0) + 1));
  $("#bench-items").innerHTML = b.items.length ? `${b.items.length} questions loaded · ${Object.entries(byCat).map(([k, v]) => `${esc(k.toLowerCase())} ${v}`).join(" · ")}` : "No benchmark in this corpus yet.";
  if (b.job) { renderBenchStatus(b.job); if (["queued", "running"].includes(b.job.state) && !S.jobs.has(b.job.id)) trackJob(b.job); }
  renderBenchResults(b);
}

function renderBenchResults(b) {
  const rows = b.rows || [], tp = b.test_predictions || [];
  if (!rows.length) { $("#bench-results").innerHTML = ""; return; }
  const test = rows.filter((r) => r.split === "test");
  const tm = S.brain?.metrics?.test_metrics || {};
  const labels = tp.map((r) => r.label), sf = tp.map((r) => r.s_final), sq = tp.map((r) => r.s_query), cl = tp.map((r) => r.c_llm);
  const rf = rocPoints(labels, sf), rq = rocPoints(labels, sq), rl = rocPoints(labels, cl);
  const mt = b.matching;
  const mtCard = mt ? `<div class="grid g4" style="margin-bottom:16px">
      <div class="card tile"><div class="k">Matches the judge</div><div class="v">${mt.matched_at_0_5 ?? mt["matched_at_0.5"]} / ${mt.n}</div><div class="s">${pct((mt["matched_at_0.5"]) / mt.n, 1)} · out-of-fold, every answer scored by a model that never saw it</div></div>
      <div class="card tile"><div class="k">Baseline: always "hallucinated"</div><div class="v">${mt.majority_baseline} / ${mt.n}</div><div class="s">what raw accuracy gives for free on imbalanced labels</div></div>
      <div class="card tile"><div class="k">Balanced accuracy</div><div class="v">${pct(mt.balanced_accuracy, 1)}</div><div class="s">supported found ${pct(mt.recall_supported, 0)} · hallucinated found ${pct(mt.recall_hallucinated, 0)}</div></div>
      <div class="card tile"><div class="k">Committed decisions correct</div><div class="v">${mt.committed_correct} / ${mt.committed}</div><div class="s">α = 0.10 · abstained on ${mt.abstained} uncertain answers</div></div>
    </div>` : "";
  const xc = b.crosscheck;
  const xcCard = xc ? `<div class="card" style="margin-top:16px"><h4>Label reliability · second judge on a fixed sample</h4>
      <div class="grid g4">
        <div class="tile"><div class="k">Cohen's κ (binary)</div><div class="v">${num(xc.cohen_kappa_binary, 2)}</div><div class="s">${esc(xc.primary_judge)} vs ${esc(xc.second_judge)}</div></div>
        <div class="tile"><div class="k">Raw agreement</div><div class="v">${pct(xc.raw_agreement_binary, 1)}</div><div class="s">n = ${xc.n} answers</div></div>
        <div class="tile"><div class="k">Supported rate</div><div class="v">${pct(xc.supported_rate_primary, 0)} / ${pct(xc.supported_rate_second, 0)}</div><div class="s">primary / second judge</div></div>
        <div class="tile"><div class="k">Disagreements</div><div class="v">${(xc.disagreements || []).length}</div><div class="s">${Object.entries(xc.agreement_by_category || {}).map(([c, v]) => `${esc(c.toLowerCase())} ${pct(v, 0)}`).join(" · ")}</div></div>
      </div><p class="muted" style="font-size:12.5px;margin:10px 0 0">κ ≥ 0.6 is substantial agreement. Training uses the primary judge's labels; this card measures how reliable they are.</p></div>` : "";
  const refNote = b.reference ? `<p class="muted" style="font-size:13px;margin:0 0 14px">Showing the research results of the reference study <b>${esc(b.reference.name)}</b>: this corpus has no benchmark of its own, and its audits use the brain trained on that study.</p>` : "";
  $("#bench-results").innerHTML = `${refNote}${mtCard}
    <div class="grid g5">
      <div class="card tile"><div class="k">Labelled answers</div><div class="v">${rows.length}</div><div class="s">${test.length} in the held-out test split</div></div>
      <div class="card tile"><div class="k">Judge: supported</div><div class="v">${pct(rows.filter((r) => r.label === 1).length / rows.length, 0)}</div><div class="s">share of faithful answers</div></div>
      <div class="card tile"><div class="k">Test AUROC</div><div class="v">${num(tm.S_final?.roc_auc ?? rf.auc, 3)}</div><div class="s">LLM self-confidence: ${num(rl.auc, 3)}</div></div>
      <div class="card tile"><div class="k">Conformal coverage</div><div class="v">${pct(tm.conformal?.coverage, 1)}</div><div class="s">guarantee ≥ ${pct(1 - (S.state.brain?.alpha ?? 0.1), 0)}</div></div>
      <div class="card tile"><div class="k">Selective accuracy</div><div class="v">${pct(tm.conformal?.selective_accuracy, 1)}</div><div class="s">abstains on ${pct(tm.conformal?.abstention_rate, 0)}</div></div>
    </div>
    ${xcCard}
    <div class="grid g2" style="margin-top:16px">
      <div class="card"><h4>AUROC with 95% bootstrap CI · what each feature family adds</h4><div class="chart" id="ch-forest"></div><div class="forest-sig" id="sig"></div></div>
      <div class="card"><h4>ROC · held-out test split</h4><div class="chart" id="ch-roc"></div></div>
      <div class="card"><h4>Reliability · predicted vs observed</h4><div class="chart" id="ch-rel"></div></div>
      <div class="card"><h4>Decisions by question category (test split)</h4><div class="chart" id="ch-cat"></div></div>
      <div class="card"><h4 style="justify-content:space-between">Feature by judge label
        <select id="feat-sel" style="width:auto;padding:4px 8px;font-size:12px">${Object.keys(FEAT).map((k) => `<option value="${k}">${esc(FEAT[k].name)}</option>`).join("")}<option value="hallucination_extent">Hallucination extent (claim-level)</option></select></h4><div class="chart" id="ch-box"></div></div>
    </div>
    <div class="card" style="margin-top:16px"><h4 style="justify-content:space-between">All answers
      <span style="display:flex;gap:8px"><select id="b-cat" style="width:auto;padding:4px 8px;font-size:12px"><option value="">all categories</option>${[...new Set(rows.map((r) => r.category))].map((c) => `<option>${esc(c)}</option>`).join("")}</select>
      <select id="b-split" style="width:auto;padding:4px 8px;font-size:12px"><option value="">all splits</option><option>test</option><option>cal</option><option>train</option></select></span></h4>
      <div class="table-wrap" id="b-table"></div><p class="muted" style="font-size:12px;margin:8px 0 0">Report metrics on the test split only: train/cal answers were seen while fitting.</p></div>`;
  if (tp.length) {
    lines($("#ch-roc"), [{ name: `EpistemeGraph S_final (AUC ${num(rf.auc, 3)})`, color: C.blue, points: rf.points },
      { name: `before pooling S_query (AUC ${num(rq.auc, 3)})`, color: C.aqua, points: rq.points },
      { name: `LLM self-confidence (AUC ${num(rl.auc, 3)})`, color: C.orange, points: rl.points }], { xLabel: "false positive rate", yLabel: "true positive rate" });
    lines($("#ch-rel"), [{ name: "EpistemeGraph S_final", color: C.blue, points: reliabilityPoints(labels, sf) },
      { name: "LLM self-confidence", color: C.orange, points: reliabilityPoints(labels, cl) }], { xLabel: "predicted P(supported)", yLabel: "observed frequency", markers: true });
  } else { $("#ch-roc").innerHTML = $("#ch-rel").innerHTML = `<div class="empty">Available after training.</div>`; }
  const cats = [...new Set(test.map((r) => r.category))];
  stacked($("#ch-cat"), cats.map((c) => ({ label: c.toLowerCase().replace("_", " "), parts: test.filter((r) => r.category === c).reduce((a, r) => ((a[r.decision] = (a[r.decision] || 0) + 1), a), {}) })),
    ["VERIFIED", "ABSTAIN", "HALLUCINATED"].map((d) => ({ key: d, label: `${STATUS[d].icon} ${STATUS[d].label}`, color: STATUS[d].color })));
  const ab = tm.ablations || {};
  const fr = [{ label: "EpistemeGraph S_final", value: tm.S_final?.roc_auc, lo: tm.S_final_ci95?.roc_auc?.[0], hi: tm.S_final_ci95?.roc_auc?.[1], color: C.blue }]
    .concat(Object.entries(ab).map(([k, v]) => ({ label: k.replace(/_/g, " "), value: v.roc_auc, lo: v.ci95?.roc_auc?.[0], hi: v.ci95?.roc_auc?.[1], color: C.ink })))
    .concat([{ label: "LLM self-confidence", value: tm.baselines?.llm_self_confidence?.roc_auc, lo: tm.baselines?.llm_self_confidence?.ci95?.roc_auc?.[0],
      hi: tm.baselines?.llm_self_confidence?.ci95?.roc_auc?.[1], color: C.orange }]).filter((r) => isNum(r.value));
  if (fr.length) forest($("#ch-forest"), fr, { ref: 0.5, refLabel: "chance" }); else $("#ch-forest").innerHTML = `<div class="empty">Available after training.</div>`;
  const sig = tm.significance || {};
  $("#sig").innerHTML = Object.entries(sig).map(([k, v]) => `<div><span>DeLong · ${esc(k.replace(/_/g, " "))}</span><span>ΔAUC ${num(v.delta, 3)} · p ${v.p_value < 0.001 ? "< 0.001" : num(v.p_value, 3)}</span></div>`).join("");
  const drawBox = () => {
    const k = $("#feat-sel").value;
    boxes($("#ch-box"), [{ name: "supported", color: C.blue, values: rows.filter((r) => r.label === 1).map((r) => r[k]).filter(isNum).map(Number) },
      { name: "hallucinated", color: C.orange, values: rows.filter((r) => r.label === 0).map((r) => r[k]).filter(isNum).map(Number) }]);
  };
  $("#feat-sel").addEventListener("change", drawBox); drawBox();
  const drawTable = () => {
    const c = $("#b-cat").value, s = $("#b-split").value;
    const v = rows.filter((r) => (!c || r.category === c) && (!s || r.split === s));
    $("#b-table").innerHTML = `<table><thead><tr><th>ID</th><th>Category</th><th>Judge</th><th>Verdict (out-of-fold)</th><th>S (out-of-fold)</th><th>Match</th><th>Unsupported</th><th>LLM conf.</th><th>Split</th><th>Question</th></tr></thead><tbody>${
      v.slice(0, 600).map((r) => `<tr><td class="mono muted">${esc(r.id)}</td><td class="mono" style="font-size:11px">${esc(r.category)}</td><td class="mono" style="font-size:11px">${esc(r.judge_verdict)}</td>
        <td>${badge(r.oof_decision || r.decision)}</td><td>${bar(isNum(r.oof_S) ? r.oof_S : r.S_final, C.blue)}</td><td class="mono">${r.match === 1 ? "✓" : r.match === 0 ? "✗" : "–"}</td><td>${bar(r.hallucination_extent, C.orange)}</td><td>${bar(r.c_llm, C.orange)}</td><td class="mono muted">${esc(r.split)}</td><td class="q">${esc(r.question)}</td></tr>`).join("")}</tbody></table>`;
  };
  $("#b-cat").addEventListener("change", drawTable); $("#b-split").addEventListener("change", drawTable); drawTable();
}

// ============================================================================ 005 brain
async function loadBrain() {
  S.brain = await api("/api/brain").catch(() => null);
  if (S.bench) renderBenchResults(S.bench);   // benchmark charts depend on the brain's test metrics (avoid a load race)
  const el = $("#brain-body");
  const sm = S.brain?.summary || {}, m = S.brain?.metrics || {};
  if (S.rec) renderWhy($("#why") || document.createElement("div"), S.rec.features || {});
  if (sm.status === "uncalibrated" || !m.leaderboard) {
    el.innerHTML = `<div class="card corner"><div class="grid g-wide" style="align-items:center"><div>
      <h3>No calibrated brain for this corpus yet</h3>
      <p class="ink2" style="margin:0">The decision layer is trained only on real, response-level labels. No synthetic data, no self-labelling. Run the research benchmark once
      (section 005); until then every audit still reports the 13 features and the claim-level hallucination extent, with the verdict marked <b>uncalibrated</b>.</p></div>
      <div style="text-align:right"><a class="btn" href="#benchmark">Go to benchmark →</a></div></div></div>`;
    return;
  }
  const t = m.test_metrics || {}, cb = m.decision_bundle?.conformal || {}, pool = m.decision_bundle?.pool || {};
  el.innerHTML = `
    ${sm.status === "transferred" ? `<div class="card" style="border-color:#fab219;margin-bottom:16px">◐ This corpus has no brain of its own; audits use the brain calibrated on <b>${esc(sm.source_workspace)}</b>.
      Verdicts are usable, but the conformal coverage guarantee is only certified on that corpus. Run the benchmark here to calibrate.</div>` : ""}
    <div class="grid g4">
      <div class="card tile corner"><div class="k">Champion</div><div class="v" style="font-size:26px">${esc(m.winning_model)}</div><div class="s">lowest nested-CV log-loss of 10 families</div></div>
      <div class="card tile"><div class="k">Test AUROC</div><div class="v">${num(t.S_final?.roc_auc, 3)}</div><div class="s">log-loss ${num(t.S_final?.log_loss, 3)} · ECE ${num(t.S_final?.ece, 3)}</div></div>
      <div class="card tile"><div class="k">Labelled answers</div><div class="v">${int(m.n_labelled)}</div><div class="s">${Object.entries(m.split_sizes || {}).map(([k, v]) => `${k} ${v}`).join(" · ")}</div></div>
      <div class="card tile"><div class="k">Promotion</div><div class="v" style="font-size:26px">${m.is_promoted_to_production ? "promoted" : "incumbent"}</div><div class="s">updated ${ago(m.updated_at)}</div></div>
    </div>
    <div class="grid g2" style="margin-top:16px">
      <div class="card"><h4>Tournament · out-of-fold log-loss (lower is better)</h4><div class="chart" id="ch-lb"></div></div>
      <div class="card"><h4>Learned multipliers wⱼ (standardised)</h4><div class="chart" id="ch-w"></div></div>
    </div>
    <div class="grid g2" style="margin-top:16px">
      <div class="card"><h4>Decision layer</h4><div id="pool-tex" class="formula"></div><div id="conf-tex" class="formula"></div>
        <div class="grid g4" style="margin-top:8px">
          <div class="tile"><div class="k">Coverage</div><div class="v" style="font-size:24px">${pct(t.conformal?.coverage)}</div><div class="s">≥ ${pct(1 - (cb.alpha ?? 0.1), 0)} guaranteed</div></div>
          <div class="tile"><div class="k">| supported</div><div class="v" style="font-size:24px">${pct(t.conformal?.coverage_supported)}</div><div class="s">n_cal ${int(cb.n_cal_ver)}</div></div>
          <div class="tile"><div class="k">| hallucinated</div><div class="v" style="font-size:24px">${pct(t.conformal?.coverage_hallucinated)}</div><div class="s">n_cal ${int(cb.n_cal_hall)}</div></div>
          <div class="tile"><div class="k">Abstain</div><div class="v" style="font-size:24px">${pct(t.conformal?.abstention_rate)}</div><div class="s">sel. acc ${pct(t.conformal?.selective_accuracy)}</div></div>
        </div></div>
      <div class="card"><h4>Permutation importance (Δ log-loss)</h4><div class="chart" id="ch-imp"></div></div>
    </div>
    <div class="card" style="margin-top:16px"><h4 style="justify-content:space-between">Ablations · same learner on feature subsets <button class="btn small ghost" id="retrain-btn">Retrain now</button></h4>
      <div class="table-wrap"><table><thead><tr><th>Feature set</th><th>AUROC</th><th>Log-loss</th><th>ECE</th><th>Brier</th></tr></thead><tbody>${
        Object.entries(t.ablations || {}).map(([k, v]) => `<tr><td>${esc(k.replace(/_/g, " "))}</td><td>${num(v.roc_auc)}</td><td>${num(v.log_loss)}</td><td>${num(v.ece)}</td><td>${num(v.brier)}</td></tr>`).join("")}</tbody></table></div></div>`;
  hbar($("#ch-lb"), Object.entries(m.leaderboard).map(([k, v]) => ({ label: k, value: v.log_loss, color: k === m.winning_model ? C.blue : C.gray,
    tip: `<b>${esc(k)}</b><br>log-loss ${num(v.log_loss)} · AUROC ${num(v.roc_auc)}<br>Brier ${num(v.brier)} · ECE ${num(v.ece)}` })).sort((a, b) => a.value - b.value));
  const w = m.learned_feature_weights?.standardised_coefficients || {};
  diverging($("#ch-w"), Object.entries(w).map(([k, v]) => ({ label: FEAT[k]?.short || k, value: v })).sort((a, b) => a.value - b.value), { leftLabel: "← hallucination", rightLabel: "support →" });
  const imp = m.permutation_importance || {};
  hbar($("#ch-imp"), Object.entries(imp).map(([k, v]) => ({ label: FEAT[k]?.short || k, value: Math.max(0, v.mean), color: C.ink, tip: `<b>${esc(FEAT[k]?.name || k)}</b><br>Δ log-loss ${num(v.mean, 4)} ± ${num(v.std, 4)}` })).sort((a, b) => b.value - a.value), { fmt: (v) => num(v, 3) });
  tex($("#pool-tex"), `\\operatorname{logit}S_{\\mathrm{final}}=${num(pool.b0, 2)}${pool.b_query >= 0 ? "+" : ""}${num(pool.b_query, 2)}\\,\\operatorname{logit}S_{\\mathrm{query}}${pool.b_kg >= 0 ? "+" : ""}${num(pool.b_kg, 2)}\\,\\operatorname{logit}S_{\\mathrm{KG}}(q)`);
  tex($("#conf-tex"), `\\Gamma(x)=\\{c:1-\\hat p_c(x)\\le q_c\\},\\ q_{\\text{hall}}=${num(cb.q_hall, 4)},\\ q_{\\text{ver}}=${num(cb.q_ver, 4)},\\ \\alpha=${num(cb.alpha, 2)}`);
  $("#retrain-btn").addEventListener("click", async () => { try { const r = await api("/api/brain/retrain", { method: "POST" }); trackJob(r.job); toast("Retraining in the background"); } catch (e) { toast(e.message, true); } });
}

// ============================================================================ 006 method
function renderMethod() {
  const cards = [
    ["Phase 1 · KG fidelity", "\\begin{gathered}S_{\\mathrm{KG}}=\\big(\\Phi_{\\mathrm{prov}}\\Phi_{\\mathrm{cov}}\\Phi_{\\mathrm{topo}}\\Phi_{\\mathrm{cons}}\\big)^{1/4}\\\\ \\Phi_{\\mathrm{topo}}=1-\\tfrac{k(G)-1}{|V|-1}\\end{gathered}", "Unweighted geometric mean of precision, coverage, connectivity and consistency: no tuned exponents, zero if any pillar fails."],
    ["Phase 1 · measured accuracy", "\\hat p\\pm\\text{Wilson}:\\ \\frac{\\hat p+\\frac{z^2}{2n}\\pm z\\sqrt{\\frac{\\hat p(1-\\hat p)}{n}+\\frac{z^2}{4n^2}}}{1+\\frac{z^2}{n}}", "Extrinsic precision of a uniform random sample of triples, each checked against its source passage."],
    ["Sheaf energy", "\\mathcal E_{\\mathcal F}=\\frac{\\frac1{|E'|}\\sum_{e}\\lVert x_v-x_u-t_r\\rVert^2}{\\frac1{|E|}\\sum_{e}\\lVert x_v-x_u-t_r\\rVert^2}", "Dirichlet energy of a translational knowledge sheaf (TransE as a sheaf), relative to the ground truth itself."],
    ["Forman–Ricci shift", "\\begin{gathered}F(u,v)=4-d_u-d_v+3\\,\\#\\triangle(u,v)\\\\ \\Delta\\kappa=\\log\\!\\big(1+W_1(F_G,F_{G\\cup E'})\\big)\\end{gathered}", "How far the curvature distribution moves when the claims are grafted onto the graph; zero iff nothing new is claimed."],
    ["Spectral divergence", "D_{\\mathrm{spec}}=\\tfrac12W_1\\big(\\operatorname{spec}L_{\\mathrm{sym}}(G'),\\operatorname{spec}L_{\\mathrm{sym}}(H)\\big)\\in[0,1]", "Global shape of the claim graph versus the 2-hop shadow of the knowledge graph on the same entities."],
    ["Bayesian provenance", "\\pi_{\\mathrm{prov}}=\\frac{1+\\sum_e s_e}{2+|E'|},\\ s_e=\\max_w g(u,w)g(w,v)", "Posterior mean support rate, using the evidence-grounding score of every KG edge on the path."],
    ["Evidence & novelty", "\\begin{gathered}\\sigma_{\\mathrm{ent}}=\\tfrac{1}{|C|}\\textstyle\\sum_c\\max_s\\cos(\\phi(c),\\phi(s))\\\\ \\nu_{\\mathrm{ooc}}=\\frac{\\sum_t\\mathrm{idf}(t)\\,\\mathbb 1[t\\notin V]}{\\sum_t\\mathrm{idf}(t)}\\end{gathered}", "Semantic support in raw sentences and IDF-weighted out-of-corpus vocabulary, on asserted claims only."],
    ["NLI consistency (SummaC)", "\\gamma(c)=\\max_{s\\in\\mathcal S_c}\\big[P_{\\mathrm{ent}}(s\\Rightarrow c)-P_{\\mathrm{con}}(s\\Rightarrow c)\\big]", "A DeBERTa-v3 NLI cross-encoder reads each claim against its nearest source sentences: stated vs. merely on-topic, on your CPU, at no API cost."],
    ["Claim-level MIL", "\\begin{gathered}a_i=\\operatorname{softmax}_i\\big(w^\\top[\\tanh(Vh_i)\\odot\\sigma(Uh_i)]\\big)\\\\ \\mathcal L=\\mathrm{BCE}(\\ell,y)+\\lambda\\,\\mathrm{BCE}\\big(\\textstyle\\prod_i(1-r_i),y\\big)\\end{gathered}", "An answer is a bag of claims; a gated-attention network with a noisy-OR head learns WHICH claim is hallucinated from answer-level labels only."],
    ["Learned KG accuracy", "\\hat p_{\\mathrm{MA}}=\\tfrac1N\\sum_t\\hat m(v_t)+\\tfrac1n\\sum_{t\\in s}\\big(y_t-\\hat m^{(-t)}(v_t)\\big)", "Model-assisted survey estimator: a learned triple-correctness model spreads the audit over all N triples; unbiased, with a tighter CI."],
    ["Learned pool", "\\operatorname{logit}S_{\\mathrm{final}}=b_0+b_1\\operatorname{logit}S_{\\mathrm{query}}+b_2\\operatorname{logit}S_{\\mathrm{KG}}(q)", "Replaces the fixed 0.2/0.8 rule, which is the special case b₀=0, b₁=η, b₂=1−η."],
    ["Conformal decision", "\\Pr\\big(y\\in\\Gamma(x)\\mid y=c\\big)\\ge1-\\alpha", "Mondrian split-conformal sets: one class → verdict, both/none → abstain. Replaces the >0.85 / ≤0.5 thresholds."],
  ];
  $("#math").innerHTML = cards.map(([h], i) => `<div class="card math-card"><h4>${esc(h)}</h4><div data-tex="${i}"></div><p>${esc(cards[i][2])}</p></div>`).join("");
  setTimeout(motion, 400);
  $$("#math [data-tex]").forEach((el) => tex(el, cards[+el.dataset.tex][1]));
  $("#paper-btn").addEventListener("click", async () => {
    try { await api("/api/paper/export", { method: "POST" }); location.href = "/api/paper/overleaf.zip"; toast("Overleaf bundle ready"); }
    catch (e) { toast(e.message, true); }
  });
}

// ============================================================================ access settings
function bindSettings() {
  const open = () => {
    const pub = S.state?.access?.public_mode;
    $("#settings-mode").textContent = pub
      ? `Public mode: anonymous visitors get ${S.state.access.rate_limit_per_hour} audits/hour. Admin actions need the token${S.state.is_admin ? " (you are admin)" : ""}.`
      : "Local mode: everything is enabled on this machine; no token needed.";
    $("#set-admin").value = getSetting("admin");
    $("#set-key").value = getSetting("openai");
    $("#settings").hidden = false;
  };
  $("#settings-btn").addEventListener("click", open);
  $("#set-cancel").addEventListener("click", () => ($("#settings").hidden = true));
  $("#settings").addEventListener("click", (e) => { if (e.target.id === "settings") $("#settings").hidden = true; });
  $("#set-save").addEventListener("click", () => {
    setSetting("admin", $("#set-admin").value.trim());
    setSetting("openai", $("#set-key").value.trim());
    $("#settings").hidden = true;
    toast("Saved in this browser only");
    loadAll();
  });
}

// ============================================================================ opening chapters (hero + premise + signals)
function bindOpening() {
  const op = $("#opening");
  if (!op) return;
  let queued = false;
  const update = () => {
    queued = false;
    const span = Math.max(1, op.offsetHeight - innerHeight * 0.55);
    hero.setProgress(Math.max(0, Math.min(1.2, scrollY / span)));
    scrubWords();
  };
  const onScroll = () => { if (!queued) { queued = true; requestAnimationFrame(update); } };
  addEventListener("scroll", onScroll, { passive: true });
  addEventListener("resize", onScroll);
  update();
}

/** Split .scrub paragraphs into words that light up as they scroll through the viewport. */
function initScrub() {
  $$(".scrub").forEach((el) => {
    el.innerHTML = el.textContent.trim().split(/\s+/).map((w) => `<span class="w">${esc(w)}</span>`).join(" ");
  });
  if (document.documentElement.classList.contains("static")) $$(".scrub .w").forEach((w) => w.classList.add("on"));
}
function scrubWords() {
  if (document.documentElement.classList.contains("static")) return;
  $$(".scrub").forEach((el) => {
    const r = el.getBoundingClientRect();
    if (r.bottom < -50 || r.top > innerHeight + 50) return;
    const p = (innerHeight * 0.88 - r.top) / (innerHeight * 0.5 + r.height * 0.4);
    const ws = el.children, k = Math.round(Math.max(0, Math.min(1, p)) * ws.length);
    for (let i = 0; i < ws.length; i++) ws[i].classList.toggle("on", i < k);
  });
}

