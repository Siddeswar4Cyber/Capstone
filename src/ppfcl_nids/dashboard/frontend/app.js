import {createServerView} from "./fl-server.js";
import {createClientDashboard} from "./client-dashboard.js";

const pages = {
  overview: ["Overview", "◈", "Session activity and verified research evidence"],
  server: ["Federated Server", "▤", "Session controls, FedProx receipts and aggregation"],
  "client-a": ["Client A", "▦", "Manual control for C1–C20"],
  "client-b": ["Client B", "▦", "Manual control for D1–D20"],
  models: ["Model Registry", "◇", "Checkpoint integrity and session lineage"],
  privacy: ["Privacy & Security", "◉", "Execution scope, release boundaries and restricted results"],
  health: ["System Health", "∿", "Infrastructure probes and operational alerts"],
  audit: ["Audit History", "▥", "Recorded transitions, recoveries and rejected actions"],
  "server-tools": ["Synthetic Session Console", "▤", "Session controls, FedProx receipts and aggregation"],
  "client-a-tools": ["Client A Synthetic Console", "▦", "Manual synthetic session controls"],
  "client-b-tools": ["Client B Synthetic Console", "▦", "Manual synthetic session controls"],
};
const researchClient = () => ({"client-a": "A", "client-b": "B"})[ui.page];
const syntheticClient = () => ({"client-a-tools": "A", "client-b-tools": "B"})[ui.page];
const pageAtLocation = () => location.pathname === "/" ? "server" :
  Object.hasOwn(pages, location.pathname.slice(1)) ? location.pathname.slice(1) : "overview";
const app = document.querySelector("#app");
const ui = {data: null, csrf: "", session: "", chunk: {}, busy: false, chart: {}, query: "", filter: "",
  mode: "LOCAL_SYNTHETIC", workers: {}, checkingWorkers: new Set(), refreshId: 0,
  analyticsClient: "", analyticsHours: "", monitor: null, monitoring: false,
  page: pageAtLocation()};
const esc = value => String(value ?? "—").replaceAll("FedAsync", "FedProx").replace(/[&<>"']/g, char => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
}[char]));
const short = value => value ? `${value.slice(0, 9)}…` : "—";
const time = value => value && Number.isFinite(Date.parse(value)) ?
  new Date(value).toLocaleString("en-IN", {dateStyle: "medium", timeStyle: "medium", timeZone: "Asia/Kolkata"}) + " IST" : "UNKNOWN";
const fmt = (value, digits = 2) => Number.isFinite(value) ? value.toFixed(digits) : "NOT MEASURED";
const gib = value => Number.isFinite(value) ? `${fmt(value / 2**30)} GiB` : "NOT MEASURED";
const seconds = value => Number.isFinite(value) ? `${fmt(value, 3)} s` : "NOT MEASURED";
const percent = value => Number.isFinite(value) ? `${fmt(value, 1)}%` : "NOT MEASURED";
const badge = (label, color = "") => `<span class="badge ${color}">${esc(label)}</span>`;
const hash = value => `<span class="hash">${esc(value)}</span>`;
const empty = message => `<div class="empty">${esc(message)}</div>`;
const button = (label, action, {disabled = false, css = "", extra = ""} = {}) =>
  `<button class="button ${css}" type="button" data-action="${action}" ${disabled ? "disabled" : ""} ${extra}>${esc(label)}</button>`;
const panel = (title, source, body, controls = "") => `<section class="panel"><div class="panel-head"><div>
  <h2>${esc(title)}</h2><span class="source">${esc(source)}</span></div>${controls}</div>${body}</section>`;
const stat = (label, value, source) => `<div class="card"><div class="label">${esc(label)}</div>
  <div class="value">${esc(value)}</div><small>${esc(source)}</small></div>`;
const table = (headings, rows, message = "No records in this view.") => rows.length ?
  `<div class="table-wrap"><table><thead><tr>${headings.map(h => `<th scope="col">${esc(h)}</th>`).join("")}
   </tr></thead><tbody>${rows.map(row => `<tr>${row.map(cell => `<td>${cell}</td>`).join("")}</tr>`).join("")}
   </tbody></table></div>` : empty(message);
const list = pairs => `<dl class="stat-list">${pairs.map(([key, value]) => `<dt>${esc(key)}</dt><dd>${value}</dd>`).join("")}</dl>`;
const tone = state => state === "COMPLETED" || state === "ACTIVE" ? "green" :
  state === "FAILED" ? "red" : state === "ACCEPTED_AWAITING_EWC" || state === "ABANDONING" ? "amber" :
  ["PENDING", "CLOSED"].includes(state) ? "" : "cyan";
const version = value => Number.isInteger(value) ? `v${value}` : "—";
const brand = () => `<div class="brand"><span class="brand-mark" aria-hidden="true">◇</span>
  <div><strong>Federated IDS</strong><small>Research console</small></div></div>`;
const selected = () => ui.data?.session;
const active = () => selected()?.status === "ACTIVE";
const remote = () => selected()?.execution_mode === "REMOTE_SYNTHETIC";
const modeName = mode => mode?.replaceAll("_", " ") || "No session";
const source = () => selected() ? `${modeName(selected().execution_mode)} · SQLite` : "Dashboard synthetic session · empty";
const dashboardPicker = () => `<label class="fl-page-picker">Dashboard
  <select data-dashboard-picker>${["server", "client-a", "client-b"].map(page =>
    `<option value="${page}" ${ui.page === page ? "selected" : ""}>${esc(pages[page][0])}</option>`).join("")}</select></label>`;

async function api(path, payload, signal) {
  const response = await fetch(path, payload === undefined ? {signal, headers: {"X-CSRF-Token": ui.csrf}} : {
    method: "POST", signal, headers: {"Content-Type": "application/json", "X-CSRF-Token": ui.csrf},
    body: JSON.stringify(payload),
  });
  const value = await response.json();
  if (!response.ok) throw Object.assign(new Error(value.error), {status: response.status});
  return value;
}

async function refresh(poll = false) {
  const id = ++ui.refreshId;
  try {
    const scoped = researchClient();
    const next = scoped ? await api("/api/client-" + scoped.toLowerCase()) :
      await api("/api/state" + (ui.session ? `?session=${encodeURIComponent(ui.session)}` : ""));
    if (id !== ui.refreshId) return;
    if (scoped) {
      ui.data = {clientOnly: true};
      ui.csrf = next.csrf;
      ui.localOnly = next.local_only;
      render();
      return;
    }
    const changed = ["sessions", "session", "models", "chunks", "runs", "updates", "revisions", "audit", "continual_jobs"]
      .some(key => JSON.stringify(next[key]) !== JSON.stringify(ui.data?.[key])) ||
      JSON.stringify(next.analytics.alerts.map(row => row.code)) !==
        JSON.stringify(ui.data?.analytics?.alerts.map(row => row.code));
    ui.data = next;
    ui.localOnly = next.local_only;
    ui.csrf = ui.data.csrf;
    if (!poll || changed && ui.page !== "server") render();
  } catch (error) {
    if (id !== ui.refreshId) return;
    if (error.status === 401) { ui.data = null; login(); }
    else if (error.status === 404 && ui.session) { ui.session = ""; await refresh(); }
    else if (!ui.data) login("Dashboard data unavailable. Try signing in again.");
    else notice(error.message, true);
  }
}

const flServer = createServerView({
  app, api, esc, time, gib, table, list, dashboardPicker,
  expired: () => { ui.data = null; login(); },
  active: () => Boolean(ui.data) && ui.page === "server",
  localOnly: () => ui.localOnly,
});
const clientDashboard = createClientDashboard({
  app, api, esc, time, gib, table, list, dashboardPicker, scope: researchClient,
  expired: () => { ui.data = null; login(); },
  csrf: token => { ui.csrf = token; },
  localOnly: () => ui.localOnly,
});

async function checkWorker(role) {
  if (ui.checkingWorkers.has(role) || !ui.data?.remote_enabled) return;
  ui.checkingWorkers.add(role);
  try {
    ui.workers[role] = await api(`/api/workers/${role}`);
  } catch {
    ui.workers[role] = {state: "UNAVAILABLE", last_verified_at: ui.workers[role]?.last_verified_at};
  } finally {
    ui.checkingWorkers.delete(role);
    if (ui.data && (ui.page === "health" || syntheticClient() === role)) render();
  }
}

function checkVisibleWorkers() {
  if (ui.page === "health" || syntheticClient()) void loadMonitor();
  if (remote() && syntheticClient()) void checkWorker(syntheticClient());
}

async function loadMonitor(refresh = false) {
  if (ui.monitoring) return;
  ui.monitoring = true;
  if (refresh && ui.page === "health") render();
  try {
    ui.monitor = await api("/api/monitor", refresh ? {} : undefined);
    ui.monitorError = false;
  } catch {
    ui.monitorError = true;
  } finally {
    ui.monitoring = false;
    if (ui.data && (ui.page === "health" || syntheticClient())) render();
  }
}

function login(message = "") {
  flServer.reset();
  clientDashboard.reset();
  app.innerHTML = `<main id="content" class="login"><section class="panel">${brand()}<h1>Operator sign-in</h1>
    <p class="muted">Enter the access key shown in the dashboard startup terminal.</p>
    <form id="login"><label for="key">Operator access key</label>
      <input class="field" id="key" type="password" required autocomplete="off">
      <button class="button primary" type="submit">Open research console</button>
      <output role="status">${esc(message)}</output>
    </form></section></main>`;
}

function notice(message, error = false) {
  const output = document.querySelector("#feedback");
  if (!output) return;
  output.textContent = message;
  output.classList.toggle("error", error);
  output.hidden = false;
  clearTimeout(notice.timer);
  notice.timer = setTimeout(() => { output.hidden = true; }, 5500);
}

function chart(id, label, items, mode = "line") {
  const series = items.filter(item => Number.isFinite(item.value));
  if (!series.length) return empty("No measured values in this view.");
  ui.chart[id] = {items: series, label, mode};
  return `<div class="chartbox"><canvas tabindex="0" role="img" data-chart="${id}" aria-label="${esc(label)}. Use arrow keys to inspect values.">
    ${series.map(i => `${esc(i.name)} ${esc(i.value)}`).join("; ")}
    </canvas><output class="chart-tooltip" aria-live="polite">${esc(series[0].name)} · ${esc(series[0].value)}</output></div>`;
}

function draw(canvas, index = 0) {
  const series = ui.chart[canvas.dataset.chart];
  if (!series) return;
  const {items, label, mode} = series;
  const width = Math.max(canvas.clientWidth, 140), height = 206, ratio = window.devicePixelRatio || 1;
  canvas.width = width * ratio; canvas.height = height * ratio;
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  const left = 42, bottom = 174, plotWidth = width - left - 14, top = 13;
  const max = Math.max(...items.map(item => item.value), 0.001);
  const x = i => left + (i + 0.5) * plotWidth / items.length;
  const y = value => bottom - value * (bottom - top) / max;
  ctx.font = '10px Consolas, monospace';
  for (let step = 0; step <= 4; step++) {
    const value = max * step / 4, pos = y(value);
    ctx.strokeStyle = "#25354d"; ctx.beginPath(); ctx.moveTo(left, pos); ctx.lineTo(width - 12, pos); ctx.stroke();
    ctx.fillStyle = "#9bafcb"; ctx.textAlign = "right"; ctx.fillText(Number(value.toPrecision(3)), left - 6, pos + 3);
  }
  if (mode === "line") {
    ctx.beginPath();
    items.forEach((item, i) => i ? ctx.lineTo(x(i), y(item.value)) : ctx.moveTo(x(i), y(item.value)));
    ctx.strokeStyle = "#63e1e9"; ctx.lineWidth = 2; ctx.stroke();
    items.forEach((item, i) => {
      ctx.beginPath(); ctx.arc(x(i), y(item.value), i === index ? 5 : 3, 0, 2 * Math.PI);
      ctx.fillStyle = i === index ? "#e9f1fb" : "#63e1e9"; ctx.fill();
    });
  } else {
    items.forEach((item, i) => {
      const bar = Math.min(48, plotWidth / items.length * 0.58);
      ctx.fillStyle = i === index ? "#63e1e9" : "#4d7eb6";
      if (mode === "paired") {
        ctx.fillRect(x(i), y(item.value), bar / 2, Math.max(1, bottom - y(item.value)));
        ctx.fillStyle = "#b89eea";
        ctx.fillRect(x(i) - bar / 2, y(item.base), bar / 2, Math.max(1, bottom - y(item.base)));
      } else ctx.fillRect(x(i) - bar / 2, y(item.value), bar, Math.max(1, bottom - y(item.value)));
    });
  }
  ctx.textAlign = "center"; ctx.fillStyle = "#9bafcb";
  const stride = Math.max(1, Math.ceil(items.length / Math.max(1, Math.floor(plotWidth / 48))));
  items.forEach((item, i) => { if (i % stride === 0) ctx.fillText(item.name, x(i), bottom + 17); });
  canvas.dataset.index = index;
  const item = items[index];
  canvas.nextElementSibling.textContent = mode === "paired" ? `${item.name} · base v${item.base} · accepted v${item.value}` :
    `${item.name} · ${fmt(item.value)} ${label}${item.at ? " · " + time(item.at) : ""}`;
}

const inRange = stamp => !ui.analyticsHours || (Number.isFinite(Date.parse(stamp)) &&
  Date.parse(stamp) >= Date.now() - Number(ui.analyticsHours) * 3600000);
const rangeControl = () => `<select id="analytics-range" aria-label="Analytics time range">
  ${[["", "All session time"], ["1", "Last hour"], ["24", "Last 24 hours"], ["168", "Last 7 days"]]
    .map(([key,label]) => `<option value="${key}" ${key === ui.analyticsHours ? "selected" : ""}>${label}</option>`).join("")}</select>`;
const countRows = (items, getKey) => Object.entries(items.reduce((counts, row) => {
  const key = getKey(row); counts[key] = (counts[key] || 0) + 1; return counts;
}, {}));
const alertNames = {
  NODE_UNREACHABLE:"Infrastructure probe failed", LOW_DISK:"Low available disk", HIGH_MEMORY:"High memory utilization",
  STALE_MONITORING:"Stale infrastructure sample", CHUNK_FAILED:"Chunk failed", CHUNK_INCOMPLETE:"Chunk needs attention",
  EWC_PENDING:"EWC reconciliation pending", CHECKPOINT_VERIFICATION_FAILED:"Checkpoint verification failed",
  BASELINE_VERIFICATION_FAILED:"Research v0 verification failed", PROLONGED_WORKER:"Prolonged worker operation",
};
const alertsTable = rows => table(["Alert", "Scope", "Evidence time", "Measured value"], rows.map(row => [
  esc(alertNames[row.code]), esc(row.node ?? (row.client ? `Client ${row.client} · ${row.chunk}` :
    row.version == null ? "Research global v0" : version(row.version))), esc(time(row.at)),
  Number.isFinite(row.value) ? esc(`${fmt(row.value)} ${row.unit}`) : "—",
]), "No active operational alerts in this view.");
const performanceTable = (summary, labels) => table(["Phase", "Measured runs", "Minimum", "Maximum", "Average"],
  Object.entries(labels).map(([key, label]) => [esc(label), esc(summary[key]?.count ?? 0),
    seconds(summary[key]?.minimum), seconds(summary[key]?.maximum), seconds(summary[key]?.average)]));

function stats() {
  const data = ui.data;
  return `<div class="kpis">${stat("Global version", version(selected()?.head_version), source())}
    ${stat("Verified acceptances", selected() ? data.analytics.updates.length : "—", source())}
    ${stat("Completed chunks", selected() ? data.chunks.filter(c => c.state === "COMPLETED").length : "—", source())}
    ${stat("Pending chunks", selected() ? data.chunks.filter(c => c.state === "PENDING").length : "—", source())}</div>`;
}

function timeline(models, historical = false) {
  if (!models.length) return empty("No committed versions for this view.");
  return `<ol class="timeline">${models.map(model => {
    const id = version(historical ? model.accepted_version : model.version);
    return `<li><strong>${id}</strong> · ${esc(model.chunk_id || "Research root")}
      <small>${historical ? "Recorded Task 9E successor" : time(model.created_at)}
      ${model.staleness == null ? "" : ` · staleness ${esc(model.staleness)} · α ${fmt(model.aggregation_weight)}`}</small></li>`;
  }).join("")}</ol>`;
}

function versionChart(models, historic = false) {
  return chart("versions", "global version", models.map(row => ({
    name: historic ? String(row.chunk_id) : version(row.version),
    value: historic ? row.accepted_version : row.version,
  })));
}

function updateTable(updates) {
  return table(["Version", "Client", "Chunk", "Base", "Staleness", "Alpha", "Accepted", "Published", "Receipt ID", "Update SHA-256"],
    updates.map(r => [version(r.accepted_version), esc(r.client_id), esc(r.chunk_id), version(r.base_version),
      esc(r.staleness), fmt(r.aggregation_weight), esc(time(r.accepted_at)), esc(time(r.published_at)),
      hash(r.update_id), hash(r.payload_sha256)]),
    "No accepted updates in this session.");
}

function recoveryNotice() {
  const runs = ui.data.runs.filter(r => ["RUNNING", "SUBMITTED", "ACCEPTED_AWAITING_EWC", "ABANDONING"].includes(r.state));
  return runs.length ? `<div class="note">${runs.length} unfinished execution(s) require inspection before closure.
    Open the owning client page for recovery.</div>` : "";
}

function overview() {
  const data = ui.data, historical = !selected();
  const models = historical ? data.historical.runs : data.models;
  const label = historical ? "Task 9E · historical synthetic verification" : source();
  const progress = ["A", "B"].map(role => {
    const chunks = data.chunks.filter(c => c.client_id === role);
    const complete = chunks.filter(c => c.state === "COMPLETED").length;
    return chunks.length ? `<div class="client-progress"><strong>Client ${role}</strong><span>${complete}/${chunks.length}</span>
      <progress aria-label="Client ${role} completed chunks" value="${complete}" max="${chunks.length}"></progress></div>` :
      `<div class="client-progress"><strong>Client ${role}</strong><span>No session</span></div>`;
  }).join("");
  return `<section class="hero"><div><span class="eyebrow">Pinned research baseline</span><h2>Immutable global v0</h2>
    <div class="mono">${esc(data.baseline.contract_id)} · ${esc(data.baseline.architecture.join(" → "))}</div>
    <div class="hash">${esc(data.baseline.sha256)}</div></div>
    <div>${badge(data.baseline.verified ? "SHA-256 verified" : "Verification unavailable", data.baseline.verified ? "green" : "amber")}</div></section>
    ${stats()}${recoveryNotice()}<div class="split">
      ${panel("Federated version timeline", label, versionChart(models, historical))}
      ${panel("Client progress", source(), progress + list([
        ["Active session", esc(data.sessions.find(s => s.status === "ACTIVE")?.session_id || "None")],
        ["Active execution mode", esc(modeName(data.sessions.find(s => s.status === "ACTIVE")?.execution_mode))],
        ["Windows coordinator", badge("Current API response", "green")],
        ["VM reachability", esc(Object.keys(data.historical.vm_checks).length ? "Recorded checks available" : "Unavailable")],
      ]))}
    </div><div class="split">
      ${panel("Recent verified updates", source(), updateTable([...data.analytics.updates].reverse().slice(0, 5)))}
      ${panel("Research scope", "Release policy · synthetic", "<p class='muted'>Remote synthetic sessions use the assigned Ubuntu workers. Real training, production release and Organization inference are disabled.</p>")}
    </div>${panel("Operational alerts", "Selected session · verified records", alertsTable(data.analytics.alerts))}`;
}

function realResearch(jobs) {
  const records = jobs ?? [];
  const receipts = records.flatMap(job => job.receipt ? [job.receipt] : []).sort((a,b) => a.version-b.version);
  const events = records.flatMap(job => (job.export_events ?? []).map(event => ({...event, ...{client:job.client, chunk:job.chunk}})))
    .sort((a,b) => Date.parse(a.at)-Date.parse(b.at));
  return panel("Real FedProx research lineage", "Unvalidated research checkpoints", table(
    ["Update", "Client / chunk", "Base", "Parent → successor", "Staleness", "Alpha", "Checkpoint SHA-256"],
    receipts.map(r => [hash(r.job),esc(`${r.client} / ${r.chunk}`),version(r.base_version),
      esc(`${version(r.parent_version)} → ${version(r.version)}`),esc(r.staleness),esc(fmt(r.alpha)),hash(r.checkpoint_sha256)]),
    "No accepted real update recorded.") + table(["Time", "Client / chunk", "Event", "Verified bytes"], events.map(event =>
      [esc(time(event.at)),esc(`${event.client} / ${event.chunk}`),esc(event.event),esc(event.bytes)]), "No real export events recorded."));
}

function serverPage() {
  const data = ui.data;
  const analytics = data.analytics;
  const includeClient = role => !ui.analyticsClient || role === ui.analyticsClient;
  const updates = analytics.updates.filter(row => includeClient(row.client_id) && inRange(row.accepted_at));
  const plotted = updates;
  const models = data.models.filter(row => (!ui.analyticsClient || row.client_id === ui.analyticsClient)
    && inRange(row.created_at));
  const historicTimes = data.historical.task10b_runs.map(row => Date.parse(row.at)).filter(Number.isFinite);
  const historicRange = historicTimes.length ?
    ` · ${time(Math.min(...historicTimes))} to ${time(Math.max(...historicTimes))}` : "";
  const metric = ui.serverMetric || "aggregation_weight";
  const labels = {aggregation_weight: "mixing alpha", staleness: "staleness", contribution: "accepted updates",
    completed: "completed chunks"};
  const items = metric === "completed" ?
    (selected() ? ["A", "B"].filter(includeClient).map(role => ({name: `Client ${role}`,
      value: analytics.runs.filter(c => c.client_id === role && c.state === "COMPLETED" && inRange(c.completed_at)).length})) : []) :
    metric === "contribution" ?
    (selected() ? ["A", "B"].filter(includeClient).map(role => ({
      name: `Client ${role}`,
      value: plotted.filter(r => r.client_id === role).length,
    })) : []) : plotted.map(r => ({name: version(r.accepted_version), value: r[metric], at: r.accepted_at}));
  return `${realResearch(data.continual_jobs)}${stats()}${recoveryNotice()}${panel("Session control", source(), list([
    ["Session", esc(selected()?.session_id || "No session selected")],
    ["Status", selected() ? badge(selected().status, tone(selected().status)) : esc("Unavailable")],
    ["Execution mode", esc(modeName(selected()?.execution_mode))],
    ["Created", esc(time(selected()?.created_at))], ["Closed", esc(time(selected()?.closed_at))],
  ]), `<div class="actions"><select id="execution-mode" aria-label="New demo execution mode"
    ${data.sessions.some(s => s.status === "ACTIVE") || ui.busy ? "disabled" : ""}>
    <option value="LOCAL_SYNTHETIC" ${ui.mode === "LOCAL_SYNTHETIC" ? "selected" : ""}>LOCAL SYNTHETIC</option>
    <option value="REMOTE_SYNTHETIC" ${ui.mode === "REMOTE_SYNTHETIC" ? "selected" : ""}
    ${!data.remote_enabled ? "disabled" : ""}>REMOTE SYNTHETIC${data.remote_enabled ? "" : " · server opt-in required"}</option>
    </select>${button("Start Demo", "start", {css:"primary", disabled: ui.busy || !data.baseline.verified || data.sessions.some(s => s.status === "ACTIVE")})}
    ${button("Close Demo", "close", {css:"danger", disabled:!active()})}</div>`)}
    <div class="filters">${rangeControl()}<select id="analytics-client" aria-label="Analytics client">
      ${[["","All clients"],["A","Client A"],["B","Client B"]].map(([key,label]) =>
        `<option value="${key}" ${ui.analyticsClient === key ? "selected" : ""}>${label}</option>`).join("")}
    </select><span class="source">Updated ${esc(time(analytics.updated_at))}</span></div>
    <div class="split">${panel("FedProx aggregation", analytics.source, chart("mix", labels[metric], items, "bar"),
      `<select id="server-metric" aria-label="Aggregation metric">${Object.entries(labels).map(([key,label]) =>
        `<option value="${key}" ${key === metric ? "selected" : ""}>${esc(label)}</option>`).join("")}</select>`)}
    ${panel("Immutable version timeline", source(), timeline(models))}</div>
    <div class="split">${panel("Global version progression", "Verified acceptance order · SQLite",
      chart("server-versions", "accepted version", updates.map(r => ({name:r.chunk_id,value:r.accepted_version,at:r.accepted_at}))))}
    ${panel("Staleness distribution", "Verified receipts grouped by number of versions behind",
      chart("staleness-distribution", "accepted updates", countRows(updates, r => r.staleness)
        .map(([name,value]) => ({name,value})), "bar"))}</div>
    ${panel("Training base versus acceptance", "Purple: training base · blue: accepted global version",
      chart("base-comparison", "versions", updates.map(r =>
        ({name:r.chunk_id,value:r.accepted_version,base:r.base_version})), "paired"))}
    ${panel("Verified acceptance timeline", source(), updateTable(updates))}
    ${panel("Acceptance outcomes", "Whole selected session · one count per execution or verified receipt",
      table(["Outcome","Count"], Object.entries(analytics.outcomes).map(([label,value]) =>
        [esc(label.replaceAll("_"," ")),esc(value)])) +
        (analytics.unverified_receipts ? `<p class="note">${analytics.unverified_receipts} receipt(s) excluded from verified analytics.</p>` : ""))}
    ${panel("Session performance", "Whole selected session · elapsed transition intervals can include recovery",
      performanceTable(analytics.performance, analytics.phases) +
      '<p class="source">Standalone preparation, Fisher, transfer, FedProx commit, checkpoint write and EWC promotion times: NOT MEASURED for this session.</p>')}
    ${panel("Historical distributed performance", "Task 10B · isolated synthetic verification" + historicRange,
      performanceTable(data.historical.task10b_performance, data.historical.task10b_phases))}
    ${panel("Recovery and session audit", "Session and action records · SQLite and dashboard audit", auditTable(
      analytics.recovery.filter(row => includeClient(row.client_id) && inRange(row.time)).slice(0, 30)))}
    ${panel("Operational alerts", "Whole selected session", alertsTable(analytics.alerts))}`;
}

function clientPage(role) {
  const data = ui.data, chunks = data.chunks.filter(c => c.client_id === role);
  const allRuns = data.runs.filter(r => r.client_id === role);
  const storedChunk = selected() && sessionStorage.getItem(`chunk-${selected().session_id}-${role}`);
  const chunk = chunks.find(c => c.chunk_id === (ui.chunk[role] || storedChunk));
  const run = allRuns.find(r => r.execution_id === chunk?.execution_id);
  const state = chunk?.state || "";
  const worker = ui.workers[role];
  const online = worker?.state === "VERIFIED";
  const approvedExpired = remote() && state === "APPROVED" &&
    Date.now() >= Date.parse(run.approved_at) + 15 * 60 * 1000;
  const operating = allRuns.some(r => r.operation_in_progress);
  const enabled = action => !ui.busy && !operating && ({
    select: active() && ["PENDING", "FAILED"].includes(state) &&
      !allRuns.some(r => !["COMPLETED", "FAILED"].includes(r.state)),
    approve: active() && state === "SELECTED",
    run: active() && state === "APPROVED" && (!remote() || (worker?.ready && !approvedExpired)),
    submit: active() && ["STAGED", "SUBMITTED"].includes(state),
    complete: active() && ["SUBMITTED", "ACCEPTED_AWAITING_EWC"].includes(state),
    recover: active() && ["RUNNING", "SUBMITTED", "ACCEPTED_AWAITING_EWC", "ABANDONING"].includes(state),
    abandon: active() && ["SELECTED", "APPROVED", "RUNNING", "STAGED", "SUBMITTED", "ABANDONING"].includes(state),
  })[action] === true;
  const reasons = {select: "Select an eligible chunk after resolving other client work.",
    approve: "Select this chunk before approval.", run: approvedExpired ? "Approval expired. Abandon and select a new run."
      : state !== "APPROVED" ? "Approve this chunk before Run."
        : "Verify worker availability and resource reserves before Run.",
    submit: "Wait for a verified staged model update.", complete: "An accepted FedProx receipt is required.",
    recover: "Recovery is available for interrupted or unreconciled work.",
    abandon: "Only unaccepted runs can be abandoned."};
  const control = (label, action, css = "") => button(label, action, {css, disabled: !enabled(action),
    extra: !enabled(action) ? `title="${esc(operating || ui.busy ? "An operation is in progress." : reasons[action])}"` : ""});
  const grid = chunks.length ? `<div class="chunk-grid">${chunks.map(item => `<button type="button"
    class="chunk ${item.state === "COMPLETED" ? "completed" : item.state === "FAILED" ? "failed" : item.state !== "PENDING" ? "active" : ""}"
    aria-pressed="${item.chunk_id === chunk?.chunk_id}" data-chunk="${esc(item.chunk_id)}">
    <strong>${esc(item.chunk_id)}</strong><small>${esc(item.state.replaceAll("_", " "))}</small></button>`).join("")}</div>` :
    empty("Start a demo on Federated Server to initialize owned chunks.");
  const analytics = data.analytics;
  const clientStats = analytics.clients[role];
  const history = analytics.runs.filter(r => r.client_id === role);
  const order = new Map(history.filter(r => r.training_order != null).map((r, index) => [r.execution_id, index + 1]));
  const filtered = history.filter(r => inRange(r.started_at ?? r.state_at));
  const completed = history.filter(r => r.state === "COMPLETED" && inRange(r.completed_at))
    .sort((a,b) => Date.parse(a.completed_at) - Date.parse(b.completed_at));
  const receipts = analytics.updates.filter(r => r.client_id === role && inRange(r.accepted_at));
  const revisions = analytics.revisions.filter(r => r.client_id === role && inRange(r.created_at));
  const chunkStats = clientStats?.chunk_states;
  const readiness = data.continual_readiness?.[role];
  const health = ui.monitor?.nodes?.[`client-${role.toLowerCase()}`];
  const workerFresh = online && Date.now() - Date.parse(worker.last_verified_at) < 300000;
  const connection = ui.checkingWorkers.has(role) ? "RECONNECTING" : workerFresh || health?.state === "ONLINE" || health?.state === "PARTIAL" ?
    "CONNECTED" : health?.state === "UNAVAILABLE" || worker?.state === "UNAVAILABLE" ? "DISCONNECTED" : "UNKNOWN";
  const phase = run?.operation_in_progress && state === "RUNNING" ? "Dispatch in progress" : state === "RUNNING" ?
    "Recovery check required" : state?.replaceAll("_", " ") || "NOT STARTED";
  const isMoving = Boolean(run?.operation_in_progress && state === "SUBMITTED");
  const events = data.audit.filter(item => item.client_id === role && (!selected() || item.session_id === selected().session_id));
  const points = events.filter(item => run && item.execution_id === run.execution_id && item.event.startsWith("SYNTHETIC_STEP:"))
    .map(item => {const [,step,total,samples,workload,loss] = item.event.split(":");
      return {step:+step,total:+total,samples:+samples,workload:+workload,value:+loss,at:item.time};})
    .filter(item => Number.isFinite(item.value)).sort((a,b) => a.step-b.step);
  const latest = points.at(-1);
  const real = data.continual_jobs?.find(job => job.client === role);
  const evaluation = real?.evaluation_events ?? [];
  const retention = real?.old_task_evaluation;
  const recorded = real?.events.at(-1);
  const realStep = real?.events.filter(event => event.state === "STEP").at(-1);
  const realPanel = panel("Real continual training", "Authenticated worker events · restricted metrics stay client-local", real ?
    (realStep ? `<progress value="${realStep.completed}" max="${realStep.total}" aria-label="Real logical training steps"></progress>` : "") +
    list([["Job",esc(real.job)], ["Chunk",esc(real.chunk)], ["Last recorded event",esc(recorded?.state ?? "Awaiting worker events")],
      ["Event time",esc(recorded ? time(new Date(recorded.at * 1000).toISOString()) : "—")],
      ["Logical steps",esc(realStep ? `${realStep.completed}/${realStep.total}` : "—")],
      ["Staged hash",hash(recorded?.sha256)], ["FedProx version",version(real.receipt?.version)]]) :
    empty(data.continual_jobs === null ? "Real job evidence unavailable." : "No authorized real job recorded."));
  const evaluationPanel = evaluation.length ? panel("Client-local model comparison", "Execution evidence · v2 remains unvalidated", list([
    ["Global v0", hash(evaluation[0].v0)], ["Research v2", hash(evaluation[0].v2)], ["Metrics", "Client-local · RESTRICTED"],
  ]) + table(["Time", "Evaluation event"], evaluation.map(item => [esc(time(item.at)), esc(item.event)]))) : "";
  const retentionPanel = retention ? panel("Old-task retention comparison",
    "Development-exposed validation · private metrics stay client-local", list([
      ["Dataset", esc(retention.dataset)], ["Selection SHA-256", hash(retention.selection_sha256)],
      ["Verified sample count", esc(retention.rows ?? "Pending")],
      ["Global v0", hash(retention.v0)], ["Research v2", hash(retention.v2)],
      ["Receipt verification", retention.receipt_verified ? "VERIFIED" : "UNCONFIRMED"],
      ["Receipt SHA-256", hash(retention.receipt_sha256)],
    ]) + table(["Time", "Evaluation event"], retention.events.map(item => [esc(time(item.at)), esc(item.event)]))) : "";
  return realPanel + evaluationPanel + retentionPanel + (real?.export_events?.length ? realResearch([real]) : "") + `<div class="kpis training-kpis">${[
    ["Connection status", connection, "Verified probes"],
    ["Current chunk", chunk?.chunk_id || "—", readiness?.chunks ? `${readiness.chunks.length} frozen manifest chunks` : "Manifest unavailable"],
    ["Training state", phase, selected() ? modeName(selected().execution_mode) : "No demo session"],
    ["Samples processed", latest ? `${latest.samples}/${latest.workload}` : "—", latest ? "Synthetic sample visits" : "No released sample counter"],
    ["Current epoch", "—", "No recorded epoch counter"],
    ["Local loss", latest ? fmt(latest.value, 4) : "—", latest ? "Measured synthetic objective" : "No released loss measurements"],
    ["EWC status", run?.used_ewc ? "Prior revision loaded" : "—", "Synthetic run evidence"],
  ].map(args => stat(...args)).join("")}</div>
  <div class="note info">${remote() ? `Client ${role} Ubuntu worker · 192.168.252.${role === "A" ? "10" : "11"}
    ${badge(worker?.state ?? "CHECK PENDING", online ? "green" : "amber")}
    Last verified ${esc(time(worker?.last_verified_at))}.
    ${button("Check worker", "worker", {disabled: ui.checkingWorkers.has(role)})}` :
    "Local synthetic training runs in the Windows process."}</div>
    <div class="training-panels">
    ${panel("Federated training architecture", "Selected synthetic session", `<div class="training-link"><strong>Client ${role}</strong><span class="${isMoving ? "processing" : ""}">${isMoving ? "Submitting update" : "↔"}</span><strong>Windows FL Server</strong></div><details><summary>Combined federation</summary><div class="training-link"><strong>Client A</strong><span>↔</span><strong>Windows</strong><span>↔</span><strong>Client B</strong></div></details>`)}
    ${panel("Local training progress", "Durable run state", (latest ? `<progress value="${latest.step}" max="${latest.total}" aria-label="Synthetic training steps"></progress>` : "") + list([["Phase",esc(phase)], ["Base version",version(run?.base_version)],
      ["Completed logical steps",esc(latest ? `${latest.step}/${latest.total}` : run?.worker_steps ?? "NOT MEASURED")], ["Elapsed",seconds(run?.elapsed_ms == null ? null : run.elapsed_ms / 1000)],
      ["Last transition",esc(time(run?.state_at))], ["ETA", "NOT MEASURED"]]))}
    ${panel("Local training loss", latest ? "Measured synthetic objective, including EWC when applied" : "Measured points only", latest ? chart(`training-loss-${role}`, "Loss by step", points.map(item => ({...item,name:`Step ${item.step}`}))) : empty("No approved loss series for this run."))}
    ${panel("Elastic Weight Consolidation", "Client-local reference evidence", list([["Revision",esc(revisions.at(-1)?.revision)],
      ["Reference SHA-256",hash(revisions.at(-1)?.state_sha256)], ["Penalty and lambda", "NOT MEASURED"],
      ["Real Fisher review", "PENDING"], ["Retention evaluation", "RESTRICTED"]]))}
    ${panel("Dataset and chunk status", "Frozen manifest · earlier hash-verified deployment", list([["Assignment",esc(readiness?.dataset)],
      ["Dataset",esc(readiness?.filename)], ["Verified manifest chunks",esc(readiness?.chunks?.length)],
      ["Manifest SHA-256",hash(readiness?.manifest_sha256)], ["Current VM file integrity", "Requires fresh verification"]]))}
    ${panel("Update packaging and upload", "Durable checkpoint and coordinator receipt", list([["Update",hash(run?.update_sha256)],
      ["Transfer progress", "NOT MEASURED"], ["Base version",version(run?.base_version)],
      ["FedProx acceptance",receipts.length ? version(receipts.at(-1).accepted_version) : "No verified receipt"]]))}
    ${panel("Local evaluation", "Client-local results", empty("RESTRICTED — separate disclosure approval required."))}
    ${panel("System health", "Cached infrastructure probe", list([["Connection",esc(connection)], ["CPU",percent(health?.metrics?.cpu_percent)],
      ["RAM available",gib(health?.metrics?.ram_available_bytes)], ["Disk free",gib(health?.metrics?.disk_available_bytes)],
      ["Last probe",esc(time(health?.checked_at ?? worker?.last_verified_at))]]))}
    </div>
    ${panel("Activity log", "Coordinator audit · IST", table(["Timestamp", "Client", "Event", "Job", "Outcome"], events.slice(0,15).map(item =>
      [esc(time(item.time)),esc(item.client_id),esc(item.event),esc(item.execution_id),esc(item.outcome)])))}
    <div class="kpis">${["COMPLETED","PENDING","RUNNING","FAILED"].map(key =>
      stat(key[0] + key.slice(1).toLowerCase(), chunkStats ? (chunkStats[key] ?? 0) : "—", "Whole selected session")).join("")}
    </div><div class="split">
    ${panel(`Owned chunks · Client ${role}`, source(), grid)}
    ${panel("Manual execution", "Select → Approve → Run → Submit → Reconcile", chunk ? `
      <div class="run-detail"><strong>${esc(chunk.chunk_id)} · ${badge(state, tone(state))}</strong>
      ${list([["Session", esc(selected()?.session_id)], ["Execution", esc(run?.execution_id)], ["Training base", version(run?.base_version)],
        ["Base SHA-256", hash(run?.base_checkpoint_sha256)], ["Update SHA-256", hash(run?.update_sha256)],
        ["Training steps", esc(run?.worker_steps ?? "NOT MEASURED")],
        ["VM training, Fisher and staging", seconds(run?.worker_training_seconds)],
        ["Run to staged", Number.isFinite(run?.elapsed_ms) ? `${fmt(run.elapsed_ms)} ms` : "NOT MEASURED"],
        ["EWC at Run", run?.used_ewc == null ? "Unavailable" : run.used_ewc ? "Prior revision loaded" : "No prior revision"],
        ["EWC promotion", state === "COMPLETED" ? "Reference published" :
          ["STAGED", "SUBMITTED", "ACCEPTED_AWAITING_EWC"].includes(state) ? "Reconciliation pending" : "—"],
        ["Completed", esc(time(run?.completed_at))],
        ["Recovery", esc(operating ? "Operation in progress" : run?.failure_reason ??
          (["RUNNING","SUBMITTED","ACCEPTED_AWAITING_EWC","ABANDONING"].includes(state) ? "Inspect or recover" : "—"))]])}</div>
      <div class="actions">${control("Select", "select")}${control("Approve", "approve")}${control("Run", "run", "primary")}
        ${control("Submit", "submit")}${control("Reconcile EWC", "complete")}${control("Recover", "recover")}
        ${control("Abandon", "abandon", "danger")}</div>` : empty("Choose a chunk to inspect its actions."))}
    </div><div class="filters">${rangeControl()}<span class="source">Updated ${esc(time(analytics.updated_at))}</span></div>
    <div class="split">
      ${panel("Recorded duration per completed chunk", remote() ? "Worker training, Fisher and staging" : "Elapsed coordinator Run → staged",
        chart("duration", "seconds", completed.map(r => ({name:r.chunk_id,at:r.completed_at,
          value:remote() ? r.worker_training_seconds : r.run_to_staged_seconds})), "bar"))}
      ${panel("EWC revisions", "Coordinator opaque references · SQLite",
        chart("ewc", "revision", revisions.map(r => ({name:`r${r.revision}`,value:r.revision,at:r.created_at}))) +
        table(["Revision", "Reference SHA-256", "Published"], revisions.map(r => [esc(r.revision), hash(r.state_sha256), esc(time(r.created_at))])))}
    </div><div class="split">${panel("Chunk completion history", "Completions in the selected time range",
      chart("chunk-completions", "completed chunks", completed.map((r,i) => ({name:r.chunk_id,value:i+1,at:r.completed_at}))))}
    ${panel("Contribution to global versions", "Verified receipts for this client",
      chart("client-versions", "accepted version", receipts.map(r => ({name:r.chunk_id,value:r.accepted_version,at:r.accepted_at}))))}</div>
    ${panel("Chunk execution history", "Order from durable RUNNING events · timestamps in IST",
      table(["Training order","Chunk","State","Base","Accepted","Steps","Submitted at","Submit → accept","Accept → complete","Completed at"],
      filtered.map(r => [esc(order.get(r.execution_id)),esc(r.chunk_id),badge(r.state,tone(r.state)),version(r.base_version),
        version(r.verified_accepted_version),esc(r.worker_steps ?? "NOT MEASURED"),esc(time(r.submitted_at)),
        seconds(r.submission_seconds),seconds(r.reconciliation_seconds),esc(time(r.completed_at))])))}
    ${panel("Performance summary", "Whole selected session · elapsed times can include waiting and recovery",
      performanceTable(clientStats?.performance ?? {}, analytics.phases))}
    ${panel("Latest verified receipt", source(), updateTable(receipts.slice(-1)))}
    ${panel("Operational alerts", `Client ${role} · selected session`, alertsTable(analytics.alerts.filter(row => row.client === role)))}
    `;
}

function registry() {
  const data = ui.data, b = data.baseline;
  return `${panel("Research global v0", "Pinned write-once registry reference", list([
    ["Checkpoint SHA-256", hash(b.sha256)], ["Contract", esc(b.contract_id)], ["Fingerprint", hash(b.contract_fingerprint)],
    ["Architecture", esc(b.architecture.join(" → "))], ["Integrity", badge(b.verified ? "Verified" : "Unavailable or mismatch", b.verified ? "green" : "amber")],
  ]))}<div class="split">${panel("Model lineage", source(), timeline(data.models))}
  ${panel("Version chart", source(), versionChart(data.models))}</div>
  ${panel("Session checkpoint history", "Hashes checked against persisted files", table(
    ["Version","Parent checkpoint","Training base","Client","Chunk","Staleness","Alpha","SHA-256","Integrity"],
    data.models.map(m => [version(m.version),m.parent_version == null ? "—" :
      `${version(m.parent_version)} · ${hash(data.models.find(p => p.version === m.parent_version)?.checkpoint_sha256)}`,
      version(m.base_version),esc(m.client_id),esc(m.chunk_id),
      esc(m.staleness),m.aggregation_weight == null ? "—" : fmt(m.aggregation_weight),hash(m.checkpoint_sha256),
      badge(m.integrity,m.integrity === "Verified" ? "green" : "amber")])) )}
  ${panel("Recorded Task 9E checkpoints", "Historical synthetic verification · disposable successor files", table(
    ["Version","Chunk","Base","Staleness","Alpha","Recorded SHA-256"], data.historical.runs.map(r => [
      version(r.accepted_version),esc(r.chunk_id),version(r.base_version),esc(r.staleness),fmt(r.aggregation_weight),hash(r.checkpoint_sha256),
    ]), "Historical integration artifact unavailable."))}
  ${panel("Earlier synthetic verification", "Task 9D · independent disposable run",
    data.historical.task9d_verified_updates == null ? empty("Task 9D evidence unavailable.") : list([
      ["Verified updates", esc(data.historical.task9d_verified_updates)],
      ["Artifact modification", esc(time(data.historical.task9d_artifact_modified_at))],
    ]))}
  ${panel("Recorded Task 10B remote run", "Historical synthetic evidence · separate from this dashboard session",
    table(["Chunk","Version","Staleness","Alpha","Recorded acceptance"], data.historical.task10b_runs.map(r =>
      [esc(r.chunk_id),version(r.version),esc(r.staleness),fmt(r.alpha),esc(time(r.at))])))}`;
}

function privacy() {
  return `<div class="split">${panel("Research privacy boundary", "Pinned v0 release policy", list([
    ["Starting checkpoint mode", badge("research_nonsecure","amber")], ["Dashboard execution", badge(modeName(selected()?.execution_mode),"cyan")],
    ["Secure RNG", esc("Unverified for production")], ["Real training", badge("Disabled","amber")],
    ["DP randomness", esc("Non-cryptographic research randomness")],
    ["Evaluation scope", esc("Research-only model · no independent holdout · incomplete attack-family validation")],
    ["Privacy ledger", esc("Restricted · client-local")], ["Organization inference", esc("Disabled")],
    ["Production and public release", esc("Disabled")],
  ]))}${panel("Dashboard security", "Current server configuration", list([
    ["Listener", esc("127.0.0.1 only")], ["Operator access", esc("Authenticated · active session cookie")],
    ["State changes", esc("Same-origin and CSRF checked")], ["Client controls", esc("Server-side A/B ownership checks")],
  ]))}</div>${panel("Client evaluation", "Release boundary · quantitative results remain client-local",
    table(["Evaluation measure", "Disclosure"], ["Accuracy", "Precision", "Recall", "F1-score", "False-positive rate",
      "Confusion matrix", "Attack-family detection"].map(label => [esc(label),badge("RESTRICTED","amber")])))}`;
}

function health() {
  const monitor = ui.monitor;
  const names = {"Windows":"Windows FL Server","client-a":"Client A","client-b":"Client B",organization:"Organization"};
  const rows = Object.entries(monitor?.nodes ?? {});
  const usage = m => [percent(m?.cpu_percent), gib(m?.ram_total_bytes), gib(m?.ram_used_bytes),
    gib(m?.ram_available_bytes), gib(m?.disk_available_bytes), percent(m?.disk_used_percent)].map(esc);
  const known = rows.filter(([,row]) => row.last_known && !row.metrics);
  const checkText = ui.monitoring || monitor?.refreshing ? "Probe in progress" :
    monitor?.cooldown_seconds ? `Last probe cached · refresh limit ${monitor.cooldown_seconds} s` : "Ready for manual probe";
  return `${panel("Infrastructure observations", "CPU sampled over ~200 ms · Windows project volume / Ubuntu home filesystem",
    (ui.monitorError ? '<p class="note">Monitoring endpoint unavailable.</p>' : "") +
    `<p class="muted" role="status">${esc(checkText)} · ${monitor?.ssh_enabled ?
      "Strict SSH probes enabled" : "Remote probes require server opt-in"}</p>` +
    `<p class="source">UNKNOWN means this dashboard has no usable health sample. STALE means the last successful sample aged out. Neither status confirms a disconnection.</p>` +
    table(["Node","Current status","Attempt","Last success","CPU","RAM total","RAM used","RAM available",
      "Disk free","Disk used","Probe duration","Error"], rows.map(([role,row]) => [
      `<strong>${esc(names[role])}</strong><span class="source">${esc(row.address)} · ${esc(row.source ?? "No probe yet")}</span>`,
      badge(row.state,row.state === "ONLINE" ? "green" : "amber"), esc(time(row.checked_at)),
      esc(time(row.last_known?.checked_at)), ...usage(row.metrics), seconds(row.duration_seconds),
      esc(row.error?.replaceAll("_"," ")),
    ]), ui.monitoring ? "Loading infrastructure status…" : "No infrastructure sample loaded."),
    button(monitor?.ssh_enabled ? "Refresh probes" : "Refresh Windows health", "monitor-refresh", {disabled:ui.monitoring, css:"primary"}))}
    ${known.length ? panel("Last-known values", "Historical samples from this dashboard process · stale or unavailable now",
      table(["Node","Sample time","CPU","RAM total","RAM used","RAM available","Disk free","Disk used"], known.map(([role,row]) =>
        [esc(names[role]),esc(time(row.last_known.checked_at)),...usage(row.last_known.metrics)]))) : ""}
    ${panel("Infrastructure alerts", "Measured thresholds and probe failures", alertsTable(monitor?.alerts ?? []))}
    ${panel("Session alerts", "Selected session · SQLite and checkpoint verification", alertsTable(ui.data.analytics.alerts))}`;
}

function auditTable(rows) {
  return table(["Timestamp","Event","Outcome","Session","Client","Execution"], rows.map(row => [
    esc(time(row.time)),esc(row.event),esc(row.outcome),hash(row.session_id),esc(row.client_id),hash(row.execution_id),
  ]), "No matching audit records.");
}

function auditResults() {
  const query = ui.query.toLowerCase();
  const rows = ui.data.audit.filter(row => (!ui.filter || (ui.filter === "rejected" ? row.outcome.startsWith("Rejected") :
    ui.filter === "recovery" ? ["recover","abandon","ABANDONING","FAILED"].includes(row.event.replace(/^REMOTE /, ""))
      : row.client_id === ui.filter)) &&
    Object.values(row).some(value => String(value ?? "").toLowerCase().includes(query)));
  return auditTable(rows);
}

function auditPage() {
  return panel("Audit records", "SQLite session and run events · recent dashboard action log", `
    <div class="filters"><input class="field" id="audit-query" type="search" aria-label="Search audit history"
      placeholder="Search action, session or execution…" value="${esc(ui.query)}">
    <select id="audit-filter" aria-label="Filter audit history">
      ${[["","All events"],["A","Client A"],["B","Client B"],["rejected","Rejected actions"],["recovery","Recovery"]]
        .map(([key,label]) => `<option value="${key}" ${ui.filter === key ? "selected" : ""}>${label}</option>`).join("")}</select></div>
    <div id="audit-results">${auditResults()}</div>`);
}

function render() {
  if (!ui.data) return;
  ui.chart = {};
  if (researchClient()) {
    clientDashboard.mount(researchClient());
    return;
  }
  if (ui.page === "server") {
    flServer.mount();
    return;
  }
  const views = {overview, "server-tools": serverPage, "client-a-tools": () => clientPage("A"), "client-b-tools": () => clientPage("B"),
    models: registry, privacy, health, audit: auditPage};
  const [pageTitle,,pageDescription] = pages[ui.page];
  const clientView = Boolean(syntheticClient());
  const title = clientView ? `${pageTitle} Training Dashboard` : pageTitle;
  const description = clientView ? "Federated Continual Learning Monitor" : pageDescription;
  document.title = `${title} | Federated IDS`;
  const content = views[ui.page]();
  app.innerHTML = `<div class="shell"><aside class="sidebar">${brand()}<nav aria-label="Dashboard pages">
    ${Object.entries(pages).filter(([key]) => !key.endsWith("-tools")).map(([key,[name,icon]],index) => `
      ${index === 0 ? '<div class="nav-label">Workspace</div>' : index === 5 ? '<div class="nav-label">Governance</div>' : ""}
      <a class="nav-link" href="/${key}" data-page="${key}" ${key === ui.page ? 'aria-current="page"' : ""}>
        <span class="nav-icon" aria-hidden="true">${icon}</span>${name}</a>`).join("")}</nav>
    <div class="sidebar-foot">${badge(modeName(selected()?.execution_mode),"cyan")}<div>Windows coordinator</div></div></aside>
    <div><header class="topbar"><div class="breadcrumb">Research console / <strong>${esc(title)}</strong></div>
      <div class="top-actions">${ui.data.sessions.length ? `<select id="session-picker" aria-label="Inspect session">
        <option value="">Auto · active or latest</option>${ui.data.sessions.map(s => `<option value="${esc(s.session_id)}"
          ${s.session_id === ui.session ? "selected" : ""}>${esc(short(s.session_id))} · ${esc(s.status)} · ${esc(modeName(s.execution_mode))}</option>`).join("")}</select>` : ""}
        ${button("Refresh","refresh",{css:"icon"})}${ui.localOnly ? "" : button("Sign out","logout",{css:"icon"})}
      </div></header>
      <main id="content" class="content"><div class="page-head"><div><span class="eyebrow">Federated continual learning</span>
      <h1>${esc(title)}</h1><p>${esc(description)}</p></div>
      <div class="actions">${selected() ? badge(modeName(selected().execution_mode), "cyan") +
        badge(selected().status,tone(selected().status)) : badge("No demo session")}</div></div>${content}</main>
    </div></div>`;
  document.querySelectorAll("canvas[data-chart]").forEach(canvas => draw(canvas));
}

function runId(key) {
  let value = sessionStorage.getItem(key);
  if (!value) { value = crypto.randomUUID(); sessionStorage.setItem(key, value); }
  return value;
}

app.addEventListener("submit", async event => {
  if (event.target.id !== "login") return;
  event.preventDefault();
  try {
    const result = await api("/api/login", {access_key: document.querySelector("#key").value});
    ui.csrf = result.csrf;
    await refresh();
    checkVisibleWorkers();
  } catch (error) { login(error.message); }
});

app.addEventListener("click", async event => {
  const nav = event.target.closest("[data-page]");
  if (nav && ui.data && !event.metaKey && !event.ctrlKey) {
    const next = nav.dataset.page;
    const nextIsResearchClient = next === "client-a" || next === "client-b";
    if (Boolean(researchClient()) !== nextIsResearchClient) return;
    event.preventDefault(); ui.page = nav.dataset.page; history.pushState({}, "", nav.href);
    render();
    checkVisibleWorkers(); return;
  }
  if (ui.page === "server" && flServer.click(event)) return;
  if (researchClient() && clientDashboard.click(event)) return;
  const chunk = event.target.closest("[data-chunk]");
  if (chunk) {
    const role = syntheticClient();
    ui.chunk[role] = chunk.dataset.chunk;
    sessionStorage.setItem(`chunk-${selected().session_id}-${role}`, chunk.dataset.chunk);
    render(); return;
  }
  const target = event.target.closest("[data-action]");
  if (!target || target.disabled || ui.busy || !ui.data) return;
  const action = target.dataset.action;
  if (action === "refresh") {
    await refresh();
    checkVisibleWorkers();
    return;
  }
  if (action === "monitor-refresh") { await loadMonitor(true); return; }
  const role = syntheticClient();
  if (action === "worker" && role) { await checkWorker(role); return; }
  const sessionId = selected()?.session_id;
  const chunkId = ui.chunk[role] || sessionStorage.getItem(`chunk-${sessionId}-${role}`);
  const record = ui.data.chunks?.find(c => c.client_id === role && c.chunk_id === chunkId);
  let body = {}, requestKey;
  if (action === "start") {
    requestKey = `start-demo-${ui.mode}`; body.request_id = runId(requestKey); body.execution_mode = ui.mode;
  }
  else if (action === "close") { if (!sessionId || !confirm("Close this synthetic demo?")) return; body.session_id = sessionId; }
  else if (action !== "logout") {
    if (!role || !record || !sessionId) return;
    body = {client_id: role, session_id: sessionId, execution_id: record.execution_id};
    if (action === "select") {
      requestKey = `select-${sessionId}-${role}-${chunkId}`;
      body.execution_id = runId(requestKey); body.chunk_id = chunkId;
    }
    if (action === "abandon" && !confirm("Abandon this synthetic execution?")) return;
    if (action === "run" && remote()) {
      if (!confirm(`Run synthetic chunk ${chunkId} on Client ${role}'s Ubuntu VM?`)) return;
      body.confirm_remote = true;
    }
  }
  ui.busy = true;
  render();
  try {
    const result = await api(action === "logout" ? "/api/logout" : `/api/actions/${action}`, body);
    if (requestKey) sessionStorage.removeItem(requestKey);
    if (action === "logout") { ui.data = null; ui.session = ""; login(); return; }
    if (action === "start") { ui.session = result.result.session_id; ui.chunk = {}; }
    ui.busy = false;
    await refresh();
    notice(`Recorded ${action} · ${result.result.state ?? result.result.status ?? version(result.result.accepted_version)}` +
      (result.audit_recorded === false ? " · Supplemental audit log unavailable" : ""));
  } catch (error) {
    ui.busy = false;
    await refresh();
    if (ui.data) notice(error.message, true);
  } finally {
    ui.busy = false;
    checkVisibleWorkers();
  }
});

app.addEventListener("change", async event => {
  const {id, value} = event.target;
  if (event.target.matches("[data-dashboard-picker]") && Object.hasOwn(pages, value) && value !== ui.page) {
    location.assign("/" + value);
    return;
  }
  if (id === "session-picker") { ui.session = value; ui.chunk = {}; await refresh(); checkVisibleWorkers(); }
  if (id === "execution-mode") ui.mode = value;
  if (id === "server-metric") { ui.serverMetric = value; render(); }
  if (id === "analytics-range") { ui.analyticsHours = value; render(); }
  if (id === "analytics-client") { ui.analyticsClient = value; render(); }
  if (id === "audit-filter") { ui.filter = value; document.querySelector("#audit-results").innerHTML = auditResults(); }
});
app.addEventListener("input", event => {
  if (event.target.id !== "audit-query") return;
  ui.query = event.target.value; document.querySelector("#audit-results").innerHTML = auditResults();
});
app.addEventListener("pointermove", event => {
  const canvas = event.target.closest("canvas[data-chart]");
  if (!canvas) return;
  const items = ui.chart[canvas.dataset.chart].items;
  const width = Math.max(canvas.clientWidth, 140);
  const index = Math.min(items.length - 1, Math.max(0, Math.floor((event.offsetX - 42) / (width - 56) * items.length)));
  if (+canvas.dataset.index !== index) draw(canvas, index);
});
app.addEventListener("keydown", event => {
  const canvas = event.target.closest("canvas[data-chart]");
  if (!canvas || !["ArrowLeft", "ArrowRight"].includes(event.key)) return;
  event.preventDefault();
  const size = ui.chart[canvas.dataset.chart].items.length;
  draw(canvas, Math.min(size - 1, Math.max(0, +canvas.dataset.index + (event.key === "ArrowRight" ? 1 : -1))));
});
window.addEventListener("resize", () => document.querySelectorAll("canvas[data-chart]").forEach(c => draw(c, +c.dataset.index)));
window.addEventListener("popstate", async () => {
  ui.page = pageAtLocation();
  if (ui.data?.clientOnly && !researchClient()) await refresh();
  else render();
  checkVisibleWorkers();
});
void refresh().then(checkVisibleWorkers);
setInterval(() => {
  if (ui.data && !document.hidden && !researchClient() && (ui.page === "server-tools" || syntheticClient() || active() && remote())) void refresh(true);
}, 2500);
setInterval(() => {
  if (ui.data && !document.hidden && researchClient()) void clientDashboard.poll();
}, 1700);
setInterval(() => {
  if (ui.page === "server" && ui.data && !document.hidden) void flServer.loadHealth();
  if (ui.data && !document.hidden && researchClient()) void clientDashboard.pollHealth();
}, 7000);
setInterval(() => {
  if (ui.data && !document.hidden && (ui.page === "health" || syntheticClient())) void loadMonitor();
}, 15000);
setInterval(() => {
  if (ui.data && !document.hidden && remote() && syntheticClient()) {
    void checkWorker(syntheticClient());
  }
}, 60000);
