const stageIcons = ["⌁", "▤", "▱", "✓", "⚙", "▣", "↑", "⟳", "↓"];
const count = value => Number.isFinite(value) ? new Intl.NumberFormat("en-IN").format(value) : "—";
const percent = value => Number.isFinite(value) ? Math.round(value) + "%" : "—";
const humanState = state => state === "VERIFIED LOCAL COPY" ? state : state.replaceAll("_", " ");
const level = state => ({
  COMPLETED: "green", ONLINE: "green", "VERIFIED LOCAL COPY": "green",
  RUNNING: "blue", FAILED: "red", OFFLINE: "red", STALE: "amber",
  "AWAITING APPROVAL": "amber", INTERRUPTED: "amber",
}[state] || "neutral");
const healthErrors = {
  SSH_CONNECTION_TIMEOUT: "SSH connection timed out",
  SSH_HOST_KEY_MISMATCH: "Pinned SSH host key mismatch",
  SSH_AUTHENTICATION_FAILED: "SSH authentication failed",
  SSH_CONNECTION_OR_HOST_TRUST_FAILED: "Connection or host verification failed",
  PROBE_TIMEOUT: "System probe timed out",
  PROBE_CONFIGURATION_UNAVAILABLE: "Pinned SSH configuration unavailable",
  PROBE_OR_ROUTE_FAILED: "Probe or pinned SSH route unavailable",
  MEASUREMENTS_INCOMPLETE: "Some measurements unavailable",
};
const past = state => ["COMPLETED", "VERIFIED LOCAL COPY"].includes(state);

export function createClientDashboard({app, api, esc, time, gib, table, list, dashboardPicker, scope, expired, csrf, localOnly}) {
  let client, state, health, selected;
  let request, healthRequest, epoch = 0;
  let stale = false, healthStale = false, busy = false, message = "";
  let logIds = "", inspected = null;
  const find = selector => app.querySelector(selector);
  const route = () => "/api/client-" + client.toLowerCase();
  const timestamp = value => value ? time(value) : "—";
  const model = version => Number.isInteger(version) ? (version === 0 ? "Global v0" : "Research v" + version) : "—";
  const badge = (name, style = "neutral") => '<span class="cd-state ' + style + '">' + esc(name) + '</span>';
  const button = (label, name, value = "") => '<button class="cd-button" type="button" data-cd-' + name +
    '="' + esc(value) + '">' + esc(label) + '</button>';
  const header = (symbol, title) => '<h2><span aria-hidden="true">' + symbol + '</span>' + esc(title) + '</h2>';
  const value = (label, body) => [label, esc(body ?? "—")];
  const selectedRow = () => state?.chunks.find(row => row.id === selected);
  const focusedJob = () => state?.jobs.find(job => job.id === (selectedRow()?.job || state.active_job));
  const set = (selector, html) => {
    const node = find(selector);
    if (node && node.dataset.last !== html) {
      const open = new Set([...node.querySelectorAll("details[open]")].map(detail => detail.querySelector("summary").textContent));
      const scroll = node.querySelector(".table-wrap")?.scrollTop;
      node.innerHTML = html;
      node.dataset.last = html;
      node.querySelectorAll("details").forEach(detail => { detail.open = open.has(detail.querySelector("summary").textContent); });
      if (scroll) node.querySelector(".table-wrap").scrollTop = scroll;
    }
  };

  function paintHealth() {
    const i = health?.identity || {};
    const m = healthStale ? null : health?.metrics;
    const status = healthStale ? "STALE" : health?.status || "UNKNOWN";
    const ip = i.active_ip || health?.configured_ip;
    const ram = m?.ram_total_bytes ? 100 * m.ram_used_bytes / m.ram_total_bytes : null;
    const summaries = [
      ["●", "Client status", badge(status, level(status))],
      ["⌘", "IP address", esc(ip) + (status !== "ONLINE" || !i.active_ip ? '<small>' + (i.active_ip ? "Last verified" : "Configured") + '</small>' :
        i.active_ip !== health.configured_ip ? '<small>Configured ' + esc(health.configured_ip) + '</small>' : "")],
      ["◉", "Operating system", esc(i.os || "Unavailable") + (i.hostname ? '<small>' + esc(i.hostname) + '</small>' : "")],
      ["▱", "Role", state ? "Training (Dataset " + esc(state.dataset) + ")" : "Training client"],
      ["▣", "Resources", "CPU " + percent(m?.cpu_percent) + " · RAM " + percent(ram) + " · Disk " + percent(m?.disk_used_percent)],
    ];
    set("#cd-summary", summaries.map(([symbol, label, body]) => '<div class="cd-info-card"><span aria-hidden="true">' +
      symbol + '</span><div><small>' + label + '</small><strong>' + body + '</strong></div></div>').join("") +
      '<div class="cd-health-more">' + button("Machine details", "machine") + '</div>');
    find("#cd-health-time").textContent = "Last probe " + timestamp(health?.checked_at);
  }

  function paintChunks() {
    const caption = state ? "Dataset " + state.dataset + " – Chunks (" +
      state.chunks[0]?.id + "–" + state.chunks.at(-1)?.id + ")" : "Dataset chunks";
    const rows = (state?.chunks || []).map(row => {
      const chosen = row.id === selected;
      const name = '<button type="button" class="cd-select" data-cd-select="' + esc(row.id) + '" aria-pressed="' +
        chosen + '">' + esc(row.id) + '</button>';
      return [name, count(row.rows), count(row.selection_size), esc(row.verification),
        client === "B" ? badge("Completed (preview)", "green") : badge(row.status, level(row.status === "Completed" ? "COMPLETED" : row.status.toUpperCase())),
        button(row.job ? "View" : "Select", "select", row.id)];
    });
    set("#cd-chunks", header("▱", caption) + (client === "B" ? '<p class="cd-note">Visual preview: all chunk indicators show completed. Actual training status and actions are unchanged.</p>' : "") + '<p class="cd-subline">' + esc(state?.filename) + '</p>' +
      table(["Chunk", "Rows", "Slots", "Verification", "Status", "Action"], rows, "Loading frozen manifest…"));
  }

  function paintWorkflow(job) {
    const stages = job?.stages || ["SSH connection", "Receive global model", "Verify dataset chunk", "Privacy preflight and reservation",
      "Local training", "Checkpoint and evaluation", "Send model update", "FL validation and FedProx",
      "Receive updated global model"].map(label => ({label, state: "PENDING"}));
    const active = !stale && job?.active ? stages.findIndex(stage => stage.state === "RUNNING") : -1;
    const prep = state?.preflight;
    const preflighting = !stale && !job?.active && prep?.chunk === selected && prep.status === "Checking";
    const processState = stale ? "RECONNECTING" : preflighting ? "Preflight checking" :
      prep?.chunk === selected && prep.status === "APPROVAL REQUIRED" ? "APPROVAL REQUIRED" :
      job ? (job.active ? "Current job · " : "Recorded job · ") + job.chunk + " · " + job.status :
      "No job for " + (selected || "this chunk");
    const finished = stages.filter(stage => past(stage.state)).length;
    set("#cd-workflow", '<div class="cd-panel-head">' + header("⌁", "Training Workflow – Federated Continual Learning") +
      badge(processState, stale ? "amber" : active !== -1 ? "blue" : "neutral") + '</div><div class="cd-stages-scroll"><ol class="cd-stages">' +
      stages.map((stage, index) => {
        const running = index === active;
        const status = stage.state === "RUNNING" && !running ? (stale ? "STALE" : "PENDING") : stage.state;
        const detail = running && stage.label === "Local training" && job.total
          ? '<progress max="' + job.total + '" value="' + (job.steps || 0) + '"></progress>' : "";
        const note = localOnly() && index === 8 && status === "AWAITING APPROVAL"
          ? "Awaiting research transfer · NO APPROVED NEW GLOBAL MODEL AVAILABLE" : stage.note;
        return '<li class="' + level(status) + '"><span class="cd-stage-icon" aria-hidden="true">' + stageIcons[index] +
          '</span><strong>' + (index + 1) + '. ' + esc(stage.label) + '</strong><span class="cd-stage-state">' +
          (running ? '<i class="cd-spinner" aria-hidden="true"></i>' : past(status) ? '✓ ' : "") + esc(humanState(status)) +
          '</span><small>' + esc(note || "") + '</small>' + detail +
          (stage.at ? '<time>' + esc(timestamp(stage.at)) + '</time>' : "") + '</li>';
      }).join("") + '</ol></div><div class="cd-overall"><strong>Overall progress</strong><progress aria-label="Completed workflow stages" max="' +
      stages.length + '" value="' + finished + '"></progress><span>' + finished + ' / ' + stages.length + ' recorded stages</span></div>');
  }

  function schedule(data) {
    if (!data) return [];
    const s = data.schedule;
    return [value("Selection", data.selection), value("Selected slots / steps", s.sample_count + " / " + s.steps),
      value("Input SHA-256", data.base_sha256), value("Feature contract", data.contract), value("Slot SHA-256", data.slot_sha256),
      value("Batch / microbatch", s.batch_size + " / " + s.microbatch_size), value("Epochs / learning rate", s.epochs + " / " + s.learning_rate),
      value("DP clip / noise", s.clip + " / " + s.noise), value("Delta / epsilon cap", s.delta + " / " + s.epsilon_limit),
      value("Attempt limit", data.attempt_limit)];
  }

  function jobDetails(job) {
    return [
      value("Client ID", job?.client || client), value("Job ID", job?.id), value("Chunk", job?.chunk || selected),
      value("Input global model", model(job?.approval?.base)), value("Selected slots", job?.approval?.schedule.sample_count),
      value("Training steps", job?.total ? (job.steps ?? 0) + " / " + job.total : null),
      value("DP-SGD", job?.approval ? "Enabled · research accounting" : null),
      value("EWC", job?.approval ? job.approval.ewc ? "Enabled" : "Disabled" : null),
      value("Privacy ledger", job?.privacy),
      value("Resources", health?.status === "ONLINE" && !healthStale ? "Fresh machine readings available" : "Current readings unavailable"),
      value("Checkpoint", job?.checkpoint ? "Staged · " + job.checkpoint.slice(0, 12) + "…" : "Unavailable"),
      value("Evaluation receipt", job?.evaluation), value("Update submission", job?.submission),
      value("FedProx", job?.receipt ? "Accepted · " + model(job.receipt.version) + " · Experimental / unvalidated" : "No verified acceptance"),
      value("Returned research checkpoint", job?.research_returns?.length ?
        "RESEARCH CHECKPOINT RECEIVED · " + model(job.research_returns.at(-1).version) : "Awaiting transfer"),
      value("Approved deployment model", localOnly() ? "NO APPROVED NEW GLOBAL MODEL AVAILABLE" :
        job?.returned ? model(job.returned.version) + " · verified in inventory" : "Awaiting approved global model"),
      value("Current action", job?.action), value("Last job event", timestamp(job?.last_event)),
    ];
  }

  function paintRun(job) {
    const row = selectedRow();
    const prep = state?.preflight?.chunk === selected ? state.preflight : null;
    const details = jobDetails(job);
    if (row) details.splice(3, 0, value("Logical range", row.start == null ? "Unavailable" :
      count(row.start) + "–" + count(row.stop - 1) + " · " + count(row.rows) + " rows"),
    value("Eligibility", row.reason));
    const controls = row?.eligible && !job && prep?.status !== "Checking" ? (prep?.status === "APPROVAL REQUIRED" ?
      button(localOnly() ? "Confirm & Start Training" : "Confirm approved dispatch", "approve", prep.job) : button("Run Training", "preflight", row.id)) : "";
    const grant = !localOnly() && prep?.status === "APPROVAL REQUIRED" ? JSON.stringify({
      operation: "local_continual_training", job: prep.job, scope: prep.scope,
      slot_sha256: prep.slot_sha256, expires: Math.floor(Date.parse(prep.expires_at) / 1000),
      ...(prep.scope?.protocol ? {privacy_projection: prep.privacy} : {}),
    }, null, 2) : "";
    const stateAction = job?.status.startsWith("Interrupted") ? "reconcile" :
      job?.receipt ? job.returned || localOnly() ? null : "return-model" :
      job?.checkpoint ? job.submission.startsWith("Received") ? "aggregate" : "export" : null;
    const federationGrant = !localOnly() && (stateAction === "export" || stateAction === "aggregate") ? JSON.stringify({
      operation: stateAction === "export" ? "submit_continual_update" : "aggregate_continual_update",
      client, job: job.id, sha256: job.checkpoint, base_sha256: job.approval?.base_sha256, alpha_policy: "inverse-staleness-0.5-v1",
      ...(stateAction === "aggregate" ? {head_version: state.head?.version, head_sha256: state.head?.sha256} : {}),
    }, null, 2) : "";
    const candidates = job && !job.active && localOnly() ? state.research_models.filter(candidate =>
      !job.research_returns?.some(item => item.version === candidate.version)) : [];
    const researchPicker = candidates.length ? '<details><summary>Transfer research-only checkpoint</summary>' +
      candidates.map(item => '<p class="cd-note">' + esc(model(item.version) + " · " + item.contract +
        (item.version ? " · experimental / unvalidated · " : " · research baseline · ") + item.sha256) +
        ' ' + button("Confirm research transfer", "research", item.version) + '</p>').join("") + '</details>' : "";
    const summary = details.filter(([key]) => !["Client ID", "Chunk", "Logical range", "Resources", "Job ID"].includes(key));
    set("#cd-run", header("▤", "Run Details" + (selected ? " – " + selected : "")) +
      (controls && prep?.status !== "APPROVAL REQUIRED" ? '<div class="cd-controls">' + controls + '</div>' : "") +
      (stateAction && !job.active ? '<div class="cd-controls">' + button({
        export: localOnly() ? "Confirm update export" : "Confirm approved export",
        aggregate: localOnly() ? "Confirm FedProx aggregation" : "Confirm approved aggregation", reconcile: "Reconcile worker evidence",
        "return-model": "Check approved return model",
      }[stateAction], "federate", stateAction) + '</div>' : "") +
      (localOnly() && stateAction === "aggregate" && state.head ? '<p class="cd-note">' +
        esc("Update " + job.checkpoint + " · server head v" + state.head.version + " · " + state.head.sha256) + '</p>' : "") +
      (localOnly() && stateAction === "export" ? '<p class="cd-note">' + esc("Update SHA-256 " + job.checkpoint) + '</p>' : "") +
      researchPicker +
      (federationGrant ? '<details><summary>Exact ' + stateAction + ' authorization</summary><label>Owner-only ' +
        esc(job.id + (stateAction === "export" ? ".submit.json" : ".aggregate.json")) +
        '<textarea readonly rows="4">' + esc(federationGrant) + '</textarea></label></details>' : "") +
      (stateAction === "return-model" ? '<p class="cd-note">' + esc("Requires " + job.id +
        ".return.json with exact client, job, verified model version/hash, contract, expiry and client_inventory_only purpose.") + '</p>' : "") +
      (prep ? '<p class="cd-note">' + esc((localOnly() && prep.status === "APPROVAL REQUIRED" ?
        "Ready for local confirmation" : prep.status) + " · " + prep.message) + '</p>' : "") +
      (prep?.schedule ? '<details><summary>Exact approval parameters</summary>' +
        list([...schedule(prep), value("Approval expires", timestamp(prep.expires_at)),
          value("Job ID", prep.job), value("Client / chunk", client + " / " + prep.chunk),
          value("Input model", "Global v0"), value("Optimizer / sampler", "SGD / Poisson"), value("EWC", "Disabled"),
          value("Logical rows", count(prep.full_rows)), value("Training ranks [start, stop)", prep.training_range?.join(" – ")),
          value("Validation ranks [start, stop)", prep.validation_range?.join(" – ")),
          value("Current / projected epsilon", prep.privacy?.current?.toFixed(6) + " / " + prep.privacy?.projected?.toFixed(6)),
          value("Projected budget headroom", prep.privacy?.headroom_after_run?.toFixed(6)),
          value("Processor", "CPU only"), value("Free / required RAM", gib(prep.ram_available) + " / " + gib(prep.ram_min)),
          value("Free / required disk", gib(prep.disk_available) + " / " + gib(prep.disk_min)),
          value("Maximum worker runtime", prep.maximum_seconds + " s"), value("Checkpoint destination", "Client-local · immutable attempt checkpoint"),
          value("Recovery", "Reconcile durable evidence before any new attempt. No automatic retry."),
          ...(!localOnly() ? [value("Required grant", "state/continual-real/approvals/" + prep.job + ".train.json")] : [])]) +
        (grant ? '<label>Owner-only grant JSON<textarea readonly rows="4">' + esc(grant) + '</textarea></label>' : "") + '</details>' : "") +
      (controls && prep?.status === "APPROVAL REQUIRED" ? '<div class="cd-controls">' + controls + '</div>' : "") +
      list(summary) + '<details><summary>Chunk and job identifiers</summary>' +
      list(details.filter(([key]) => ["Client ID", "Chunk", "Logical range", "Resources", "Job ID"].includes(key))) + '</details>');
  }

  function paintCommunication(job) {
    const i = health?.identity || {};
    const staleHealth = healthStale || health?.status !== "ONLINE";
    const activeStage = !stale && job?.active ? job.stages.find(s => s.state === "RUNNING")?.label?.replaceAll("FedAsync", "FedProx") : null;
    const arrow = activeStage === "Receive global model";
    const input = job?.approval ? model(job.approval.base) + " · " +
      (job.stages.find(s => s.label === "Receive global model")?.state || "PENDING") : "Input model pending";
    const outcome = job?.receipt ? model(job.receipt.version) + " · α " + job.receipt.alpha.toFixed(2) +
      " · staleness " + job.receipt.staleness : "Awaiting verified acceptance";
    set("#cd-communication", header("⇄", "Federated Communication (Live)") + '<div class="cd-link">' +
      '<div class="cd-node"><span>▣</span><strong>Client-' + esc(client) + '</strong><small>' +
      esc(i.active_ip || health?.configured_ip || "—") + (staleHealth ? ' · Unverified now' : '') + '</small></div>' +
      '<div class="cd-arrows ' + (arrow ? "transferring" : activeStage === "Send model update" ? "uploading" : "") + '"><span>← ' + esc(input) + '</span><span>→ ' +
      esc(job?.submission || "Update pending") + '</span></div>' +
      '<div class="cd-node"><span>▤</span><strong>FL Server</strong><small>' + esc(health?.server_ip) + '</small></div>' +
      '<div class="cd-arrows ' + (activeStage === "FL validation and FedProx" ? "transferring" :
        activeStage === "Receive updated global model" ? "uploading" : "") + '"><span>→ ' + esc(outcome) + '</span><span>← ' +
      esc(job?.returned ? model(job.returned.version) + " verified" : job?.receipt ? "Awaiting approved global model" : "Return not executed") + '</span></div>' +
      '<div class="cd-node purple"><span>⟳</span><strong>Aggregation</strong><small>FedProx</small></div></div>');
  }

  function paintLogs() {
    const entries = state?.logs || [];
    const ids = entries.map(row => row.id).join("|");
    const box = find("#cd-log-body");
    if (box && (ids !== logIds || !box.childElementCount)) {
      const scroll = box.scrollTop;
      const atEnd = box.scrollHeight - box.clientHeight - scroll < 35;
      box.innerHTML = entries.length ? entries.map(row => '<div data-event-id="' + esc(row.id) + '"><time>[' +
        esc(timestamp(row.at)) + ']</time> <b>' + esc(row.severity) + '</b> Client-' + esc(row.client) + ' · ' +
        esc(row.category) + ' · ' + esc(row.job) + ' · ' + esc(row.message) + '</div>').join("") :
        '<div>No persisted events available.</div>';
      box.scrollTop = atEnd ? box.scrollHeight : scroll;
      logIds = ids;
    }
    find("#cd-log-source").textContent = state ? "Application and worker events · " + state.system_logs : "Checking event sources…";
  }

  function paintHistory() {
    const rows = (state?.jobs || []).map(job => [
      '<code>' + esc(job.id.slice(0, 10)) + '…</code>', esc(job.chunk), esc(model(job.approval?.base)),
      esc(timestamp(job.started_at)), esc(timestamp(job.ended_at)),
      job.total ? esc((job.steps ?? 0) + " / " + job.total) : "—", job.approval ? "Enabled" : "—",
      job.approval ? job.approval.ewc ? "Enabled" : "Disabled" : "—",
      badge(job.status, level(job.status.toUpperCase())), job.checkpoint ? "Staged" : "—", esc(job.submission),
      job.receipt ? badge("Accepted", "green") : "—", esc(model(job.receipt?.version)),
      button("View Details", "history", job.id),
    ]);
    set("#cd-history", header("◷", "Client-" + client + " Run History") +
      table(["Job ID", "Chunk", "Input model", "Start time (IST)", "End time (IST)", "Steps", "DP-SGD",
        "EWC", "Result", "Checkpoint", "Update to FL", "FedProx", "Resulting model", "Receipt / Details"],
      rows, "No durable real training jobs recorded."));
  }

  function paint() {
    if (!find(".cd-shell") || scope() !== client) return;
    const job = focusedJob();
    const tag = find("#cd-feed");
    tag.textContent = stale ? "RECONNECTING" : state ? "Job feed connected" : "Connecting job feed";
    if (job?.last_event) tag.title = "Last durable job event " + timestamp(job.last_event);
    tag.classList.toggle("stale", stale);
    find("#cd-feedback").textContent = message;
    paintHealth();
    paintChunks();
    paintWorkflow(job);
    paintRun(job);
    paintCommunication(job);
    paintLogs();
    paintHistory();
    if (inspected) paintDetails();
  }

  function paintDetails() {
    const title = find("#cd-details-title");
    let pairs;
    if (inspected === "machine") {
      const i = health?.identity || {};
      const m = healthStale ? {} : health?.metrics || {};
      title.textContent = "Client-" + client + " machine details";
      pairs = [
        value("Status", healthStale ? "STALE" : health?.status), value("Hostname", i.hostname),
        value("Operating system", i.os), value("Configured address", health?.configured_ip),
        value("Detected address", i.active_ip), value("CPU utilization / cores", percent(m.cpu_percent) + " / " + count(m.cpu_cores)),
        value("RAM used / total", gib(m.ram_used_bytes) + " / " + gib(m.ram_total_bytes)),
        value("RAM available", gib(m.ram_available_bytes)), value("Disk used / total", gib(m.disk_used_bytes) + " / " + gib(m.disk_total_bytes)),
        value("Disk available", gib(m.disk_available_bytes)), value("Uptime", Number.isFinite(m.uptime_seconds) ?
          Math.floor(m.uptime_seconds / 3600) + "h " + Math.floor(m.uptime_seconds / 60) % 60 + "m" : null),
        value("Python worker", health?.status === "ONLINE" && !healthStale && i.runtime_available === true ?
          "Runtime executable verified" : "Unavailable"),
        value("Last checked", timestamp(health?.checked_at)), value("Last successful probe", timestamp(health?.last_success)),
        value("Probe error", health?.error ? healthErrors[health.error] || "Probe unavailable" : "—"),
      ];
    } else {
      const job = state?.jobs.find(row => row.id === inspected);
      if (!job) return;
      title.textContent = job.chunk + " · Run details";
      pairs = [...jobDetails(job), ...schedule(job.approval)];
      if (job.checkpoint) pairs.push(value("Local checkpoint SHA-256", job.checkpoint));
      if (job.evaluation_sha256) pairs.push(value("Local evaluation receipt SHA-256", job.evaluation_sha256));
      if (job.returned) pairs.push(value("Returned checkpoint SHA-256", job.returned.sha256),
        value("Returned model verified at", timestamp(job.returned.at)));
      if (job.receipt) pairs.push(value("Acceptance time", timestamp(job.receipt.created_at)),
        value("Update ID", job.id), value("Lineage parent", model(job.receipt.parent_version)),
        value("Staleness / alpha", job.receipt.staleness + " / " + job.receipt.alpha.toFixed(2)),
        value("Resulting checkpoint SHA-256", job.receipt.sha256));
    }
    set("#cd-details-body", list(pairs));
    find("#cd-details").hidden = false;
  }

  async function poll() {
    if (request || !client || scope() !== client) return;
    const current = epoch;
    request = true;
    try {
      const response = await api(route(), undefined, AbortSignal.timeout(6500));
      if (current !== epoch || scope() !== client) return;
      if (response.client !== client) throw new Error("Unexpected client");
      csrf(response.csrf);
      state = response;
      stale = false;
      if (!selectedRow()) {
        selected = response.jobs[0]?.chunk || response.chunks[0]?.id;
      }
      paint();
    } catch (error) {
      if (current !== epoch || scope() !== client) return;
      if (error.status === 401) return expired();
      stale = true;
      paint();
    } finally {
      if (current === epoch) request = false;
    }
  }

  async function pollHealth() {
    if (healthRequest || !client || scope() !== client) return;
    const current = epoch;
    healthRequest = true;
    try {
      const next = await api(route() + "/health", undefined, AbortSignal.timeout(6500));
      if (current !== epoch || scope() !== client) return;
      if (next.client !== client) throw new Error("Unexpected client probe");
      health = next;
      healthStale = false;
      paint();
    } catch (error) {
      if (current !== epoch || scope() !== client) return;
      if (error.status === 401) return expired();
      healthStale = true;
      paint();
    } finally {
      if (current === epoch) healthRequest = false;
    }
  }

  async function action(name, payload) {
    if (busy || stale) return;
    const current = epoch;
    busy = true;
    message = "Submitting request…";
    paint();
    try {
      const response = await api(route() + "/" + name, payload, AbortSignal.timeout(7000));
      if (current !== epoch || scope() !== client) return;
      message = response.status;
    } catch (error) {
      if (current !== epoch || scope() !== client) return;
      if (error.status === 401) return expired();
      message = error.status ? error.message : "Request outcome unconfirmed. Waiting for durable status.";
    } finally {
      if (current === epoch) {
        busy = false;
        void poll();
        paint();
      }
    }
  }

  function click(event) {
    const target = event.target.closest("[data-cd-select],[data-cd-history],[data-cd-machine],[data-cd-close],[data-cd-preflight],[data-cd-approve],[data-cd-federate],[data-cd-research]");
    if (!target) return false;
    const data = target.dataset;
    if (data.cdSelect !== undefined) {
      selected = data.cdSelect;
      message = "";
      sessionStorage.setItem("dashboard-chunk-" + client, selected);
      paint();
    } else if (data.cdHistory) {
      inspected = data.cdHistory;
      paintDetails();
      find("#cd-details-title").focus();
    } else if (data.cdMachine !== undefined) {
      inspected = "machine";
      paintDetails();
      find("#cd-details-title").focus();
    } else if (data.cdClose !== undefined) {
      const selector = inspected === "machine" ? "[data-cd-machine]" :
        `[data-cd-history="${CSS.escape(inspected)}"]`;
      inspected = null;
      find("#cd-details").hidden = true;
      find(selector)?.focus();
    } else if (data.cdPreflight) {
      void action("preflight", {chunk: selected});
    } else if (data.cdResearch !== undefined) {
      const job = focusedJob();
      const chosen = state.research_models.find(item => item.version === Number(data.cdResearch));
      if (job && chosen && confirm("Transfer " + model(chosen.version) + " to Client-" + client + " for research only?")) {
        void action("return-model", {chunk: job.chunk, job: job.id, confirm: true, version: chosen.version, sha256: chosen.sha256});
      }
    } else if (data.cdFederate) {
      const job = focusedJob();
      if (job && (data.cdFederate === "reconcile" || confirm("Confirm this exact " + data.cdFederate + "?"))) {
        const pins = localOnly() && data.cdFederate === "export" ? {sha256: job.checkpoint} :
          localOnly() && data.cdFederate === "aggregate" ? {version: state.head?.version, sha256: state.head?.sha256,
            update_sha256: job.checkpoint} : {};
        void action(data.cdFederate, {chunk: job.chunk, job: job.id, confirm: true, ...pins});
      }
    } else if (data.cdApprove && (localOnly() || confirm("Dispatch this exact owner-approved attempt? Training can consume privacy budget."))) {
      void action("approve", {chunk: selected, job: data.cdApprove, confirm: true});
    }
    return true;
  }

  function reset() {
    epoch++;
    client = state = health = selected = inspected = null;
    request = healthRequest = stale = healthStale = busy = false;
    message = logIds = "";
  }

  function mount(role) {
    if (client === role && find(".cd-shell")) return;
    reset();
    client = role;
    selected = sessionStorage.getItem("dashboard-chunk-" + client);
    document.title = "Client-" + client + " Dashboard | PPFCL-NIDS";
    app.innerHTML = '<div class="fl-shell cd-shell"><header class="fl-header"><div class="fl-brand"><span class="fl-icon" aria-hidden="true">✓</span><strong>PPFCL-NIDS</strong></div>' +
      '<span class="fl-description">Privacy-Preserving Federated Continual Learning Network Intrusion Detection System</span>' +
      '<div class="fl-header-status">' + dashboardPicker() + '<span id="cd-feed" role="status"></span><time id="cd-health-time"></time>' +
      (localOnly() ? "" : '<button class="fl-sign-out" type="button" data-action="logout">Sign out</button>') +
      '</div></header><main id="content" class="cd-main"><div class="cd-intro"><div class="cd-title"><span aria-hidden="true">▣</span><div><h1>Client-' +
      client + ' Dashboard</h1><p id="cd-role">Federated continual learning training client</p></div></div><div id="cd-summary"></div></div>' +
      '<p id="cd-feedback" class="cd-feedback" role="status"></p>' +
      '<section class="fl-inline-details" id="cd-details" aria-labelledby="cd-details-title" hidden><header><h2 id="cd-details-title" tabindex="-1"></h2>' +
      button("Close details", "close") + '</header><div class="fl-details-body" id="cd-details-body"></div></section>' +
      '<div class="cd-grid"><section class="cd-panel" id="cd-chunks"></section>' +
      '<section class="cd-panel" id="cd-workflow"></section><section class="cd-panel" id="cd-run"></section>' +
      '<section class="cd-panel" id="cd-communication"></section><section class="cd-panel" id="cd-logs">' +
      header("⌨", "System Logs (Live)") + '<p id="cd-log-source" class="cd-subline"></p><div id="cd-log-body" tabindex="0" aria-label="Authorized application events"></div></section></div>' +
      '<section class="cd-panel" id="cd-history"></section><footer class="fl-footer"><a data-page="client-' + client.toLowerCase() +
      '-tools" href="/client-' + client.toLowerCase() + '-tools">Synthetic client controls</a></footer></main></div>';
    paint();
    void poll();
    void pollHealth();
  }

  return {mount, reset, click, poll, pollHealth};
}
