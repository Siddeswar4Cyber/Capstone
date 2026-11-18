const healthRoute = "/api/fl-server/health";
const errors = {
  SSH_CONNECTION_OR_HOST_TRUST_FAILED: "Connection or host verification failed",
  SSH_AUTHENTICATION_FAILED: "SSH authentication failed",
  SSH_HOST_KEY_MISMATCH: "Pinned host key mismatch",
  SSH_CONNECTION_TIMEOUT: "SSH connection timed out",
  PROBE_TIMEOUT: "Probe timed out",
  PROBE_OR_ROUTE_FAILED: "Probe or pinned route unavailable",
  PROBE_CONFIGURATION_UNAVAILABLE: "Probe configuration unavailable",
  MEASUREMENTS_INCOMPLETE: "Some measurements unavailable",
};
const icon = (symbol, classes = "") => '<span class="fl-icon ' + classes + '" aria-hidden="true">' + symbol + '</span>';
const showPercent = value => Number.isFinite(value) ? value.toFixed(1) + "%" : "—";
const compactTime = new Intl.DateTimeFormat("en-IN", {
  timeZone: "Asia/Kolkata", day: "2-digit", month: "short", year: "numeric",
  hour: "2-digit", minute: "2-digit", hourCycle: "h23",
});
const tableTime = value => value ? compactTime.format(new Date(value)) : "—";
const memoryPercent = metrics => metrics?.ram_total_bytes && Number.isFinite(metrics.ram_used_bytes)
  ? 100 * metrics.ram_used_bytes / metrics.ram_total_bytes : null;
const uptime = seconds => Number.isFinite(seconds)
  ? [Math.floor(seconds / 86400) + "d", Math.floor(seconds / 3600) % 24 + "h", Math.floor(seconds / 60) % 60 + "m"].join(" ")
  : "Unavailable";

export function createServerView({app, api, esc, time, gib, table, list, dashboardPicker, expired, active, localOnly}) {
  let registry = null;
  let health = null;
  let registryError = "";
  let healthError = "";
  let healthRequest;
  let registryRequest;
  let updating = false;
  let generation = 0;
  let detail = null;

  const find = selector => app.querySelector(selector);
  const stamp = value => value ? time(value) : "—";
  const modelName = value => Number.isInteger(value) ? (value ? "Research v" + value : "Global v0") : "—";
  const client = value => value ? "Client " + value : "—";
  const status = (text, style) => '<span class="fl-tag ' + style + '">' + esc(text) + '</span>';
  const integrity = value => status(value, value === "Verified" ? "good" : "warn");
  const connection = value => status(value, "connection " + value.toLowerCase());
  const source = model => model.client ? "Client " + model.client + " " + model.chunk : model.source;
  const button = (label, kind, id) => '<button class="fl-view" type="button" data-fl-' + kind + '="' +
    esc(id) + '">' + esc(label) + '</button>';
  const ip = node => {
    const actual = node.identity.active_ip;
    const note = actual && actual !== node.configured_ip ? "Configured " + node.configured_ip :
      node.identity_historical ? "Last verified" : actual ? "" : "Configured";
    return '<span class="fl-ip">' + esc(actual || node.configured_ip) + '</span>' +
      (note ? '<small class="fl-caption">' + esc(note) + '</small>' : "");
  };

  function card(title, value, subtitle, color, symbol) {
    return '<article class="fl-card ' + color + '">' + icon(symbol) + '<div><h2>' + esc(title) +
      '</h2><strong>' + esc(value) + '</strong><p>' + esc(subtitle) + '</p></div></article>';
  }

  function paintCards() {
    const clients = health?.machines.filter(node => node.dataset) || [];
    const latest = registry?.aggregations.at(-1);
    const count = registry?.models.filter(model => model.integrity === "Verified").length;
    find("#fl-cards").innerHTML =
      card("Total Clients", health?.registered_clients ?? "—",
        health ? health.connected_clients + " connected · " +
          clients.map(node => node.name).join(", ") : "Connection data pending", "blue", "♟") +
      card("Global Models", count ?? "—",
        registry?.models.filter(model => model.integrity === "Verified").map(model => "v" + model.version).join(" · ") ||
          "No verified models", "green", "▱") +
      card("Aggregations", registry?.registry_readable ? registry.aggregations.length : "—",
        latest ? "FedProx · Accepted" : "No verified aggregation", "purple", "⟳") +
      card("Last Aggregation", latest ? tableTime(latest.created_at) + " IST" : "No verified aggregation",
        latest ? source(latest) + " → " + latest.name : "", "gold fl-last", "✓");
  }

  function modelCard(label, model, style) {
    if (!model) return '<article class="fl-model-card"><h3>' + label + '</h3><p>Verified record unavailable</p></article>';
    return '<article class="fl-model-card ' + style + '">' + icon(style === "approved" ? "▱" : "◇") +
      '<div><h3>' + label + '</h3><div class="fl-model-name"><strong>' + esc(model.name) + '</strong>' +
      status(model.status, style === "approved" ? "good" : "warn") + '</div><p>' +
      esc(model.architecture?.join(" → ")) + ' · ' + esc(model.contract) + '</p><p>' +
      esc(model.time_basis || "Created") + ' · ' + esc(model.created_at ? tableTime(model.created_at) + " IST" : "Unavailable") +
      '</p><p><code>SHA-256 ' +
      esc(model.sha256.slice(0, 12)) + '…</code> ' + integrity(model.integrity) + '</p></div>' +
      button("View Details", "model", model.version) + '</article>';
  }

  function paintRegistry() {
    const models = registry?.models || [];
    const latest = models.filter(model => model.type === "Research").at(-1);
    find("#fl-model-data").innerHTML = (registryError ? '<p class="fl-alert">' + esc(registryError) + '</p>' : "") +
      modelCard("Approved Global Model", registry?.baseline, "approved") +
      modelCard("Latest Research Model", latest, "research") +
      '<h3 class="fl-table-title">Model History (Lineage)</h3>' +
      (registry?.baseline ? '<p class="fl-lineage">' +
        [registry.baseline, ...registry.aggregations].map(model => esc(model.name)).join(" → ") + '</p>' : "") +
      table(["Version", "Type", "Created At (IST)", "Source", "Contributors", "Status", "Integrity", "Details"],
        models.slice().reverse().map(model => [
          '<strong>' + esc(model.name) + '</strong>', esc(model.type),
          esc(tableTime(model.created_at)) + (model.time_basis === "Registry designation" ? '<small class="fl-caption">Designation</small>' : ""),
          esc(source(model)), esc(model.contributors.map(client).join(", ") || "—"),
          status(model.type === "Research" ? "Experimental" : "Approved baseline", model.type === "Research" ? "warn" : "good"),
          integrity(model.integrity), button("View", "model", model.version),
        ]), registry ? "No verified model records available." : "Loading model registry…");

    const fed = registry?.aggregations.length ? "Implemented · ordered aggregation verified" : "No verified aggregation";
    const dp = registry?.dp_sgd_runs.length ? "Bounded training verified · " + registry.dp_sgd_runs.join(", ") : "Evidence unavailable";
    const ewc = [registry?.ewc_synthetic_verified ? "Synthetic testing verified" : "Synthetic evidence unavailable",
      registry?.ewc_disabled_runs.length ? "disabled in " + registry.ewc_disabled_runs.join(", ") : ""].filter(Boolean).join(" · ");
    find("#fl-info").innerHTML = '<div class="fl-info-title">' + icon("i") + '<h2>System Information</h2></div>' +
      list([["FedProx", esc(fed)], ["DP-SGD", esc(dp)], ["EWC", esc(ewc)],
        ["Model validation", latest ? esc(latest.name + " · experimental, independently unvalidated") : "Research evidence unavailable"]]);
    paintCards();
  }

  function paintHealth() {
    const rows = health?.machines || [];
    const windows = rows.find(node => node.id === "Windows");
    find("#fl-dashboard-status").textContent = healthError ? "Health data unavailable" : health ? "Dashboard Online" : "Checking dashboard";
    find("#fl-dashboard-status").classList.toggle("healthy", Boolean(health && !healthError));
    find("#fl-header-time").textContent = windows?.last_checked ? stamp(windows.last_checked) : "Asia/Kolkata (IST)";
    find("#fl-machine-data").innerHTML = (healthError ? '<p class="fl-alert">' + esc(healthError) + '</p>' : "") +
      table(["Machine", "Role", "IP Address", "Operating System", "Status", "CPU", "RAM", "Last Checked (IST)", "Details"],
        rows.map(node => [
          '<strong>' + esc(node.name) + '</strong>', esc(node.role), ip(node),
          esc(node.identity.os || "Unavailable") + (node.identity_historical ? '<small class="fl-caption">Last verified</small>' : ""),
          connection(node.status), showPercent(node.metrics?.cpu_percent), showPercent(memoryPercent(node.metrics)),
          esc(tableTime(node.last_checked)), button("View", "machine", node.id),
        ]), health ? "No machine entries available." : "Loading machine observations…");
    find("#fl-refresh-state").textContent = updating || health?.refreshing ? "Checking machines…" :
      windows?.last_checked ? "Last checked " + stamp(windows.last_checked) : "";
    find("[data-fl-refresh]").setAttribute("aria-busy", String(updating || Boolean(health?.refreshing)));
    paintCards();
    paintDetails();
  }

  function paintDetails() {
    if (!detail || !find("#fl-details")) return;
    let title, pairs;
    if (detail.kind === "model") {
      const model = registry?.models.find(row => String(row.version) === detail.id);
      if (!model) return;
      title = model.name;
      pairs = [
        ["SHA-256", '<code>' + esc(model.sha256) + '</code>'], ["Architecture", esc(model.architecture?.join(" → "))],
        ["Feature contract", esc(model.contract)], ["Source", esc(source(model))],
        ["Contributors", esc(model.contributors.map(client).join(", ") || "—")],
        ["Training base", esc(modelName(model.base_version))], ["Lineage parent", esc(modelName(model.parent_version))],
        ["Accepted update ID", esc(model.update_id)], ["FedProx alpha", Number.isFinite(model.alpha) ? model.alpha.toFixed(2) : "—"],
        ["Staleness", esc(model.staleness)], [model.time_basis || "Created", esc(stamp(model.created_at))],
        ["Checkpoint integrity", integrity(model.integrity)], ["Approval", esc(model.status)],
        ["Evaluation", esc(model.evaluation)], ["Validation limitations", esc(model.limitations.join(" · ") || "Unavailable")],
      ];
    } else {
      const node = health?.machines.find(row => row.id === detail.id);
      if (!node) return;
      const m = node.metrics || {};
      const i = node.identity;
      title = node.name;
      pairs = [
        ["Role", esc(node.role)], ["Hostname", esc(i.hostname || "Unavailable")],
        ["Operating system", esc(i.os || "Unavailable")], ["Host-only IP", ip(node)],
        ["Observed interface IPs", esc(i.addresses?.join(", ") || "Unavailable")],
        ["Connection status", connection(node.status)], ["SSH", esc(node.ssh)],
        ["Last checked", esc(stamp(node.last_checked))], ["Last successful probe", esc(stamp(node.last_success))],
        ["Probe result", esc(node.error ? errors[node.error] || "Probe unavailable" : node.status === "ONLINE" ? "Succeeded" : "Unavailable")],
        ["CPU utilization", showPercent(m.cpu_percent)], ["RAM used / total", gib(m.ram_used_bytes) + " / " + gib(m.ram_total_bytes)],
        ["RAM available", gib(m.ram_available_bytes)], ["Disk used / total", gib(m.disk_used_bytes) + " / " + gib(m.disk_total_bytes)],
        ["Disk available", gib(m.disk_available_bytes)], ["Uptime", uptime(m.uptime_seconds)],
      ];
      if (node.dataset) {
        const job = registry?.training[node.name.at(-1)];
        pairs.push(["Assigned dataset", esc(node.dataset)], ["Worker availability", i.runtime_available && node.status === "ONLINE"
          ? "Client runtime executable · worker service unverified" : i.runtime_available === false && node.status === "ONLINE" ? "Runtime unavailable" : "Unavailable"],
          ["Latest completed training", job ? esc(job.chunk + " · " + stamp(job.completed_at)) : "Unavailable"],
          ["Latest checkpoint", esc(job?.checkpoint || "Unavailable")]);
      }
      if (node.id === "organization") {
        const org = health.organization;
        const model = org.inference_model;
        pairs.push(["Monitoring agent", esc(org.agent)], ["Last genuine flow heartbeat", esc(stamp(org.heartbeat_at))],
          ["Authorized inference model", model ? esc(modelName(model.version) + " · " + model.scope) +
            "<br><code>" + esc(model.sha256) + "</code>" : "Unavailable"]);
      }
      if (node.id === "Windows") pairs.push(["Dashboard process", healthError ? "Health data unavailable" : "Responding"],
        ["Coordinator / registry", registry?.registry_readable && registry.baseline ? "Registry readable · baseline verified" : "Readiness unverified"],
        ["Aggregation activity", "No live job telemetry"]);
    }
    find("#fl-details-title").textContent = title;
    find("#fl-details-body").innerHTML = list(pairs);
  }

  async function loadRegistry() {
    if (registryRequest) return registryRequest;
    const current = generation;
    registryRequest = (async () => {
      try {
        const value = await api("/api/fl-server");
        if (current !== generation) return;
        registry = value;
        registryError = "";
      } catch (error) {
        if (current !== generation) return;
        if (error.status === 401) return expired();
        registry = null;
        registryError = "Model registry request failed.";
      } finally {
        if (current === generation) registryRequest = null;
      }
      if (active() && find("#fl-model-data")) {
        paintRegistry();
        paintDetails();
      }
    })();
    return registryRequest;
  }

  async function loadHealth() {
    if (healthRequest) return healthRequest;
    const current = generation;
    healthRequest = (async () => {
      try {
        const value = await api(healthRoute);
        if (current !== generation) return;
        health = value;
        healthError = "";
      } catch (error) {
        if (current !== generation) return;
        if (error.status === 401) return expired();
        health = null;
        healthError = "Machine health request failed.";
      } finally {
        if (current === generation) healthRequest = null;
      }
      if (active() && find("#fl-machine-data")) paintHealth();
    })();
    return healthRequest;
  }

  async function refresh() {
    if (updating) return;
    updating = true;
    const current = generation;
    find("#fl-refresh-state").textContent = "Requesting fresh probes…";
    try {
      const result = await api(healthRoute, {});
      await Promise.all([loadRegistry(), loadHealth()]);
      if (current !== generation || !active()) return;
      if (!result.queued && !result.refreshing) {
        find("#fl-refresh-state").textContent = "Recently checked · retry in " + result.cooldown_seconds + " s";
      }
    } catch (error) {
      if (current !== generation || !active()) return;
      if (error.status === 401) return expired();
      find("#fl-refresh-state").textContent = "Refresh request failed.";
    } finally {
      if (current === generation) updating = false;
    }
  }

  function mount() {
    document.title = "FL Server Dashboard | PPFCL-NIDS";
    if (find(".fl-shell")) return;
    detail = null;
    app.innerHTML = '<div class="fl-shell"><header class="fl-header"><div class="fl-brand">' + icon("✓") +
      '<strong>PPFCL-NIDS</strong></div><span class="fl-description">Privacy-Preserving Federated Continual Learning Network Intrusion Detection System</span>' +
      '<div class="fl-header-status">' + dashboardPicker() + '<span id="fl-dashboard-status"></span><time id="fl-header-time"></time>' +
      (localOnly() ? "" : '<button class="fl-sign-out" type="button" data-action="logout">Sign out</button>') + '</div></header>' +
      '<main id="content" class="fl-main"><div class="fl-heading">' + icon("▤") +
      '<div><h1>FL-Server Dashboard</h1><p>Federated Learning Server – Global Models &amp; Aggregation</p></div></div>' +
      '<section id="fl-cards" class="fl-cards" aria-label="Summary statistics"></section>' +
      '<section class="fl-inline-details" id="fl-details" aria-labelledby="fl-details-title" hidden><header><h2 id="fl-details-title" tabindex="-1"></h2>' +
      '<button type="button" class="fl-refresh" data-fl-close>Close details</button></header><div class="fl-details-body" id="fl-details-body"></div></section>' +
      '<div class="fl-columns"><section class="fl-panel"><h2>' + icon("▱") +
      'Global Model (Version &amp; History)</h2><div id="fl-model-data"></div></section>' +
      '<section class="fl-panel"><div class="fl-panel-head"><h2>' + icon("▣") +
      'Registered Machines</h2><button class="fl-refresh" type="button" data-fl-refresh>↻ Refresh</button></div>' +
      '<div id="fl-machine-data"></div><p id="fl-refresh-state" class="fl-refresh-state" role="status"></p>' +
      '<section id="fl-info" class="fl-info" aria-label="System Information"></section></section></div>' +
      '<footer class="fl-footer"><a href="/overview" data-page="overview">Research console</a>' +
      '<a href="/server-tools" data-page="server-tools">Synthetic session controls</a></footer></main></div>';
    paintRegistry();
    paintHealth();
    void loadRegistry();
    void loadHealth();
  }

  function click(event) {
    const target = event.target.closest("[data-fl-refresh],[data-fl-model],[data-fl-machine],[data-fl-close]");
    if (!target) return false;
    if (target.hasAttribute("data-fl-refresh")) void refresh();
    else if (target.hasAttribute("data-fl-close")) {
      const {kind, id} = detail;
      detail = null;
      find("#fl-details").hidden = true;
      find(`[data-fl-${kind}="${CSS.escape(id)}"]`)?.focus();
    }
    else {
      detail = target.dataset.flModel !== undefined ? {kind: "model", id: target.dataset.flModel} :
        {kind: "machine", id: target.dataset.flMachine};
      paintDetails();
      find("#fl-details").hidden = false;
      find("#fl-details-title").focus();
    }
    return true;
  }

  return {mount, click, loadHealth, reset() {
    generation += 1;
    health = registry = detail = null;
    healthRequest = registryRequest = null;
    registryError = healthError = "";
    updating = false;
  }};
}
