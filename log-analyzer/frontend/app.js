/* ═══════════════════════════════════════════════════════════════════════════
   Log Analyzer — SPA
   ═══════════════════════════════════════════════════════════════════════════ */
'use strict';

// ── Constants ────────────────────────────────────────────────────────────────
const POLL_INTERVAL = 1500;  // ms
const SUGGESTIONS = [
  "What types of errors occurred most frequently?",
  "Are there any error bursts or spikes?",
  "What system components are most active?",
  "Show me the first 10 events in the log",
  "What happened during initialization?",
  "Summarize the overall health of this system",
];

// ── State ─────────────────────────────────────────────────────────────────────
let pollTimer     = null;
let charts        = {};
let chatWs        = null;
let chatMode      = 'agent';   // 'agent' | 'rag'
let wsStreaming    = false;
let currentPage   = 1;
let tlFilters     = { level: '', keyword: '', ts_from: '', ts_to: '' };
let tlLimit       = 50;
let currentBubble = null;   // DOM element being streamed into
let bubbleText    = '';     // raw text accumulating in stream
let selectedModel = null;   // null = server default

// ── Settings ──────────────────────────────────────────────────────────────────
function loadSettings() {
  const s = JSON.parse(localStorage.getItem('logai_settings') || '{}');
  const html = document.documentElement;
  html.dataset.theme     = s.theme     || 'dark';
  html.dataset.palette   = s.palette   || 'violet';
  html.dataset.density   = s.density   || 'normal';
  html.dataset.fontScale = s.fontScale || 1;
  html.style.setProperty('--font-scale', s.fontScale || 1);

  // Sync settings UI
  document.querySelectorAll('.seg-btn[data-setting="theme"]').forEach(b =>
    b.classList.toggle('active', b.dataset.value === (s.theme || 'dark')));
  document.querySelectorAll('.seg-btn[data-setting="density"]').forEach(b =>
    b.classList.toggle('active', b.dataset.value === (s.density || 'normal')));
  document.querySelectorAll('.palette-dot').forEach(d =>
    d.classList.toggle('active', d.dataset.palette === (s.palette || 'violet')));
  const fs = parseFloat(s.fontScale || 1);
  document.getElementById('font-scale').value = fs;
  document.getElementById('font-scale-val').textContent = Math.round(fs * 100) + '%';
}

function saveSetting(key, value) {
  const s = JSON.parse(localStorage.getItem('logai_settings') || '{}');
  s[key] = value;
  localStorage.setItem('logai_settings', JSON.stringify(s));
}

function applyTheme(v)    { document.documentElement.dataset.theme = v;     saveSetting('theme', v);     rebuildCharts(); }
function applyPalette(v)  { document.documentElement.dataset.palette = v;   saveSetting('palette', v);   rebuildCharts(); }
function applyDensity(v)  { document.documentElement.dataset.density = v;   saveSetting('density', v);   }
function applyFontScale(v){ document.documentElement.style.setProperty('--font-scale', v); document.documentElement.dataset.fontScale = v; saveSetting('fontScale', v); }

// ── Router ────────────────────────────────────────────────────────────────────
function navigateTo(page) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('page-' + page).classList.add('active');
  document.querySelector(`.nav-item[data-page="${page}"]`)?.classList.add('active');
  if (page === 'timeline' && document.getElementById('tl-tbody').children.length === 0) loadTimeline();
  if (page === 'analytics') loadAnalytics();
}

// ── Upload flow ───────────────────────────────────────────────────────────────
function initUpload() {
  const dropZone  = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');

  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) startUpload(fileInput.files[0]);
  });

  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault(); dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files[0]) startUpload(e.dataTransfer.files[0]);
  });

  document.getElementById('upload-new-btn').addEventListener('click', () => {
    document.getElementById('upload-overlay').classList.add('active');
    document.getElementById('app').classList.add('hidden');
  });
}

async function startUpload(file) {
  const prog  = document.getElementById('upload-progress');
  const bar   = document.getElementById('progress-bar');
  const label = document.getElementById('progress-label');
  prog.classList.remove('hidden');
  bar.style.width = '10%';
  label.textContent = 'Uploading…';

  const fd = new FormData();
  fd.append('file', file);

  try {
    const res = await fetch('/api/upload', { method: 'POST', body: fd });
    if (!res.ok) throw new Error(await res.text());
    bar.style.width = '20%';
    label.textContent = 'Processing…';
    _lastDetailCount = 0;
    const logEl = document.getElementById('progress-log');
    if (logEl) logEl.innerHTML = '';
    startPolling();
  } catch (e) {
    label.textContent = 'Error: ' + e.message;
  }
}

function startPolling() {
  clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    try {
      const s = await fetch('/api/status').then(r => r.json());
      updateProgress(s);
      if (!s.is_processing && (s.has_summary || s.error)) {
        clearInterval(pollTimer);
        if (s.has_summary) onLoadComplete();
        else showError(s.error);
      }
    } catch (e) { /* network hiccup, retry */ }
  }, POLL_INTERVAL);
}

const STEP_PROGRESS = {
  'Uploading…':                         8,
  'Parsing log file…':                 20,
  'Characterising log with AI…':       45,
  'Building RAG index… (chunking)':    58,
  'Building RAG index… (loading model)': 65,
  'Building RAG index… (encoding)':    72,
  'Ready':                            100,
};

let _lastDetailCount = 0;

function updateProgress(s) {
  const bar   = document.getElementById('progress-bar');
  const label = document.getElementById('progress-label');
  const pct   = document.getElementById('progress-pct');

  // Compute progress — if step contains a % use that
  let prog = STEP_PROGRESS[s.step];
  if (!prog) {
    const m = s.step && s.step.match(/\((\d+)%/);
    prog = m ? (65 + Math.round(parseInt(m[1]) * 0.25)) : (s.is_processing ? 50 : 100);
  }
  bar.style.width = prog + '%';
  if (label) label.textContent = s.step || 'Processing…';
  if (pct) pct.textContent = prog + '%';

  // Append new detail lines
  const logEl = document.getElementById('progress-log');
  if (logEl && s.detail) {
    const newLines = s.detail.slice(_lastDetailCount);
    newLines.forEach((line, idx) => {
      const isLast = (idx === newLines.length - 1) && s.is_processing;
      const div = document.createElement('div');
      div.className = 'plog-line';
      const tick = document.createElement('span');
      tick.className = isLast ? 'plog-tick spin' : 'plog-tick';
      tick.textContent = isLast ? '◌' : '✓';
      div.appendChild(tick);
      const txt = document.createElement('span');
      txt.textContent = line;
      div.appendChild(txt);
      logEl.appendChild(div);
    });
    if (newLines.length) logEl.scrollTop = logEl.scrollHeight;
    _lastDetailCount = s.detail.length;
  }
}

async function onLoadComplete() {
  document.getElementById('upload-overlay').classList.remove('active');
  document.getElementById('app').classList.remove('hidden');

  const summary = await fetch('/api/summary').then(r => r.json());
  const status  = await fetch('/api/status').then(r => r.json());

  populateDashboard(summary, status);
  populateSuggestions(summary);
  openWebSocket();
}

function showError(msg) {
  document.getElementById('progress-label').textContent = 'Error: ' + (msg || 'unknown');
  document.getElementById('progress-bar').style.background = 'var(--c-error)';
}

// ── Dashboard ─────────────────────────────────────────────────────────────────
function populateDashboard(summary, status) {
  const levels = summary.level_counts || {};
  const total = summary.total_events || 0;
  const errors = (levels.ERROR || 0) + (levels.CRITICAL || 0);
  const warnings = levels.WARNING || 0;

  document.getElementById('dash-filename').textContent = status.filename || 'Loaded';
  document.getElementById('s-total').textContent    = fmt(total);
  document.getElementById('s-errors').textContent   = fmt(errors);
  document.getElementById('s-warnings').textContent = fmt(warnings);
  document.getElementById('s-loggers').textContent  = fmt((summary.top_loggers || []).length);

  // Error rate percentage
  const errorRate = total > 0 ? ((errors / total) * 100).toFixed(1) : 0;
  document.getElementById('s-error-rate').textContent = `${errorRate}% error rate`;

  // Health score: 100 - error_rate - (warnings_rate * 0.5)
  const warningRate = total > 0 ? (warnings / total) * 100 : 0;
  const health = Math.max(0, Math.min(100, 100 - errorRate - (warningRate * 0.5)));
  const healthColor = health > 80 ? 'var(--c-success)' : health > 50 ? 'var(--c-warning)' : 'var(--c-error)';
  const healthLabel = health > 80 ? '✓ Good' : health > 50 ? '⚠ Fair' : '✗ Poor';
  document.getElementById('s-health').textContent = Math.round(health) + '%';
  document.querySelector('[id="s-health"]').parentElement.parentElement.style.borderLeftColor = healthColor;

  const dr = summary.date_range || {};
  const span = dr.span_hours > 48
    ? Math.round(dr.span_hours / 24) + ' days'
    : (dr.span_hours || 0).toFixed(1) + ' hrs';
  document.getElementById('s-span').textContent = span;

  // Chips
  const chipsEl = document.getElementById('dash-chips');
  chipsEl.innerHTML = '';
  if (dr.first) chipsEl.insertAdjacentHTML('beforeend', `<span class="chip chip-info">${dr.first.slice(0,10)} → ${(dr.last||'').slice(0,10)}</span>`);
  chipsEl.insertAdjacentHTML('beforeend', `<span class="chip chip-error">${fmt(errors)} errors</span>`);

  // Characterization
  const char = summary.characterization || {};
  if (char.log_type) {
    document.getElementById('char-card').style.display = '';
    document.getElementById('char-type').textContent = char.log_type;
    document.getElementById('char-desc').textContent = char.system_description || '';
    const entEl = document.getElementById('char-entities');
    entEl.innerHTML = '';
    (char.key_entities || []).slice(0, 12).forEach(e => {
      entEl.insertAdjacentHTML('beforeend', `<span class="chip">${e}</span>`);
    });
  }

  // Level chart
  buildLevelChart(levels);
  // Loggers chart
  buildLoggerChart(summary.top_loggers || []);

  // Error bursts
  const bursts = summary.error_bursts || [];
  if (bursts.length > 0) {
    document.getElementById('burst-card').style.display = '';
    const tbody = document.querySelector('#burst-table tbody');
    tbody.innerHTML = '';
    bursts.forEach(b => {
      tbody.insertAdjacentHTML('beforeend', `<tr><td class="ts-cell">${b.hour}</td><td><span class="level-badge level-ERROR">${fmt(b.count)}</span></td></tr>`);
    });
  }

  // Error samples
  const samples = summary.error_samples || [];
  if (samples.length > 0) {
    document.getElementById('samples-card').style.display = '';
    const samplesList = document.getElementById('error-samples-list');
    samplesList.innerHTML = '';
    samples.slice(0, 8).forEach(s => {
      samplesList.insertAdjacentHTML('beforeend', `
        <div class="error-sample-item">
          <div class="error-sample-meta">${s.logger} · ${s.ts}</div>
          <div class="error-sample-msg" title="${s.msg}">${s.msg.substring(0, 120)}${s.msg.length > 120 ? '…' : ''}</div>
        </div>`);
    });
  }
}

// ── Charts ────────────────────────────────────────────────────────────────────
function getChartColors() {
  const style = getComputedStyle(document.documentElement);
  return {
    primary:  style.getPropertyValue('--c-primary').trim(),
    secondary:style.getPropertyValue('--c-secondary').trim(),
    tertiary: style.getPropertyValue('--c-tertiary').trim(),
    error:    style.getPropertyValue('--c-error').trim(),
    warning:  style.getPropertyValue('--c-warning').trim(),
    success:  style.getPropertyValue('--c-success').trim(),
    info:     style.getPropertyValue('--c-info').trim(),
    surface:  style.getPropertyValue('--c-surface').trim(),
    onSurface:style.getPropertyValue('--c-on-surface').trim(),
    outline:  style.getPropertyValue('--c-outline').trim(),
  };
}

Chart.defaults.font.family = 'Inter, system-ui, sans-serif';
Chart.defaults.font.size   = 11;

function buildLevelChart(levels) {
  const c  = getChartColors();
  const labels = Object.keys(levels);
  const colorMap = { DEBUG: c.info, INFO: c.success, WARNING: c.warning, ERROR: c.error, CRITICAL: c.error, RAW: c.outline };
  const colors   = labels.map(l => colorMap[l] || c.primary);
  const ctx      = document.getElementById('chart-levels').getContext('2d');
  if (charts.levels) charts.levels.destroy();
  charts.levels = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{ data: labels.map(l => levels[l]), backgroundColor: colors, borderWidth: 0, hoverOffset: 8 }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { position: 'right', labels: { color: c.onSurface, boxWidth: 12, padding: 10 } },
        tooltip: { callbacks: { label: ctx => ` ${ctx.label}: ${fmt(ctx.raw)}` } },
      },
    },
  });
}

function buildLoggerChart(topLoggers) {
  const c      = getChartColors();
  const top    = topLoggers.slice(0, 10);
  const labels = top.map(l => l.logger.split('.').pop() || l.logger);
  const ctx    = document.getElementById('chart-loggers').getContext('2d');
  if (charts.loggers) charts.loggers.destroy();
  charts.loggers = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        { label: 'Events', data: top.map(l => l.count),       backgroundColor: c.primary  + 'cc', borderRadius: 4 },
        { label: 'Errors', data: top.map(l => Math.round(l.count * l.error_rate)), backgroundColor: c.error   + 'cc', borderRadius: 4 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: c.onSurface } }, tooltip: {} },
      scales: {
        x: { ticks: { color: c.onSurface, maxRotation: 30 }, grid: { color: c.outline + '40' } },
        y: { ticks: { color: c.onSurface }, grid: { color: c.outline + '40' } },
      },
    },
  });
}

function buildHourChart(eventsPerHour) {
  const c       = getChartColors();
  const labels  = Object.keys(eventsPerHour).map(h => h.slice(11)); // HH:00
  const values  = Object.values(eventsPerHour);
  const ctx     = document.getElementById('chart-hours').getContext('2d');
  if (charts.hours) charts.hours.destroy();
  charts.hours = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets: [{ label: 'Events', data: values, backgroundColor: c.primary + 'aa', borderRadius: 3 }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: c.onSurface, maxTicksLimit: 24, maxRotation: 45 }, grid: { color: c.outline + '40' } },
        y: { ticks: { color: c.onSurface }, grid: { color: c.outline + '40' } },
      },
    },
  });
}

function buildErrorHourChart(errorsPerHour) {
  const c      = getChartColors();
  const labels = Object.keys(errorsPerHour).map(h => h.slice(11));
  const values = Object.values(errorsPerHour);
  const ctx    = document.getElementById('chart-errors-hour').getContext('2d');
  if (charts.errHour) charts.errHour.destroy();
  charts.errHour = new Chart(ctx, {
    type: 'line',
    data: {
      labels, datasets: [{
        label: 'Errors', data: values,
        borderColor: c.error, backgroundColor: c.error + '22',
        fill: true, tension: 0.3, pointRadius: 2,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: c.onSurface, maxTicksLimit: 12, maxRotation: 45 }, grid: { color: c.outline + '40' } },
        y: { ticks: { color: c.onSurface }, grid: { color: c.outline + '40' } },
      },
    },
  });
}

function rebuildCharts() {
  // Re-render all charts after theme/palette change
  fetch('/api/summary').then(r => r.json()).then(s => {
    buildLevelChart(s.level_counts || {});
    buildLoggerChart(s.top_loggers || []);
    buildHourChart(s.events_per_hour || {});
    buildErrorHourChart(s.errors_per_hour || {});
  }).catch(() => {});
}

// ── Analytics ─────────────────────────────────────────────────────────────────
async function loadAnalytics() {
  try {
    const s = await fetch('/api/summary').then(r => r.json());
    buildHourChart(s.events_per_hour || {});
    buildErrorHourChart(s.errors_per_hour || {});
    buildPatternList(s.top_patterns || []);
    buildEntityGrid(s.entities || {});
  } catch (e) { console.warn('Analytics load failed', e); }
}

function buildPatternList(patterns) {
  const el  = document.getElementById('pattern-list');
  el.innerHTML = '';
  if (!patterns.length) { el.textContent = 'No patterns available.'; return; }
  const max = patterns[0].count;
  patterns.slice(0, 20).forEach(p => {
    const pct = Math.round(p.count / max * 100);
    el.insertAdjacentHTML('beforeend', `
      <div class="pattern-item">
        <span class="pattern-text" title="${p.pattern}">${p.pattern}</span>
        <div class="pattern-bar-bg"><div class="pattern-bar" style="width:${pct}%"></div></div>
        <span class="pattern-count">${fmt(p.count)}</span>
      </div>`);
  });
}

function buildEntityGrid(entities) {
  const el = document.getElementById('entity-grid');
  el.innerHTML = '';
  const groups = [
    { key: 'ip_addresses',  label: 'IP Addresses' },
    { key: 'quoted_values', label: 'Quoted Values' },
    { key: 'file_paths',    label: 'File Paths' },
  ];
  groups.forEach(g => {
    const data = entities[g.key] || {};
    if (!Object.keys(data).length) return;
    const div = document.createElement('div');
    div.className = 'entity-group';
    div.innerHTML = `<h4>${g.label}</h4>`;
    Object.entries(data).slice(0, 15).forEach(([k, v]) => {
      div.insertAdjacentHTML('beforeend', `
        <div class="entity-item">
          <span class="entity-key" title="${k}">${k}</span>
          <span class="entity-val">${fmt(v)}×</span>
        </div>`);
    });
    el.appendChild(div);
  });
}

// ── Timeline ──────────────────────────────────────────────────────────────────
async function loadTimeline(page = 1) {
  currentPage = page;
  const limit = parseInt(document.getElementById('tl-limit').value, 10);
  const params = new URLSearchParams({
    page, limit,
    level:   tlFilters.level,
    keyword: tlFilters.keyword,
    ts_from: tlFilters.ts_from,
    ts_to:   tlFilters.ts_to,
  });
  try {
    const data = await fetch('/api/events?' + params).then(r => r.json());
    renderTimeline(data);
  } catch (e) { console.warn('Timeline load failed', e); }
}

function renderTimeline(data) {
  const tbody = document.getElementById('tl-tbody');
  tbody.innerHTML = '';
  if (!data.events || data.events.length === 0) {
    tbody.insertAdjacentHTML('beforeend', `<tr><td colspan="4" style="text-align:center;padding:2rem;color:var(--c-on-surface-var)">No events match your filters.</td></tr>`);
  } else {
    data.events.forEach(e => {
      tbody.insertAdjacentHTML('beforeend', `
        <tr>
          <td class="ts-cell">${e.ts}</td>
          <td><span class="level-badge level-${e.level}">${e.level}</span></td>
          <td class="logger-cell" title="${e.logger}">${e.logger}</td>
          <td class="msg-cell">${esc(e.msg)}</td>
        </tr>`);
    });
  }
  // Pagination
  document.getElementById('tl-page-info').textContent = `Page ${data.page} / ${data.pages || 1}  (${fmt(data.total)} events)`;
  document.getElementById('tl-prev').disabled = data.page <= 1;
  document.getElementById('tl-next').disabled = data.page >= (data.pages || 1);
}

function initTimeline() {
  document.getElementById('tl-search-btn').addEventListener('click', () => {
    tlFilters.level   = document.getElementById('tl-level').value;
    tlFilters.keyword = document.getElementById('tl-keyword').value;
    tlFilters.ts_from = document.getElementById('tl-from').value;
    tlFilters.ts_to   = document.getElementById('tl-to').value;
    loadTimeline(1);
  });
  document.getElementById('tl-clear-btn').addEventListener('click', () => {
    tlFilters = { level: '', keyword: '', ts_from: '', ts_to: '' };
    document.getElementById('tl-level').value   = '';
    document.getElementById('tl-keyword').value = '';
    document.getElementById('tl-from').value    = '';
    document.getElementById('tl-to').value      = '';
    loadTimeline(1);
  });
  document.getElementById('tl-refresh-btn').addEventListener('click', () => loadTimeline(currentPage));
  document.getElementById('tl-prev').addEventListener('click', () => loadTimeline(currentPage - 1));
  document.getElementById('tl-next').addEventListener('click', () => loadTimeline(currentPage + 1));
  document.getElementById('tl-limit').addEventListener('change', () => loadTimeline(1));

  // Enter key in keyword input
  document.getElementById('tl-keyword').addEventListener('keydown', e => {
    if (e.key === 'Enter') document.getElementById('tl-search-btn').click();
  });
}

// ── Rich Component Renderer ───────────────────────────────────────────────────

const COLOR_MAP_CSS = {
  error:   'var(--c-error)',
  warning: 'var(--c-warning)',
  success: 'var(--c-success)',
  info:    'var(--c-info)',
  primary: 'var(--c-primary)',
};

// Parse :::type{json}::: blocks out of markdown text
function parseRichBlocks(text) {
  const BLOCK_RE = /:::([a-z-]+)(\{[\s\S]*?\}):::/g;
  const segments = [];
  let last = 0, m;
  while ((m = BLOCK_RE.exec(text)) !== null) {
    if (m.index > last) segments.push({ kind: 'md', text: text.slice(last, m.index) });
    try { segments.push({ kind: m[1], data: JSON.parse(m[2]) }); }
    catch { segments.push({ kind: 'md', text: m[0] }); }
    last = m.index + m[0].length;
  }
  if (last < text.length) segments.push({ kind: 'md', text: text.slice(last) });
  return segments;
}

// ── Auto-detection: convert plain AI text into rich widgets ──────────────────

function autoDetectWidgets(text) {
  // Returns an array of segments (same format as parseRichBlocks)
  const segments = [];

  // 1. Extract explicit :::blocks::: first
  const withBlocks = parseRichBlocks(text);

  // 2. For each markdown segment, try to auto-detect patterns
  withBlocks.forEach(seg => {
    if (seg.kind !== 'md') { segments.push(seg); return; }
    const sub = autoDetectInText(seg.text);
    sub.forEach(s => segments.push(s));
  });

  return segments;
}

function autoDetectInText(text) {
  const out = [];

  // ── Pattern: Log snippets block ──────────────────────────────────────────
  // Detect blocks like "Timestamp: ...\nLogger: ...\nMessage: ..."
  const LOG_BLOCK_RE = /Timestamp:\s*([^\n]+)\s*\n\s*Logger:\s*([^\n]+)\s*\n\s*Message:\s*([^\n]+)/g;
  const logMatches = [];
  let lm;
  while ((lm = LOG_BLOCK_RE.exec(text)) !== null) logMatches.push(lm);

  if (logMatches.length > 0) {
    // Split text around the first log block start
    const firstIdx = logMatches[0].index;
    if (firstIdx > 0) {
      const before = text.slice(0, firstIdx).trim();
      if (before) out.push({ kind: 'md', text: before });
    }

    // Group identical messages as clusters
    const clusters = {};
    logMatches.forEach(m => {
      const msg = m[3].trim();
      if (!clusters[msg]) clusters[msg] = { ts: [], logger: m[2].trim(), level: 'ERROR' };
      clusters[msg].ts.push(m[1].trim());
    });
    out.push({ kind: 'log-cluster-group', data: Object.entries(clusters).map(([msg, v]) => ({ msg, ...v })) });

    const lastMatch = logMatches[logMatches.length - 1];
    const after = text.slice(lastMatch.index + lastMatch[0].length).trim();
    if (after) out.push(...autoDetectInText(after));
    return out;
  }

  // ── Pattern: Level distribution list (ERROR: 457, WARNING: 14, …) ───────
  const LEVEL_LIST_RE = /\b(ERROR|WARNING|INFO|DEBUG|CRITICAL|RAW)\s*:\s*(\d[\d,]*)/gi;
  const levelMatches = [...text.matchAll(LEVEL_LIST_RE)];
  if (levelMatches.length >= 2) {
    const labels = [], data = [], colors = [];
    const LEVEL_COLORS = { ERROR: 'error', CRITICAL: 'error', WARNING: 'warning', INFO: 'success', DEBUG: 'info', RAW: 'primary' };
    levelMatches.forEach(m => {
      labels.push(m[1].toUpperCase());
      data.push(parseInt(m[2].replace(/,/g, ''), 10));
      colors.push(LEVEL_COLORS[m[1].toUpperCase()] || 'primary');
    });

    // Extract surrounding prose
    const firstLevelIdx = text.search(LEVEL_LIST_RE);
    const lastLevelMatch = levelMatches[levelMatches.length - 1];
    const beforeLevel = text.slice(0, firstLevelIdx).trim();
    const afterLevel  = text.slice(lastLevelMatch.index + lastLevelMatch[0].length).trim();

    if (beforeLevel) out.push({ kind: 'md', text: beforeLevel });

    // Metric grid + donut chart
    const total = data.reduce((a, b) => a + b, 0);
    const metrics = labels.map((l, i) => ({ label: l, value: String(data[i]), color: colors[i], note: total ? `${Math.round(data[i]/total*100)}% of total` : '' }));
    out.push({ kind: 'metric-grid', data: { metrics } });
    out.push({ kind: 'chart', data: { type: 'doughnut', title: 'Log Level Distribution', labels, datasets: [{ label: 'Events', data, color: colors[0] }], multiColor: true } });

    if (afterLevel) out.push(...autoDetectInText(afterLevel));
    return out;
  }

  // ── Pattern: ratio statement (32.64:1 or 32.64 errors per warning) ───────
  const RATIO_RE = /(\d+(?:\.\d+)?)\s*(?:errors?)\s+(?:per|to|vs\.?)\s+(?:warning|warn)/i;
  const ratioM = text.match(RATIO_RE);
  if (ratioM) {
    const ratio = parseFloat(ratioM[1]);
    const errPct = Math.round((ratio / (ratio + 1)) * 100);
    out.push({ kind: 'ratio', data: { title: 'Error vs Warning Ratio', a: { label: 'Errors', pct: errPct, color: 'var(--c-error)' }, b: { label: 'Warnings', pct: 100 - errPct, color: 'var(--c-warning)' }, note: `${ratio.toFixed(1)} errors per warning` } });
    // Keep the rest as markdown
    const idx = text.search(RATIO_RE);
    const after = text.slice(idx + ratioM[0].length).trim();
    if (idx > 0) out.push({ kind: 'md', text: text.slice(0, idx) });
    if (after) out.push(...autoDetectInText(after));
    return out;
  }

  // ── Pattern: "Logger: X\nTotal Logs: N\nError Rate: N%" ──────────────────
  const LOGGER_BLOCK_RE = /Logger:\s*(\S+)\s*\n\s*Total Logs?:\s*(\d+)\s*\n\s*Error Rate:\s*([\d.]+%?)/gi;
  const loggerMatches = [...text.matchAll(LOGGER_BLOCK_RE)];
  if (loggerMatches.length >= 1) {
    const firstIdx = loggerMatches[0].index;
    const lastM    = loggerMatches[loggerMatches.length - 1];
    const before   = text.slice(0, firstIdx).trim();
    const after    = text.slice(lastM.index + lastM[0].length).trim();
    if (before) out.push({ kind: 'md', text: before });

    const rows = loggerMatches.map(m => ({ key: m[1], val: `${fmt(parseInt(m[2]))} events · ${m[3]} errors` }));
    out.push({ kind: 'stat-grid', data: { title: 'Logger Activity', rows } });

    if (after) out.push(...autoDetectInText(after));
    return out;
  }

  // No pattern matched — pass through as markdown
  out.push({ kind: 'md', text });
  return out;
}

// ── Segment → DOM ─────────────────────────────────────────────────────────────

function renderRichSegments(segments) {
  const container = document.createElement('div');
  container.className = 'rich-content';

  // Collect consecutive metrics into a grid
  let metricBuf = [];
  const flushMetrics = () => {
    if (!metricBuf.length) return;
    if (metricBuf.length === 1) {
      container.appendChild(renderMetric(metricBuf[0]));
    } else {
      const grid = document.createElement('div');
      grid.className = 'rich-metric-grid';
      metricBuf.forEach(d => grid.appendChild(renderMetric(d)));
      container.appendChild(grid);
    }
    metricBuf = [];
  };

  segments.forEach(seg => {
    if (seg.kind === 'metric') { metricBuf.push(seg.data); return; }
    if (seg.kind === 'metric-grid') {
      flushMetrics();
      const grid = document.createElement('div');
      grid.className = 'rich-metric-grid';
      (seg.data.metrics || []).forEach(d => grid.appendChild(renderMetric(d)));
      container.appendChild(grid);
      return;
    }
    flushMetrics();

    if (seg.kind === 'md') {
      if (seg.text.trim()) {
        const md = document.createElement('div');
        md.className = 'md-block';
        md.innerHTML = marked.parse(seg.text);
        container.appendChild(md);
      }
    } else {
      const el = renderRichComponent(seg.kind, seg.data);
      if (el) container.appendChild(el);
    }
  });
  flushMetrics();
  return container;
}

function renderRichComponent(kind, data) {
  switch (kind) {
    case 'log-ref':          return renderLogRef(data);
    case 'chart':            return renderInlineChart(data);
    case 'quiz':             return renderQuiz(data);
    case 'metric':           return renderMetric(data);
    case 'timeline':         return renderTimeline2(data);
    case 'stat-grid':        return renderStatGrid(data);
    case 'ratio':            return renderRatio(data);
    case 'log-cluster-group':return renderLogClusterGroup(data);
    case 'summary':          return renderSummaryBanner(data);
    default:                 return null;
  }
}

// ── Individual widget renderers ───────────────────────────────────────────────

function renderLogRef(d) {
  const div = document.createElement('div');
  div.className = `rich-log-ref level-border-${d.level || 'INFO'}`;
  div.innerHTML = `
    <div class="rich-log-ref-meta">
      <span class="level-badge level-${d.level || 'INFO'}">${d.level || 'INFO'}</span>
      <span class="rich-log-ts">${esc(d.ts || '')}</span>
      <span class="rich-log-logger">${esc(d.logger || '')}</span>
    </div>
    <div class="rich-log-msg">${esc(d.msg || '')}</div>`;
  div.title = 'Click to find in Timeline';
  div.addEventListener('click', () => {
    if (d.ts) {
      tlFilters.keyword = d.msg ? d.msg.slice(0, 40) : '';
      document.getElementById('tl-keyword').value = tlFilters.keyword;
      loadTimeline(1); navigateTo('timeline');
    }
  });
  return div;
}

function renderInlineChart(d) {
  const wrap = document.createElement('div');
  wrap.className = 'rich-chart-wrap';
  if (d.title) {
    const h = document.createElement('div');
    h.className = 'rich-chart-title';
    h.innerHTML = `<span class="material-symbols-rounded" style="font-size:0.85rem">${d.type === 'doughnut' || d.type === 'pie' ? 'donut_large' : d.type === 'line' ? 'show_chart' : 'bar_chart'}</span>${esc(d.title)}`;
    wrap.appendChild(h);
  }
  const canvas = document.createElement('canvas');
  canvas.height = d.type === 'doughnut' || d.type === 'pie' ? 180 : 160;
  wrap.appendChild(canvas);

  const c = getChartColors();
  const colorMap = { error: c.error, warning: c.warning, success: c.success, primary: c.primary, info: c.info };
  const PALETTE = [c.primary, c.error, c.warning, c.success, c.info, c.secondary, c.tertiary];

  const isDoughnut = d.type === 'doughnut' || d.type === 'pie';

  const datasets = (d.datasets || []).map((ds, di) => {
    const base = colorMap[ds.color] || c.primary;
    const bgColors = d.multiColor
      ? (ds.data || []).map((_, i) => (PALETTE[i % PALETTE.length]) + (isDoughnut ? 'ee' : 'bb'))
      : base + (isDoughnut ? 'ee' : 'bb');
    return {
      label: ds.label || '',
      data: ds.data || [],
      backgroundColor: bgColors,
      borderColor: d.multiColor ? (isDoughnut ? 'transparent' : undefined) : base,
      borderWidth: isDoughnut ? 0 : 1.5,
      borderRadius: isDoughnut ? 0 : 5,
      fill: d.type === 'line',
      tension: 0.35,
      pointRadius: d.type === 'line' ? 3 : 0,
    };
  });

  setTimeout(() => {
    const isXY = !isDoughnut;
    new Chart(canvas.getContext('2d'), {
      type: isDoughnut ? 'doughnut' : (d.type || 'bar'),
      data: { labels: d.labels || [], datasets },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: {
            display: isDoughnut || datasets.length > 1,
            position: isDoughnut ? 'right' : 'top',
            labels: { color: c.onSurface, boxWidth: isDoughnut ? 12 : 10, padding: 8, font: { size: 10 } },
          },
          tooltip: { callbacks: { label: ctx => ` ${ctx.label || ctx.dataset.label}: ${fmt(ctx.raw)}` } },
        },
        cutout: isDoughnut ? '60%' : undefined,
        scales: isXY ? {
          x: { ticks: { color: c.onSurface, font: { size: 9 }, maxRotation: 30 }, grid: { color: c.outline + '30' } },
          y: { ticks: { color: c.onSurface, font: { size: 9 } }, grid: { color: c.outline + '30' } },
        } : undefined,
      },
    });
  }, 0);
  return wrap;
}

function renderMetric(d) {
  const col = COLOR_MAP_CSS[d.color] || 'var(--c-primary)';
  const trendIcon = d.trend === 'up' ? 'trending_up' : d.trend === 'down' ? 'trending_down' : 'trending_flat';
  const bgIcon = { error: 'error', warning: 'warning', success: 'check_circle', info: 'info', primary: 'analytics' }[d.color] || 'analytics';
  const div = document.createElement('div');
  div.className = 'rich-metric';
  div.style.borderTopColor = col;
  div.innerHTML = `
    <div class="rich-metric-value" style="color:${col}">${esc(String(d.value || ''))}</div>
    <div class="rich-metric-label">${esc(d.label || '')}</div>
    ${d.note ? `<div class="rich-metric-note">${esc(d.note)}</div>` : ''}
    ${d.trend ? `<span class="material-symbols-rounded rich-metric-trend" style="color:${col}">${trendIcon}</span>` : ''}
    <span class="material-symbols-rounded rich-metric-bg-icon" style="color:${col}">${bgIcon}</span>`;
  return div;
}

function renderStatGrid(d) {
  const wrap = document.createElement('div');
  wrap.innerHTML = `<div class="rich-pattern-header"><span class="material-symbols-rounded" style="font-size:0.85rem">table_chart</span>${esc(d.title || 'Statistics')}</div>`;
  wrap.className = 'rich-pattern-table';
  const grid = document.createElement('div');
  grid.className = 'rich-stat-grid';
  grid.style.padding = '0.375rem';
  (d.rows || []).forEach(r => {
    const row = document.createElement('div');
    row.className = 'rich-stat-row';
    row.innerHTML = `<span class="rich-stat-key">${esc(r.key)}</span><span class="rich-stat-val">${esc(String(r.val))}</span>`;
    grid.appendChild(row);
  });
  wrap.appendChild(grid);
  return wrap;
}

function renderRatio(d) {
  const div = document.createElement('div');
  div.className = 'rich-ratio';
  div.innerHTML = `
    <div class="rich-ratio-header">
      <span class="rich-ratio-title">${esc(d.title || '')}</span>
      <span class="rich-ratio-value">${esc(d.note || '')}</span>
    </div>
    <div class="rich-ratio-track">
      <div class="rich-ratio-fill" style="width:${d.a?.pct || 0}%;background:${d.a?.color || 'var(--c-error)'}"></div>
      <div class="rich-ratio-fill" style="width:${d.b?.pct || 0}%;background:${d.b?.color || 'var(--c-warning)'}"></div>
    </div>
    <div class="rich-ratio-labels">
      <span>${esc(d.a?.label || '')} ${d.a?.pct || 0}%</span>
      <span>${d.b?.pct || 0}% ${esc(d.b?.label || '')}</span>
    </div>`;
  return div;
}

function renderLogClusterGroup(clusters) {
  const wrap = document.createElement('div');
  wrap.style.display = 'flex';
  wrap.style.flexDirection = 'column';
  wrap.style.gap = '0.375rem';
  (clusters || []).forEach(c => {
    const div = document.createElement('div');
    const level = c.level || 'ERROR';
    const countClass = level === 'WARNING' ? 'warn' : level === 'INFO' ? 'info' : '';
    div.className = `rich-log-cluster level-border-${level}`;
    const header = document.createElement('div');
    header.className = 'rich-cluster-header';
    header.innerHTML = `
      <span class="level-badge level-${level}">${level}</span>
      <span class="rich-cluster-msg" title="${esc(c.msg)}">${esc(c.msg)}</span>
      <span class="rich-cluster-count ${countClass}">×${c.ts.length}</span>`;
    const body = document.createElement('div');
    body.className = 'rich-cluster-body';
    c.ts.slice(0, 8).forEach(ts => {
      const ev = document.createElement('div');
      ev.className = 'rich-cluster-event';
      ev.innerHTML = `<span class="rich-cluster-ts">${esc(ts)}</span><span class="rich-cluster-text">${esc(c.msg.slice(0, 80))}</span>`;
      body.appendChild(ev);
    });
    if (c.ts.length > 8) {
      const more = document.createElement('div');
      more.className = 'rich-cluster-event';
      more.innerHTML = `<span class="rich-cluster-ts" style="color:var(--c-primary)">+${c.ts.length - 8} more occurrences</span>`;
      body.appendChild(more);
    }
    header.addEventListener('click', () => body.classList.toggle('open'));
    div.appendChild(header); div.appendChild(body);
    wrap.appendChild(div);
  });
  return wrap;
}

function renderSummaryBanner(d) {
  const div = document.createElement('div');
  div.className = 'rich-summary-banner';
  div.innerHTML = `<span class="material-symbols-rounded">summarize</span><span>${esc(d.text || '')}</span>`;
  return div;
}

function renderQuiz(d) {
  const div = document.createElement('div');
  div.className = 'rich-quiz';
  div.innerHTML = `<div class="rich-quiz-q"><span class="material-symbols-rounded">quiz</span>${esc(d.question || '')}</div>`;
  const opts = document.createElement('div');
  opts.className = 'rich-quiz-opts';
  (d.options || []).forEach((opt, i) => {
    const btn = document.createElement('button');
    btn.className = 'rich-quiz-opt';
    btn.innerHTML = `<span class="opt-letter">${String.fromCharCode(65 + i)}</span><span>${esc(opt)}</span>`;
    btn.addEventListener('click', () => {
      if (div.dataset.answered) return;
      div.dataset.answered = '1';
      opts.querySelectorAll('.rich-quiz-opt').forEach((b, j) => {
        b.disabled = true;
        if (j === d.answer) b.classList.add('correct');
        else if (j === i && i !== d.answer) b.classList.add('wrong');
      });
      if (d.explanation) {
        const exp = document.createElement('div');
        exp.className = 'rich-quiz-exp';
        exp.innerHTML = `<span class="material-symbols-rounded">info</span>${esc(d.explanation)}`;
        div.appendChild(exp);
      }
    });
    opts.appendChild(btn);
  });
  div.appendChild(opts);
  return div;
}

function renderTimeline2(d) {
  const div = document.createElement('div');
  div.className = 'rich-timeline';
  if (d.title) {
    const h = document.createElement('div');
    h.className = 'rich-timeline-title';
    h.innerHTML = `<span class="material-symbols-rounded">timeline</span>${esc(d.title)}`;
    div.appendChild(h);
  }
  const list = document.createElement('div');
  list.className = 'rich-timeline-list';
  (d.events || []).forEach(ev => {
    const item = document.createElement('div');
    item.className = 'rich-tl-item';
    item.innerHTML = `
      <div class="rich-tl-dot level-dot-${ev.level || 'INFO'}"></div>
      <div class="rich-tl-body">
        <div class="rich-tl-meta">
          <span class="level-badge level-${ev.level || 'INFO'}">${ev.level || 'INFO'}</span>
          <span class="rich-log-ts">${esc(ev.ts || '')}</span>
        </div>
        <div class="rich-tl-msg">${esc(ev.msg || '')}</div>
      </div>`;
    list.appendChild(item);
  });
  div.appendChild(list);
  return div;
}

// ── Agent Panel ───────────────────────────────────────────────────────────────

const TOOL_ICONS = {
  search_logs:  'manage_search',
  get_stats:    'bar_chart',
  get_timeline: 'view_timeline',
  get_samples:  'receipt_long',
};

function agentPanelSetActive(active) {
  const dot = document.getElementById('agent-dot');
  if (dot) dot.classList.toggle('active', active);
}

function agentPanelClear() {
  const steps = document.getElementById('agent-steps');
  if (steps) steps.innerHTML = '';
}

function agentPanelAddStep(type, data) {
  const steps = document.getElementById('agent-steps');
  if (!steps) return;

  // Remove idle message
  steps.querySelector('.agent-idle-msg')?.remove();

  const item = document.createElement('div');
  item.className = 'agent-step';

  if (type === 'thinking') {
    item.className = 'agent-step agent-thinking';
    item.innerHTML = `
      <span class="material-symbols-rounded agent-step-icon">psychology</span>
      <div class="agent-step-body">
        <div class="agent-step-label">Thinking…</div>
        <div class="thinking-dot"><span></span><span></span><span></span></div>
      </div>`;
  } else if (type === 'tool_call') {
    const icon = TOOL_ICONS[data.name] || 'build';
    let argPreview = '';
    try {
      const args = typeof data.input === 'string' ? JSON.parse(data.input) : data.input;
      argPreview = args.query || args.fields?.join(', ') || args.logger || args.keyword || JSON.stringify(args).slice(0, 60);
    } catch { argPreview = String(data.input || '').slice(0, 60); }
    item.innerHTML = `
      <span class="material-symbols-rounded agent-step-icon tool">${icon}</span>
      <div class="agent-step-body">
        <div class="agent-step-label"><strong>${data.name}</strong></div>
        ${argPreview ? `<div class="agent-step-detail">${esc(argPreview)}</div>` : ''}
      </div>`;
    item.id = 'agent-tool-' + data.name + '-' + Date.now();
  } else if (type === 'tool_result') {
    item.className = 'agent-step agent-result';
    let preview = '';
    try {
      const res = JSON.parse(data.content);
      if (res.results) preview = `${res.results.length} results`;
      else if (res.total !== undefined) preview = `${res.total} events`;
      else if (res.count !== undefined) preview = `${res.count} samples`;
      else if (res.error) preview = '⚠ ' + res.error;
      else preview = data.content.slice(0, 80);
    } catch { preview = data.content.slice(0, 80); }
    item.innerHTML = `
      <span class="material-symbols-rounded agent-step-icon result">check_circle</span>
      <div class="agent-step-body">
        <div class="agent-step-label">${esc(data.name)} → done</div>
        <div class="agent-step-detail">${esc(preview)}</div>
      </div>`;
  } else if (type === 'done') {
    item.className = 'agent-step agent-done';
    item.innerHTML = `
      <span class="material-symbols-rounded agent-step-icon done">task_alt</span>
      <div class="agent-step-body"><div class="agent-step-label">Response complete</div></div>`;
  }

  steps.appendChild(item);
  steps.scrollTop = steps.scrollHeight;
}

// ── Chat ──────────────────────────────────────────────────────────────────────
function populateSuggestions(summary) {
  const char = summary.characterization || {};
  const tips = [...SUGGESTIONS];
  if (char.key_event_types && char.key_event_types.length) {
    tips.unshift(`What are the most common ${char.key_event_types[0]} events?`);
  }
  const el = document.getElementById('suggestion-chips');
  el.innerHTML = '';
  tips.slice(0, 5).forEach(t => {
    const btn = document.createElement('button');
    btn.className = 'suggestion-chip';
    btn.textContent = t;
    btn.addEventListener('click', () => sendMessage(t));
    el.appendChild(btn);
  });
}

function openWebSocket() {
  if (chatWs && chatWs.readyState < 2) chatWs.close();
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  chatWs = new WebSocket(`${proto}://${location.host}/ws/chat`);

  chatWs.onopen    = () => {};
  chatWs.onmessage = e => handleWsMessage(JSON.parse(e.data));
  chatWs.onclose   = () => {
    if (wsStreaming) endStreamBubble();
    wsStreaming = false;
    setTimeout(openWebSocket, 2000);
  };
  chatWs.onerror   = () => {};
}

function handleWsMessage(msg) {
  switch (msg.type) {
    case 'token':
      removeThinking();
      ensureStreamBubble();
      bubbleText += msg.content;
      // During streaming: hide partial :::component::: blocks so the user
      // sees clean markdown; full rich render happens in endStreamBubble().
      currentBubble.innerHTML = marked.parse(
        bubbleText.replace(/:::[\s\S]*?(?::::)?/g, m => m.endsWith(':::') ? '' : '')
      );
      scrollChat();
      break;
    case 'tool_call':
      removeThinking();
      agentPanelAddStep('tool_call', { name: msg.name, input: msg.input });
      break;
    case 'tool_result':
      agentPanelAddStep('tool_result', { name: msg.name, content: msg.content });
      // Add a thinking step for next iteration
      agentPanelAddStep('thinking', {});
      break;
    case 'status':
      appendStatus(msg.content);
      break;
    case 'done':
      endStreamBubble();
      agentPanelAddStep('done', {});
      agentPanelSetActive(false);
      enableInput();
      break;
    case 'error':
      endStreamBubble();
      agentPanelSetActive(false);
      appendAssistantMsg('⚠ ' + msg.content, true);
      enableInput();
      break;
  }
}

function ensureStreamBubble() {
  if (currentBubble) return;
  document.getElementById('chat-welcome')?.remove();
  const wrap = document.createElement('div');
  wrap.className = 'chat-msg assistant';
  wrap.innerHTML = `
    <div class="avatar"><span class="material-symbols-rounded">smart_toy</span></div>
    <div class="bubble"></div>`;
  document.getElementById('chat-messages').appendChild(wrap);
  currentBubble = wrap.querySelector('.bubble');
  bubbleText = '';
}

function endStreamBubble() {
  if (currentBubble && bubbleText) {
    // Auto-detect widgets from plain text + parse explicit :::blocks:::
    const segments = autoDetectWidgets(bubbleText);
    currentBubble.innerHTML = '';
    currentBubble.appendChild(renderRichSegments(segments));
  }
  currentBubble = null;
  bubbleText = '';
  wsStreaming = false;
  scrollChat();
}

function appendAssistantMsg(text, isError = false) {
  document.getElementById('chat-welcome')?.remove();
  const wrap = document.createElement('div');
  wrap.className = 'chat-msg assistant';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  if (isError) bubble.style.color = 'var(--c-error)';
  bubble.innerHTML = marked.parse(text);
  wrap.innerHTML = `<div class="avatar"><span class="material-symbols-rounded">smart_toy</span></div>`;
  wrap.appendChild(bubble);
  document.getElementById('chat-messages').appendChild(wrap);
  scrollChat();
}

function appendUserMsg(text) {
  document.getElementById('chat-welcome')?.remove();
  const wrap = document.createElement('div');
  wrap.className = 'chat-msg user';
  wrap.innerHTML = `
    <div class="bubble">${esc(text)}</div>
    <div class="avatar"><span class="material-symbols-rounded">person</span></div>`;
  document.getElementById('chat-messages').appendChild(wrap);
  scrollChat();
}

function appendStatus(text) {
  const container = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className = 'tool-indicator';
  div.innerHTML = `<span class="material-symbols-rounded">info</span><span>${esc(text)}</span>`;
  container.appendChild(div);
  scrollChat();
}

function appendThinking() {
  document.getElementById('chat-welcome')?.remove();
  const wrap = document.createElement('div');
  wrap.className = 'chat-msg assistant';
  wrap.id = 'thinking-msg';
  wrap.innerHTML = `
    <div class="avatar"><span class="material-symbols-rounded">smart_toy</span></div>
    <div class="bubble"><div class="thinking-dot"><span></span><span></span><span></span></div></div>`;
  document.getElementById('chat-messages').appendChild(wrap);
  scrollChat();
}

function removeThinking() {
  document.getElementById('thinking-msg')?.remove();
}

function scrollChat() {
  const c = document.getElementById('chat-messages');
  c.scrollTop = c.scrollHeight;
}

function disableInput() {
  document.getElementById('chat-input').disabled = true;
  document.getElementById('send-btn').disabled = true;
}
function enableInput() {
  document.getElementById('chat-input').disabled = false;
  document.getElementById('send-btn').disabled = false;
  document.getElementById('chat-input').focus();
}

function sendMessage(text) {
  const question = (text || document.getElementById('chat-input').value).trim();
  if (!question || wsStreaming) return;

  appendUserMsg(question);
  document.getElementById('chat-input').value = '';
  document.getElementById('chat-input').style.height = '';
  disableInput();
  wsStreaming = true;
  currentBubble = null;
  bubbleText = '';

  // Reset agent panel for new turn
  agentPanelClear();
  agentPanelSetActive(true);
  agentPanelAddStep('thinking', {});

  appendThinking();

  if (!chatWs || chatWs.readyState !== 1) {
    appendAssistantMsg('WebSocket not connected. Reconnecting…', true);
    enableInput(); wsStreaming = false;
    agentPanelSetActive(false);
    openWebSocket();
    return;
  }
  const payload = { question, mode: chatMode };
  if (selectedModel) payload.model = selectedModel;
  chatWs.send(JSON.stringify(payload));
}

// ── Model selector ────────────────────────────────────────────────────────────
const PROVIDER_LABELS = { groq: 'Groq', openrouter: 'OpenRouter', gemini: 'Gemini' };

async function loadModels() {
  try {
    const data = await fetch('/api/models').then(r => r.json());
    const models = data.models || [];
    const sel = document.getElementById('model-select');
    sel.innerHTML = '';

    // Group by provider
    const groups = {};
    models.forEach(m => {
      if (!groups[m.provider]) groups[m.provider] = [];
      groups[m.provider].push(m);
    });

    Object.entries(groups).forEach(([provider, list]) => {
      const og = document.createElement('optgroup');
      og.label = PROVIDER_LABELS[provider] || provider;
      list.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.id;
        opt.textContent = m.name + (!m.supports_tools ? ' (no tools)' : '');
        og.appendChild(opt);
      });
      sel.appendChild(og);
    });

    // Restore saved selection
    const saved = localStorage.getItem('logai_model');
    const allIds = models.map(m => m.id);
    if (saved && allIds.includes(saved)) {
      sel.value = saved;
    } else {
      sel.value = allIds[0] || '';
    }
    selectedModel = sel.value || null;
    updateProviderBadge(models.find(m => m.id === sel.value)?.provider);

    sel.addEventListener('change', () => {
      selectedModel = sel.value || null;
      localStorage.setItem('logai_model', sel.value);
      const info = models.find(m => m.id === sel.value);
      updateProviderBadge(info?.provider);
    });
  } catch (e) {
    console.warn('Failed to load models', e);
  }
}

function updateProviderBadge(provider) {
  const badge = document.getElementById('provider-badge');
  if (!badge) return;
  badge.textContent = PROVIDER_LABELS[provider] || provider || '';
  badge.dataset.provider = provider || '';
}

function initChat() {
  const input   = document.getElementById('chat-input');
  const sendBtn = document.getElementById('send-btn');

  input.addEventListener('input', () => {
    sendBtn.disabled = !input.value.trim() || wsStreaming;
    input.style.height = '';
    input.style.height = Math.min(input.scrollHeight, 150) + 'px';
  });
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
  sendBtn.addEventListener('click', () => sendMessage());

  // Mode toggle
  document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      chatMode = btn.dataset.mode;
      document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      // Show/hide agent panel based on mode
      document.getElementById('agent-panel').style.display = chatMode === 'agent' ? '' : 'none';
    });
  });

  // Clear history
  document.getElementById('clear-history-btn').addEventListener('click', async () => {
    await fetch('/api/history', { method: 'DELETE' });
    document.getElementById('chat-messages').innerHTML = '';
    agentPanelClear();
    currentBubble = null; bubbleText = ''; wsStreaming = false;
    enableInput();
  });
}

// ── Sessions ──────────────────────────────────────────────────────────────────
function showToast(msg) {
  const t = document.createElement('div');
  t.className = 'save-toast';
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 2600);
}

function collectChatMessages() {
  const msgs = [];
  document.querySelectorAll('#chat-messages .chat-msg').forEach(wrap => {
    const isUser = wrap.classList.contains('user');
    const bubble = wrap.querySelector('.bubble');
    if (bubble) msgs.push({ role: isUser ? 'user' : 'assistant', html: bubble.innerHTML });
  });
  return msgs;
}

function restoreChatMessages(messages) {
  const container = document.getElementById('chat-messages');
  document.getElementById('chat-welcome')?.remove();
  messages.forEach(m => {
    const wrap = document.createElement('div');
    wrap.className = 'chat-msg ' + (m.role === 'user' ? 'user' : 'assistant');
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.innerHTML = m.html || '';
    if (m.role === 'user') {
      wrap.appendChild(bubble);
      wrap.insertAdjacentHTML('beforeend', `<div class="avatar"><span class="material-symbols-rounded">person</span></div>`);
    } else {
      wrap.innerHTML = `<div class="avatar"><span class="material-symbols-rounded">smart_toy</span></div>`;
      wrap.appendChild(bubble);
    }
    container.appendChild(wrap);
  });
  scrollChat();
}

async function saveCurrentSession() {
  const messages = collectChatMessages();
  if (!messages.length) { showToast('Nothing to save — start a chat first'); return; }
  const title = document.getElementById('dash-filename')?.textContent || 'Session';
  const res = await fetch('/api/sessions/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title, messages }),
  });
  if (res.ok) {
    showToast('Session saved ✓');
    renderSessionsList();
  }
}

async function renderSessionsList() {
  const data = await fetch('/api/sessions').then(r => r.json()).catch(() => ({ sessions: [] }));
  const sessions = data.sessions || [];
  const list  = document.getElementById('sessions-list');
  const empty = document.getElementById('sessions-empty');
  if (!list) return;
  // Remove old items but keep empty placeholder
  list.querySelectorAll('.session-item').forEach(el => el.remove());

  if (!sessions.length) {
    if (empty) empty.style.display = '';
    return;
  }
  if (empty) empty.style.display = 'none';

  sessions.forEach(s => {
    const item = document.createElement('div');
    item.className = 'session-item';
    const savedAt = s.saved_at ? new Date(s.saved_at).toLocaleString([], { month:'short', day:'numeric', hour:'2-digit', minute:'2-digit' }) : '';
    item.innerHTML = `
      <div class="session-icon"><span class="material-symbols-rounded">description</span></div>
      <div class="session-body">
        <div class="session-title">${esc(s.summary_title || s.filename || 'Session')}</div>
        <div class="session-meta">
          <span>${savedAt}</span>
          <span>${s.messages} message${s.messages !== 1 ? 's' : ''}</span>
        </div>
      </div>
      <div class="session-actions">
        <button class="session-del-btn" title="Delete"><span class="material-symbols-rounded">delete</span></button>
      </div>`;

    item.querySelector('.session-body').addEventListener('click', async () => {
      const full = await fetch(`/api/sessions/${s.id}`).then(r => r.json());
      if (full.messages) {
        document.getElementById('chat-messages').innerHTML = '';
        restoreChatMessages(full.messages);
        navigateTo('chat');
        closeSessions();
        showToast(`Loaded: ${s.summary_title || s.filename}`);
      }
    });
    item.querySelector('.session-del-btn').addEventListener('click', async (e) => {
      e.stopPropagation();
      await fetch(`/api/sessions/${s.id}`, { method: 'DELETE' });
      item.remove();
      if (!list.querySelectorAll('.session-item').length && empty) empty.style.display = '';
    });
    list.appendChild(item);
  });
}

function closeSessions() {
  document.getElementById('sessions-drawer')?.classList.remove('open');
  document.getElementById('drawer-overlay')?.classList.remove('active');
}

function initSessions() {
  const drawer = document.getElementById('sessions-drawer');
  const overlay = document.getElementById('drawer-overlay');

  document.getElementById('sessions-btn')?.addEventListener('click', () => {
    drawer?.classList.add('open');
    overlay?.classList.add('active');
    renderSessionsList();
  });
  document.getElementById('close-sessions')?.addEventListener('click', closeSessions);
  document.getElementById('save-session-btn')?.addEventListener('click', saveCurrentSession);
  overlay?.addEventListener('click', () => {
    document.getElementById('settings-drawer')?.classList.remove('open');
    closeSessions();
    overlay.classList.remove('active');
  });
}

// ── Settings drawer ───────────────────────────────────────────────────────────
function initSettings() {
  const drawer  = document.getElementById('settings-drawer');
  const overlay = document.getElementById('drawer-overlay');
  const open    = () => { drawer.classList.add('open'); overlay.classList.add('active'); };
  const close   = () => { drawer.classList.remove('open'); overlay.classList.remove('active'); };

  document.getElementById('settings-btn').addEventListener('click', open);
  document.getElementById('close-settings').addEventListener('click', close);
  // overlay click handled centrally in initSessions

  document.querySelectorAll('.seg-btn[data-setting="theme"]').forEach(b =>
    b.addEventListener('click', () => {
      document.querySelectorAll('.seg-btn[data-setting="theme"]').forEach(x => x.classList.remove('active'));
      b.classList.add('active'); applyTheme(b.dataset.value);
    }));

  document.querySelectorAll('.seg-btn[data-setting="density"]').forEach(b =>
    b.addEventListener('click', () => {
      document.querySelectorAll('.seg-btn[data-setting="density"]').forEach(x => x.classList.remove('active'));
      b.classList.add('active'); applyDensity(b.dataset.value);
    }));

  document.querySelectorAll('.palette-dot').forEach(d =>
    d.addEventListener('click', () => {
      document.querySelectorAll('.palette-dot').forEach(x => x.classList.remove('active'));
      d.classList.add('active'); applyPalette(d.dataset.palette);
    }));

  const fsSlider = document.getElementById('font-scale');
  fsSlider.addEventListener('input', () => {
    const v = parseFloat(fsSlider.value);
    document.getElementById('font-scale-val').textContent = Math.round(v * 100) + '%';
    applyFontScale(v);
  });
}

// ── Utilities ─────────────────────────────────────────────────────────────────
function fmt(n) { return Number(n).toLocaleString(); }
function esc(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── Report page ───────────────────────────────────────────────────────────────

let reportCurrentId  = null;
let reportEventSource = null;
let reportSections   = {};
let reportGenRunning = false;

const RPT_SECTIONS = [
  { id: 'cover',                  title: 'Cover',                    icon: 'description',    isStatic: true },
  { id: 'table_of_contents',      title: 'Table of Contents',        icon: 'toc',            isStatic: true },
  { id: 'abstract',               title: 'Abstract',                 icon: 'article' },
  { id: 'executive_summary',      title: 'Executive Summary',        icon: 'assignment' },
  { id: 'system_overview',        title: 'System Overview',          icon: 'hub' },
  { id: 'statistical_overview',   title: 'Statistical Overview',     icon: 'bar_chart',      isStatic: true },
  { id: 'error_analysis',         title: 'Error Analysis',           icon: 'error',          isError: true },
  { id: 'error_bursts',           title: 'Error Burst Analysis',     icon: 'bolt',           isError: true },
  { id: 'performance_throughput', title: 'Performance & Throughput', icon: 'speed' },
  { id: 'component_health',       title: 'Component Health',         icon: 'developer_board' },
  { id: 'pattern_analysis',       title: 'Pattern Analysis',         icon: 'pattern' },
  { id: 'entity_analysis',        title: 'Entity Analysis',          icon: 'fingerprint' },
  { id: 'pain_points',            title: 'Pain Points',              icon: 'location_on',    isPain: true },
  { id: 'recommendations',        title: 'Recommendations',          icon: 'tips_and_updates' },
  { id: 'appendix',               title: 'Appendix',                 icon: 'table_view',     isStatic: true },
];

function initReport() {
  document.getElementById('report-generate-btn').addEventListener('click', startReportGeneration);
  document.getElementById('report-download-btn').addEventListener('click', downloadReport);
  buildSectionTracker();
}

/* Build the left-sidebar section checklist */
function buildSectionTracker() {
  const tracker = document.getElementById('report-section-tracker');
  if (!tracker) return;
  tracker.innerHTML = RPT_SECTIONS.map(s => `
    <div class="rtrack-item" id="rtrack-${s.id}">
      <span class="rtrack-icon material-symbols-rounded">${s.icon}</span>
      <span class="rtrack-label">${s.title}</span>
      <span class="rtrack-status" id="rtst-${s.id}">
        <span class="material-symbols-rounded rtrack-pending">radio_button_unchecked</span>
      </span>
    </div>`).join('');
}

function startReportGeneration() {
  if (reportGenRunning) return;
  reportGenRunning = true;
  reportSections   = {};
  reportCurrentId  = null;

  // Show document, hide landing
  document.getElementById('report-landing').style.display = 'none';
  document.getElementById('report-document').style.display = '';

  // Show overall progress in sidebar
  document.getElementById('report-overall-progress').style.display = '';

  // Disable generate, reset tracker
  const genBtn = document.getElementById('report-generate-btn');
  genBtn.disabled = true;
  genBtn.innerHTML = '<span class="material-symbols-rounded spin">autorenew</span><span>Generating…</span>';
  document.getElementById('report-download-btn').disabled = true;

  // Reset all tracker items
  RPT_SECTIONS.forEach(s => setTrackerStatus(s.id, 'pending'));

  // Build document skeleton
  buildDocumentSkeleton();

  // Open SSE (GET endpoint)
  reportEventSource = new EventSource('/api/report/generate');
  reportEventSource.onmessage = onReportSSE;
  reportEventSource.onerror   = () => onReportError();
}

function onReportSSE(event) {
  let msg;
  try { msg = JSON.parse(event.data); } catch { return; }

  if (msg.type === 'start') {
    setOverallProgress(3, 'Generating report…');
    setCurrentSectionLabel('Starting…');
  }

  if (msg.type === 'section_start') {
    const sec  = RPT_SECTIONS.find(s => s.id === msg.section);
    const pct  = Math.round(((msg.index || 0) / RPT_SECTIONS.length) * 88) + 4;
    setOverallProgress(pct, sec ? sec.title : msg.section);
    setCurrentSectionLabel(`${sec ? sec.icon + '  ' : ''}${sec ? sec.title : msg.section}`);
    setTrackerStatus(msg.section, 'active');
    // Scroll tracker item into view
    document.getElementById(`rtrack-${msg.section}`)?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  if (msg.type === 'section_content') {
    reportSections[msg.section] = (reportSections[msg.section] || '') + msg.content;
    renderSection(msg.section, reportSections[msg.section]);
  }

  if (msg.type === 'section_done') {
    setTrackerStatus(msg.section, 'done');
    // Scroll the rendered section into view (skip cover/toc)
    if (!['cover','table_of_contents'].includes(msg.section)) {
      document.getElementById(`rsec-${msg.section}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  if (msg.type === 'complete') {
    setOverallProgress(95, 'Saving report…');
    setCurrentSectionLabel('Saving…');
  }

  if (msg.type === 'saved') {
    reportCurrentId = msg.report_id;
    setOverallProgress(100, 'Report complete!');
    setCurrentSectionLabel('Done ✓');
    setTimeout(() => finishReportGeneration(true), 1000);
  }
}

function onReportError() {
  console.error('Report SSE connection error');
  setOverallProgress(0, 'Connection error — is the server running?');
  setCurrentSectionLabel('Error');
  finishReportGeneration(false);
}

function buildDocumentSkeleton() {
  // Reset TOC
  document.getElementById('report-toc-list').innerHTML =
    RPT_SECTIONS.filter(s => !['cover','table_of_contents'].includes(s.id))
      .map((s, i) => `<li><a href="#rsec-${s.id}">${i+1}. ${s.title}</a></li>`).join('');

  // Build LLM text section placeholders
  const llmContainer = document.getElementById('report-llm-sections');
  llmContainer.innerHTML = '';
  let num = 1;
  RPT_SECTIONS.forEach(s => {
    if (['cover','table_of_contents','statistical_overview','appendix'].includes(s.id)) return;
    const el = document.createElement('div');
    el.className = 'report-section' + (s.isPain ? ' report-pain-section' : '');
    el.id = `rsec-${s.id}`;
    el.innerHTML = `
      <div class="report-sec-header">
        <span class="material-symbols-rounded">${s.icon}</span>
        <h2 class="report-sec-title">${num}. ${s.title}</h2>
        <span class="report-sec-badge pending" id="rss-${s.id}">Pending</span>
      </div>
      <div class="report-sec-body" id="rsb-${s.id}">
        <div class="report-skeleton">
          ${['90%','70%','82%','55%','75%'].map(w => `<div class="skeleton-line" style="width:${w}"></div>`).join('')}
        </div>
      </div>`;
    llmContainer.appendChild(el);
    num++;
  });
}

/* Render a section's content into the document */
function renderSection(secId, content) {
  // Update badge
  const badge = document.getElementById(`rss-${secId}`);
  if (badge) { badge.className = 'report-sec-badge active'; badge.textContent = 'Writing…'; }

  if (secId === 'cover') {
    try {
      const d = JSON.parse(content);
      document.getElementById('rc-filename').textContent = d.filename || '—';
      document.getElementById('rc-system').textContent   = d.system_description || d.log_type || '—';
      document.getElementById('rc-meta').textContent     = `Generated ${d.generated_at || ''}  ·  Format: ${d.format || 'Unknown'}`;
      document.getElementById('rc-stats').innerHTML = `
        <div class="rcover-stat"><div class="rcover-val">${fmt(d.total_events||0)}</div><div class="rcover-lbl">Total Events</div></div>
        <div class="rcover-stat"><div class="rcover-val rcover-err">${fmt(d.error_count||0)}</div><div class="rcover-lbl">Errors</div></div>
        <div class="rcover-stat"><div class="rcover-val rcover-err">${(d.error_rate_pct||0).toFixed(1)}%</div><div class="rcover-lbl">Error Rate</div></div>
        <div class="rcover-stat rcover-wide">
          <div class="rcover-val" style="font-size:0.95rem">${d.time_range_first||''}  →  ${d.time_range_last||''}</div>
          <div class="rcover-lbl">Time Range · ${(d.span_hours||0).toFixed(1)} hours</div>
        </div>`;
    } catch {}
    return;
  }

  if (secId === 'table_of_contents') return;

  if (secId === 'statistical_overview') {
    try { renderReportCharts(JSON.parse(content)); } catch {}
    const b2 = document.getElementById('rss-statistical_overview');
    if (b2) b2.style.display = '';
    return;
  }

  if (secId === 'appendix') {
    try { renderReportAppendix(JSON.parse(content)); } catch {}
    const b2 = document.getElementById('rss-appendix');
    if (b2) b2.style.display = '';
    return;
  }

  // LLM markdown sections
  const body = document.getElementById(`rsb-${secId}`);
  if (!body) return;
  body.innerHTML = `<div class="report-md-content">${marked.parse(content)}</div>`;
}

/* Charts for statistical overview */
function renderReportCharts(data) {
  const lc  = data.level_counts || {};
  const LEVEL_COLORS = { DEBUG:'#6B7280', INFO:'#3B82F6', WARNING:'#F59E0B', ERROR:'#EF4444', CRITICAL:'#7C3AED', RAW:'#9CA3AF' };
  const isDark  = document.documentElement.dataset.theme !== 'light';
  const tickClr = isDark ? 'rgba(255,255,255,0.55)' : 'rgba(0,0,0,0.55)';
  const gridClr = isDark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.07)';
  const scaleOpts = { ticks: { color: tickClr }, grid: { color: gridClr } };
  const legendOpts = { labels: { color: tickClr, boxWidth: 12, font: { size: 11 } } };
  const mk = id => document.getElementById(id)?.getContext('2d');

  // Destroy existing charts to avoid duplicates on regenerate
  ['rpt-chart-levels','rpt-chart-loggers','rpt-chart-hours','rpt-chart-errors-hour'].forEach(id => {
    const el = document.getElementById(id);
    if (el && Chart.getChart(el)) Chart.getChart(el).destroy();
  });

  // Level doughnut
  const lvlCtx = mk('rpt-chart-levels');
  if (lvlCtx) new Chart(lvlCtx, {
    type: 'doughnut',
    data: { labels: Object.keys(lc), datasets: [{ data: Object.values(lc), backgroundColor: Object.keys(lc).map(l => LEVEL_COLORS[l]||'#9CA3AF'), borderWidth: 0 }] },
    options: { cutout: '62%', plugins: { legend: legendOpts } }
  });

  // Loggers bar
  const tl  = (data.top_loggers || []).slice(0, 10);
  const logCtx = mk('rpt-chart-loggers');
  if (logCtx) new Chart(logCtx, {
    type: 'bar',
    data: {
      labels: tl.map(l => l.logger),
      datasets: [
        { label: 'Total', data: tl.map(l => l.count), backgroundColor: 'rgba(129,140,248,0.8)' },
        { label: 'Errors', data: tl.map(l => Math.round((l.error_rate||0)*l.count)), backgroundColor: 'rgba(248,113,113,0.8)' },
      ]
    },
    options: { plugins: { legend: legendOpts }, scales: { x: { ...scaleOpts, ticks: { ...scaleOpts.ticks, maxRotation: 45 } }, y: scaleOpts } }
  });

  // Events/hour
  const eph  = data.events_per_hour || {};
  const erph = data.errors_per_hour || {};
  const hrKeys = [...new Set([...Object.keys(eph),...Object.keys(erph)])].sort();
  const hrCtx = mk('rpt-chart-hours');
  if (hrCtx) new Chart(hrCtx, {
    type: 'bar',
    data: { labels: hrKeys, datasets: [{ label: 'Events/hr', data: hrKeys.map(h => eph[h]||0), backgroundColor: 'rgba(129,140,248,0.7)' }] },
    options: { plugins: { legend: legendOpts }, scales: { x: { ...scaleOpts, ticks: { ...scaleOpts.ticks, maxRotation: 60 } }, y: scaleOpts } }
  });

  // Errors/hour line
  const ehCtx = mk('rpt-chart-errors-hour');
  if (ehCtx) new Chart(ehCtx, {
    type: 'line',
    data: { labels: hrKeys, datasets: [{ label: 'Errors/hr', data: hrKeys.map(h => erph[h]||0), borderColor:'#f87171', backgroundColor:'rgba(248,113,113,0.15)', fill:true, tension:0.3 }] },
    options: { plugins: { legend: legendOpts }, scales: { x: { ...scaleOpts, ticks: { ...scaleOpts.ticks, maxRotation: 60 } }, y: scaleOpts } }
  });

  // Level table
  const total = Object.values(lc).reduce((a,b)=>a+b,0)||1;
  const tbody = document.getElementById('rpt-level-tbody');
  if (tbody) tbody.innerHTML = Object.entries(lc).map(([l,c]) =>
    `<tr><td><span class="level-badge level-${l.toLowerCase()}">${l}</span></td><td>${fmt(c)}</td><td>${(c/total*100).toFixed(2)}%</td></tr>`).join('');
}

/* Appendix tables */
function renderReportAppendix(data) {
  const el = document.getElementById('report-appendix-content');
  if (!el) return;
  const loggers  = (data.all_loggers   || []).slice(0, 30);
  const samples  = (data.error_samples || []).slice(0, 50);
  const bursts   = data.error_bursts   || [];
  const patterns = (data.top_patterns  || []).slice(0, 30);

  el.innerHTML = `
    <h3 class="report-appendix-h">A. Logger Health Summary</h3>
    <div class="report-table-wrap">
    <table class="report-stat-table">
      <thead><tr><th>Logger</th><th>Events</th><th>Errors</th><th>Error Rate</th><th>Health</th></tr></thead>
      <tbody>${loggers.map(l => {
        const errs = Math.round((l.error_rate||0)*l.count);
        const rate = ((l.error_rate||0)*100).toFixed(1);
        const icon = +rate>20 ? '🔴 Critical' : +rate>5 ? '⚠️ Warning' : '✅ Healthy';
        return `<tr><td><code class="rcode">${esc(l.logger)}</code></td><td>${fmt(l.count)}</td><td>${fmt(errs)}</td><td>${rate}%</td><td class="rhealth">${icon}</td></tr>`;
      }).join('')}</tbody>
    </table></div>

    <h3 class="report-appendix-h">B. Error Samples</h3>
    <div class="report-table-wrap">
    <table class="report-stat-table">
      <thead><tr><th>Timestamp</th><th>Logger</th><th>Message</th></tr></thead>
      <tbody>${samples.map(s =>
        `<tr><td class="report-ts">${esc(s.ts||'')}</td><td><code class="rcode">${esc(s.logger||'')}</code></td><td class="report-msg">${esc((s.msg||'').slice(0,200))}</td></tr>`
      ).join('')}</tbody>
    </table></div>

    <h3 class="report-appendix-h">C. Error Bursts</h3>
    ${bursts.length
      ? `<div class="report-table-wrap"><table class="report-stat-table">
           <thead><tr><th>Hour</th><th>Error Count</th></tr></thead>
           <tbody>${bursts.map(b=>`<tr><td class="report-ts">${esc(b.hour||'')}</td><td>${fmt(b.count||0)}</td></tr>`).join('')}</tbody>
         </table></div>`
      : '<p class="report-muted">No error bursts detected.</p>'}

    <h3 class="report-appendix-h">D. Top Message Patterns</h3>
    <div class="report-table-wrap">
    <table class="report-stat-table">
      <thead><tr><th>Pattern</th><th>Count</th></tr></thead>
      <tbody>${patterns.map(p=>
        `<tr><td class="report-pattern">${esc((p.pattern||'').slice(0,120))}</td><td>${fmt(p.count||0)}</td></tr>`
      ).join('')}</tbody>
    </table></div>`;
}

/* Sidebar helpers */
function setTrackerStatus(secId, status) {
  const el = document.getElementById(`rtst-${secId}`);
  const item = document.getElementById(`rtrack-${secId}`);
  if (!el) return;
  item?.classList.remove('rtrack-active','rtrack-done');
  if (status === 'pending') {
    el.innerHTML = '<span class="material-symbols-rounded rtrack-pending">radio_button_unchecked</span>';
  } else if (status === 'active') {
    el.innerHTML = '<span class="material-symbols-rounded rtrack-active spin">autorenew</span>';
    item?.classList.add('rtrack-active');
  } else if (status === 'done') {
    el.innerHTML = '<span class="material-symbols-rounded rtrack-done">check_circle</span>';
    item?.classList.add('rtrack-done');
  }
}

function setOverallProgress(pct, label) {
  const fill = document.getElementById('report-progress-fill');
  const lbl  = document.getElementById('report-overall-label');
  const pctEl = document.getElementById('report-overall-pct');
  if (fill)  fill.style.width  = pct + '%';
  if (lbl)   lbl.textContent   = label;
  if (pctEl) pctEl.textContent = pct + '%';
}

function setCurrentSectionLabel(text) {
  const el = document.getElementById('report-current-section');
  if (el) el.textContent = text;
}

function finishReportGeneration(success) {
  reportGenRunning = false;
  if (reportEventSource) { reportEventSource.close(); reportEventSource = null; }

  const genBtn = document.getElementById('report-generate-btn');
  genBtn.disabled = false;
  genBtn.innerHTML = '<span class="material-symbols-rounded">refresh</span><span>Regenerate</span>';

  if (success && reportCurrentId) {
    document.getElementById('report-download-btn').disabled = false;
  }
  // Mark all pending sections as done (in case server closed connection)
  RPT_SECTIONS.forEach(s => {
    const st = document.getElementById(`rtst-${s.id}`);
    if (st && st.querySelector('.rtrack-active')) setTrackerStatus(s.id, 'done');
  });
}

function downloadReport() {
  if (!reportCurrentId) return;
  const a = document.createElement('a');
  a.href = `/api/report/${reportCurrentId}/html`;
  a.download = `log-report-${reportCurrentId}.html`;
  a.click();
}

// ── Bootstrap ─────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadSettings();
  initUpload();
  initTimeline();
  initChat();
  initSessions();
  initSettings();
  initReport();
  loadModels();

  // Nav routing
  document.querySelectorAll('.nav-item[data-page]').forEach(btn =>
    btn.addEventListener('click', () => navigateTo(btn.dataset.page)));

  // Check if a log is already loaded (page reload)
  fetch('/api/status').then(r => r.json()).then(s => {
    if (s.has_summary) {
      onLoadComplete();
    }
  }).catch(() => {});
});
