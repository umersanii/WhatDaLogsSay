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
  html.dataset.theme     = s.theme     || 'light';
  html.dataset.palette   = s.palette   || 'blue';
  html.dataset.density   = s.density   || 'normal';
  html.dataset.fontScale = s.fontScale || 1;
  html.style.setProperty('--font-scale', s.fontScale || 1);

  // Sync settings UI
  document.querySelectorAll('.seg-btn[data-setting="theme"]').forEach(b =>
    b.classList.toggle('active', b.dataset.value === (s.theme || 'light')));
  document.querySelectorAll('.seg-btn[data-setting="density"]').forEach(b =>
    b.classList.toggle('active', b.dataset.value === (s.density || 'normal')));
  document.querySelectorAll('.palette-dot').forEach(d =>
    d.classList.toggle('active', d.dataset.palette === (s.palette || 'blue')));
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
  'Uploading…': 15,
  'Parsing log file…': 35,
  'Computing statistics and characterising log with Claude…': 60,
  'Building RAG index…': 80,
  'Ready': 100,
};
function updateProgress(s) {
  const bar   = document.getElementById('progress-bar');
  const label = document.getElementById('progress-label');
  const pct   = STEP_PROGRESS[s.step] || (s.is_processing ? 50 : 100);
  bar.style.width = pct + '%';
  label.textContent = s.step || 'Processing…';
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

// Parse :::type{json}::: blocks out of markdown text
function parseRichBlocks(text) {
  const BLOCK_RE = /:::([a-z-]+)(\{[\s\S]*?\}):::/g;
  const segments = [];
  let last = 0;
  let m;
  while ((m = BLOCK_RE.exec(text)) !== null) {
    if (m.index > last) segments.push({ kind: 'md', text: text.slice(last, m.index) });
    try {
      const data = JSON.parse(m[2]);
      segments.push({ kind: m[1], data });
    } catch {
      segments.push({ kind: 'md', text: m[0] });
    }
    last = m.index + m[0].length;
  }
  if (last < text.length) segments.push({ kind: 'md', text: text.slice(last) });
  return segments;
}

function renderRichSegments(segments) {
  const container = document.createElement('div');
  container.className = 'rich-content';
  segments.forEach(seg => {
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
  return container;
}

function renderRichComponent(kind, data) {
  switch (kind) {
    case 'log-ref':    return renderLogRef(data);
    case 'chart':      return renderInlineChart(data);
    case 'quiz':       return renderQuiz(data);
    case 'metric':     return renderMetric(data);
    case 'timeline':   return renderTimeline2(data);
    default:           return null;
  }
}

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
  div.addEventListener('click', () => {
    // Jump to timeline with this timestamp
    if (d.ts) {
      tlFilters.keyword = d.msg ? d.msg.slice(0, 40) : '';
      document.getElementById('tl-keyword').value = tlFilters.keyword;
      loadTimeline(1);
      navigateTo('timeline');
    }
  });
  div.title = 'Click to find in Timeline';
  return div;
}

function renderInlineChart(d) {
  const wrap = document.createElement('div');
  wrap.className = 'rich-chart-wrap';
  if (d.title) {
    const h = document.createElement('div');
    h.className = 'rich-chart-title';
    h.textContent = d.title;
    wrap.appendChild(h);
  }
  const canvas = document.createElement('canvas');
  canvas.height = 160;
  wrap.appendChild(canvas);

  const c = getChartColors();
  const colorMap = { error: c.error, warning: c.warning, success: c.success, primary: c.primary, info: c.info };

  const datasets = (d.datasets || []).map(ds => ({
    label: ds.label || '',
    data: ds.data || [],
    backgroundColor: (colorMap[ds.color] || c.primary) + 'cc',
    borderColor: colorMap[ds.color] || c.primary,
    borderWidth: 1.5,
    borderRadius: 4,
    fill: d.type === 'line',
    tension: 0.3,
    pointRadius: d.type === 'line' ? 3 : 0,
  }));

  setTimeout(() => {
    new Chart(canvas.getContext('2d'), {
      type: d.type || 'bar',
      data: { labels: d.labels || [], datasets },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: { display: datasets.length > 1, labels: { color: c.onSurface, boxWidth: 10, font: { size: 10 } } },
        },
        scales: {
          x: { ticks: { color: c.onSurface, font: { size: 9 }, maxRotation: 30 }, grid: { color: c.outline + '40' } },
          y: { ticks: { color: c.onSurface, font: { size: 9 } }, grid: { color: c.outline + '40' } },
        },
      },
    });
  }, 0);
  return wrap;
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
    btn.dataset.idx = i;
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

function renderMetric(d) {
  const colorMap = { error: 'var(--c-error)', warning: 'var(--c-warning)', success: 'var(--c-success)', info: 'var(--c-info)', primary: 'var(--c-primary)' };
  const col = colorMap[d.color] || 'var(--c-primary)';
  const trendIcon = d.trend === 'up' ? 'trending_up' : d.trend === 'down' ? 'trending_down' : 'trending_flat';
  const div = document.createElement('div');
  div.className = 'rich-metric';
  div.style.borderLeftColor = col;
  div.innerHTML = `
    <div class="rich-metric-value" style="color:${col}">${esc(String(d.value || ''))}</div>
    <div class="rich-metric-label">${esc(d.label || '')}</div>
    ${d.note ? `<div class="rich-metric-note">${esc(d.note)}</div>` : ''}
    ${d.trend ? `<span class="material-symbols-rounded rich-metric-trend" style="color:${col}">${trendIcon}</span>` : ''}`;
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
    item.className = `rich-tl-item`;
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
    // Parse rich components
    const segments = parseRichBlocks(bubbleText);
    const hasRich = segments.some(s => s.kind !== 'md');
    if (hasRich) {
      currentBubble.innerHTML = '';
      currentBubble.appendChild(renderRichSegments(segments));
    } else {
      currentBubble.innerHTML = marked.parse(bubbleText);
    }
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

// ── Settings drawer ───────────────────────────────────────────────────────────
function initSettings() {
  const drawer  = document.getElementById('settings-drawer');
  const overlay = document.getElementById('drawer-overlay');
  const open    = () => { drawer.classList.add('open'); overlay.classList.add('active'); };
  const close   = () => { drawer.classList.remove('open'); overlay.classList.remove('active'); };

  document.getElementById('settings-btn').addEventListener('click', open);
  document.getElementById('close-settings').addEventListener('click', close);
  overlay.addEventListener('click', close);

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

// ── Bootstrap ─────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadSettings();
  initUpload();
  initTimeline();
  initChat();
  initSettings();
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
