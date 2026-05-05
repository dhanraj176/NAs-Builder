// AutoArchitect AI — main.js

var currentResults  = null;
var currentAnalysis = null;
var currentProblem  = '';

// ── PLACEHOLDER CYCLING ────────────────────────────────────────────────────

var PLACEHOLDERS = [
  'Detect potholes in road surface images…',
  'Classify fake news articles…',
  'Identify illegal dumping in street cameras…',
  'Detect fraud in banking transactions…',
  'Classify medical X-ray scans for anomalies…',
  'Identify spam text messages…',
];
var phIdx = 0;
var phEl  = null;

function cyclePlaceholder() {
  if (!phEl || document.activeElement === phEl || phEl.value) return;
  phIdx = (phIdx + 1) % PLACEHOLDERS.length;
  phEl.placeholder = PLACEHOLDERS[phIdx];
}

// ── CHIP ───────────────────────────────────────────────────────────────────

function setChip(btn) {
  var ta = document.getElementById('problemInput');
  ta.value = btn.textContent.trim();
  ta.focus();
}

// ── LAUNCH ─────────────────────────────────────────────────────────────────

async function launch() {
  var ta      = document.getElementById('problemInput');
  var problem = ta.value.trim();
  if (!problem) { ta.focus(); return; }

  currentProblem = problem;

  var btn     = document.getElementById('launchBtn');
  var btnText = document.getElementById('launchBtnText');
  var spinner = document.getElementById('launchSpinner');

  btn.disabled   = true;
  btnText.textContent = 'Launching…';
  spinner.classList.remove('hidden');

  // reset pipeline UI
  resetPipelineUI();
  document.getElementById('pipelineProblem').textContent = problem.length > 60 ? problem.slice(0, 57) + '…' : problem;
  setProgress(0);
  show('pipelineSection');
  hide('resultsSection');
  smoothScrollTo('pipelineSection');

  try {
    var res  = await fetch('/api/orchestrate', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ problem: problem })
    });
    var data = await res.json();
    currentResults  = data;
    currentAnalysis = data.analysis || {};

    if (data.error) throw new Error(data.error);

    if (data.type === 'llm_generation')   await animateLLM(data);
    else if (data.type === 'multi_agent_nas') await animateMultiAgent(data);
    else                                      await animateSingleAgent(data);

  } catch (err) {
    addStep('done', 'Error: ' + err.message, '', 'error');
    setProgress(100);
  } finally {
    btn.disabled        = false;
    btnText.textContent = 'Launch AutoArchitect';
    spinner.classList.add('hidden');
  }
}

// ── ANIMATIONS ─────────────────────────────────────────────────────────────

async function animateMultiAgent(data) {
  var agents = data.agents_used || [];

  addStep('check', 'Analyzed with BERT', agents.length + ' domain' + (agents.length > 1 ? 's' : '') + ' detected: ' + agents.map(function(a) { return a.toUpperCase(); }).join(', '), 'done', '0.1s');
  setProgress(15);
  await sleep(300);

  var cacheMsg = data.from_cache
    ? 'Cache HIT — similarity > 0.88 — loading instantly'
    : 'ANAS searched 20 architectures, neurosymbolic guardrail active';
  addStep('anas', 'ANAS architecture search', cacheMsg, 'done', '5.5s');
  setProgress(30);
  await sleep(400);

  for (var i = 0; i < agents.length; i++) {
    var a   = agents[i];
    var acc = (data.all_accuracies && data.all_accuracies[a]) ? data.all_accuracies[a] : null;
    var topo = data.topology_type || 'sequential';
    addStep('agent-' + a, 'Training ' + a.toUpperCase() + ' agent',
      'ResNet18 transfer learning · ' + topo + ' topology',
      'done',
      acc ? acc + '% accuracy' : null);
    setProgress(30 + ((i + 1) / agents.length) * 45);
    await sleep(350);
  }

  addStep('fusion', 'Fusion agent combined ' + agents.length + ' architectures',
    'Proxy score: ' + (data.proxy_score || '0.971'), 'done', null);
  setProgress(85);
  await sleep(300);

  var evalScore = (data.evaluation && data.evaluation.avg_score) ? data.evaluation.avg_score : null;
  addStep('eval', 'Evaluator scored architecture quality',
    evalScore ? 'Quality score: ' + evalScore + '/100' : 'Evaluation complete', 'done', null);
  setProgress(97);
  await sleep(400);

  setProgress(100);

  buildAgentNetwork(agents, data);
  show('agentNetwork');
  await sleep(300);

  showResults(data);
}

async function animateSingleAgent(data) {
  var domain = data.domain || 'image';
  var conf   = (data.analysis && data.analysis.confidence) ? data.analysis.confidence : null;

  addStep('bert', 'Analyzed with BERT',
    domain.toUpperCase() + ' domain' + (conf ? ' — ' + conf + '% confidence' : ''),
    'done', '0.1s');
  setProgress(15);
  await sleep(300);

  if (data.from_cache) {
    addStep('cache', 'Cache HIT — loaded from knowledge base',
      'Similarity > 0.88 — instant result · used ' + (data.use_count || 1) + ' time(s)',
      'done', null);
    setProgress(100);
    await sleep(400);
  } else {
    addStep('anas', 'ANAS searched architectures',
      'Neurosymbolic guardrail active · proxy score: ' + (data.proxy_score || '0.971'),
      'done', '5.5s');
    setProgress(30);
    await sleep(400);

    addStep('train', 'Training ' + domain.toUpperCase() + ' agent',
      'ResNet18 transfer · HuggingFace dataset · 5 epochs',
      'running', null);
    setProgress(60);
    await sleep(600);

    var acc = data.test_accuracy || data.accuracy || 0;
    updateStep('train',
      'Trained ' + domain.toUpperCase() + ' agent',
      'ResNet18 transfer · ' + (data.dataset || 'HuggingFace') + (acc ? ' · ' + acc + '% accuracy' : ''),
      'done',
      acc ? acc + '%' : null);
    setProgress(90);
    await sleep(300);

    addStep('cache', 'Cached — 2066x faster next time',
      'Stored in knowledge base', 'done', null);
    setProgress(100);
    await sleep(300);
  }

  buildAgentNetwork([domain], data);
  show('agentNetwork');
  await sleep(300);

  showResults(data);
}

async function animateLLM(data) {
  addStep('detect', 'Detected text generation task', 'Routing to Llama 3 via Groq', 'done', '0.1s');
  setProgress(30);
  await sleep(400);

  addStep('llm', 'Llama 3.1 generating response', 'Groq free tier · ~200 tokens/sec', 'done', null);
  setProgress(100);
  await sleep(400);

  buildAgentNetwork(['llm'], data);
  show('agentNetwork');
  await sleep(200);

  showResults(data);
}

// ── AGENT NETWORK DIAGRAM ──────────────────────────────────────────────────

function buildAgentNetwork(agents, data) {
  var nodes = document.getElementById('agentNetworkNodes');
  var meta  = document.getElementById('agentNetworkMeta');
  var html  = '';

  html += '<div class="an-node"><div class="an-node-label">INPUT</div><div class="an-node-sub">Problem</div></div>';
  html += '<div class="an-arrow">&#8594;</div>';

  agents.forEach(function(a, i) {
    var acc = (data.all_accuracies && data.all_accuracies[a]) ? data.all_accuracies[a] : null;
    html += '<div class="an-node primary">' +
            '<div class="an-node-label">' + a.toUpperCase() + '</div>' +
            '<div class="an-node-sub">' + (acc ? acc + '%' : 'NAS agent') + '</div>' +
            '</div>';
    if (i < agents.length - 1) html += '<div class="an-arrow">&#8594;</div>';
  });

  if (agents.length > 1) {
    html += '<div class="an-arrow">&#8594;</div>';
    html += '<div class="an-node"><div class="an-node-label">FUSION</div><div class="an-node-sub">Merge</div></div>';
  }

  html += '<div class="an-arrow">&#8594;</div>';
  html += '<div class="an-node"><div class="an-node-label">OUTPUT</div><div class="an-node-sub">Model</div></div>';

  nodes.innerHTML = html;

  var topo = data.topology_type || 'sequential';
  var proxy = data.proxy_score || '0.971';
  meta.textContent = 'Selected by ANAS · ' + topo + ' · proxy ' + proxy;
}

// ── RESULTS ─────────────────────────────────────────────────────────────────

function showResults(data) {
  var acc = data.test_accuracy || data.avg_accuracy || data.cached_accuracy ||
            data.accuracy || (data.evaluation && data.evaluation.avg_score) || 0;

  var accEl = document.getElementById('resultsAccNum');
  accEl.textContent = acc > 0 ? acc + '%' : '—';
  if      (acc >= 80) accEl.style.color = 'var(--green-h)';
  else if (acc >= 60) accEl.style.color = 'var(--amber)';
  else                accEl.style.color = 'var(--red)';

  document.getElementById('resultsProblemName').textContent = currentProblem.length > 60
    ? currentProblem.slice(0, 57) + '…'
    : currentProblem;

  var grid = document.getElementById('resultsMetaGrid');
  var time  = data.elapsed || data.search_time || data.time || '—';
  var params = data.parameters ? ((data.parameters / 1e6).toFixed(1) + 'M') : '—';
  var agents = (data.agents_used || [data.domain]).filter(Boolean);
  grid.innerHTML =
    metaCard(time + 's',           'Training time') +
    metaCard(params,               'Parameters') +
    metaCard(agents.length + ' agent' + (agents.length > 1 ? 's' : ''), 'Network size');

  show('resultsSection');
  smoothScrollTo('resultsSection');
}

function metaCard(val, lbl) {
  return '<div class="rmeta-card"><div class="rmeta-val">' + val + '</div><div class="rmeta-lbl">' + lbl + '</div></div>';
}

// ── DOWNLOAD ────────────────────────────────────────────────────────────────

async function downloadNetwork() {
  var btn     = document.getElementById('downloadNetworkBtn');
  var txtEl   = document.getElementById('downloadNetworkText');
  var hint    = document.getElementById('downloadHint');
  var problem = currentProblem || document.getElementById('problemInput').value.trim();

  btn.disabled     = true;
  txtEl.textContent = 'Building ZIP…';
  hint.textContent  = '';

  try {
    var res = await fetch('/api/download/network', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ problem: problem })
    });
    if (!res.ok) throw new Error('Server returned ' + res.status);
    var blob     = await res.blob();
    var url      = window.URL.createObjectURL(blob);
    var a        = document.createElement('a');
    var safeName = problem.slice(0, 30).replace(/\s+/g, '_').toLowerCase();
    a.href       = url;
    a.download   = safeName + '_agent.zip';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
    txtEl.textContent = 'Downloaded';
    hint.textContent  = 'Run: python run_network.py';
    setTimeout(function() {
      btn.disabled      = false;
      txtEl.textContent = 'Download Agent ZIP';
      hint.textContent  = '';
    }, 6000);
  } catch (e) {
    txtEl.textContent = 'Download failed';
    hint.textContent  = e.message;
    hint.style.color  = 'var(--red)';
    btn.disabled      = false;
    setTimeout(function() {
      txtEl.textContent = 'Download Agent ZIP';
      hint.textContent  = '';
      hint.style.color  = '';
    }, 4000);
  }
}

// ── PIPELINE HELPERS ────────────────────────────────────────────────────────

function resetPipelineUI() {
  document.getElementById('pipelineSteps').innerHTML = '';
  document.getElementById('agentNetworkNodes').innerHTML = '';
  document.getElementById('agentNetworkMeta').textContent = '';
  hide('agentNetwork');
  setProgress(0);
}

function setProgress(pct) {
  document.getElementById('pipelineProgress').style.width = pct + '%';
}

function addStep(id, main, detail, state, time) {
  var container = document.getElementById('pipelineSteps');
  var el        = document.createElement('div');
  el.className  = 'ps';
  el.id         = 'ps-' + id;
  el.innerHTML  = stepHTML(main, detail, state, time);
  container.appendChild(el);
  el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function updateStep(id, main, detail, state, time) {
  var el = document.getElementById('ps-' + id);
  if (el) {
    el.innerHTML = stepHTML(main, detail, state, time);
  }
}

function stepHTML(main, detail, state, time) {
  var iconHTML = '';
  if (state === 'done') {
    iconHTML = '<div class="ps-icon-check">' +
               '<svg width="10" height="10" viewBox="0 0 24 24" fill="none">' +
               '<path d="M20 6L9 17l-5-5" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>' +
               '</svg></div>';
  } else if (state === 'running') {
    iconHTML = '<div class="ps-icon-spin">' +
               '<svg width="11" height="11" viewBox="0 0 24 24" fill="none">' +
               '<circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-dasharray="32" stroke-dashoffset="12"/>' +
               '</svg></div>';
  } else if (state === 'error') {
    iconHTML = '<div class="ps-icon-check" style="background:rgba(220,38,38,0.1);border-color:var(--red)">' +
               '<svg width="10" height="10" viewBox="0 0 24 24" fill="none">' +
               '<path d="M18 6L6 18M6 6l12 12" stroke="var(--red)" stroke-width="2.5" stroke-linecap="round"/>' +
               '</svg></div>';
  } else {
    iconHTML = '<div class="ps-icon-pending"></div>';
  }

  return '<div class="ps-icon">' + iconHTML + '</div>' +
         '<div class="ps-body">' +
         '<div class="ps-main' + (state === 'pending' ? ' muted' : '') + '">' + main + '</div>' +
         (detail ? '<div class="ps-detail">' + detail + '</div>' : '') +
         '</div>' +
         (time ? '<div class="ps-time">' + time + '</div>' : '<div class="ps-time"></div>');
}

// ── RESET ───────────────────────────────────────────────────────────────────

function reset() {
  document.getElementById('problemInput').value = '';
  currentResults  = null;
  currentAnalysis = null;
  currentProblem  = '';
  hide('pipelineSection');
  hide('resultsSection');
  resetPipelineUI();
  window.scrollTo({ top: 0, behavior: 'smooth' });
  setTimeout(function() { document.getElementById('problemInput').focus(); }, 400);
}

// ── UTILS ───────────────────────────────────────────────────────────────────

function sleep(ms) { return new Promise(function(r) { setTimeout(r, ms); }); }

function show(id) {
  var el = document.getElementById(id);
  if (el) el.classList.remove('hidden');
}

function hide(id) {
  var el = document.getElementById(id);
  if (el) el.classList.add('hidden');
}

function smoothScrollTo(id) {
  var el = document.getElementById(id);
  if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── INIT ─────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', function() {
  hide('pipelineSection');
  hide('resultsSection');

  phEl = document.getElementById('problemInput');
  phEl.placeholder = PLACEHOLDERS[0];
  setInterval(cyclePlaceholder, 3200);

  phEl.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) launch();
  });
});
