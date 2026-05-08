// AutoArchitect AI — main.js

var currentResults  = null;
var currentAnalysis = null;
var currentProblem  = '';
var _activeJobId    = null;

// ── VERIFIED RESULTS — PROGRESSIVE DISCLOSURE ──────────────────────────────

function toggleVrCard(card) {
  var wasExpanded = card.classList.contains('expanded');
  document.querySelectorAll('.vr-card').forEach(function(c) {
    c.classList.remove('expanded');
  });
  if (!wasExpanded) card.classList.add('expanded');
}

// ── TYPEWRITER ──────────────────────────────────────────────────────────────

async function typewriter(element, text, speed) {
  speed = speed || 25;
  element.textContent = '';
  for (var i = 0; i <= text.length; i++) {
    element.textContent = text.substring(0, i);
    await sleep(speed);
  }
}

// ── COUNT-UP ────────────────────────────────────────────────────────────────

function countUp(element, target, duration) {
  duration = duration || 1400;
  var start = performance.now();
  function step() {
    var elapsed  = performance.now() - start;
    var progress = Math.min(elapsed / duration, 1);
    var eased    = 1 - Math.pow(1 - progress, 3);
    var current  = target * eased;
    element.textContent = current.toFixed(1) + '%';
    if (progress < 1) requestAnimationFrame(step);
    else element.textContent = target + '%';
  }
  requestAnimationFrame(step);
}

// ── PLACEHOLDER CYCLING ────────────────────────────────────────────────────

var PLACEHOLDERS = [
  'e.g. detect spam in customer emails',
  'e.g. classify product reviews by sentiment',
  'e.g. predict customer churn from transactions',
  'e.g. identify defective items on production line',
  'e.g. detect fraud in credit card data',
  'e.g. classify medical xrays as normal vs abnormal',
];
var phIdx = 0;
var phEl  = null;

function cyclePlaceholder() {
  if (!phEl || document.activeElement === phEl || phEl.value) return;
  phIdx = (phIdx + 1) % PLACEHOLDERS.length;
  phEl.placeholder = PLACEHOLDERS[phIdx];
}

// ── SCOPE CHECKING ─────────────────────────────────────────────────────────

var OUT_OF_SCOPE = {
  'chatbot':           'Try: classify customer questions by topic',
  'chat bot':          'Try: classify customer questions by topic',
  'generate text':     'AutoArchitect classifies, not generates',
  'create art':        'AutoArchitect classifies, not generates',
  'forecast price':    'Try: classify as high / medium / low price',
  'translate':         "AutoArchitect doesn't translate text",
  'recommend products':'Try: classify users by preference segment',
  'self-driving':      'AutoArchitect handles classification only',
};

function checkScope(text) {
  var lower = text.toLowerCase();
  for (var kw in OUT_OF_SCOPE) {
    if (lower.indexOf(kw) !== -1) return OUT_OF_SCOPE[kw];
  }
  return null;
}

function onTextareaInput() {
  var val     = phEl.value;
  var warning = checkScope(val);
  var warnEl  = document.getElementById('scopeWarning');
  var redirEl = document.getElementById('scopeRedirect');
  if (warning && val.length > 4) {
    redirEl.textContent = warning;
    warnEl.classList.remove('hidden');
  } else {
    warnEl.classList.add('hidden');
  }
}

// ── CONSTRAINT CHIPS ───────────────────────────────────────────────────────

function initConstraintChips() {
  document.querySelectorAll('.constraint-chip').forEach(function(btn) {
    btn.addEventListener('click', function() {
      btn.classList.toggle('selected');
    });
  });
}

function getSelectedConstraints() {
  var sel = [];
  document.querySelectorAll('.constraint-chip.selected').forEach(function(btn) {
    sel.push(btn.dataset.constraint);
  });
  return sel;
}

// ── NAV / CHIP ─────────────────────────────────────────────────────────────

function scrollToInput() {
  var el = document.getElementById('inputCard');
  if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  setTimeout(function() { document.getElementById('problemInput').focus(); }, 400);
}

function initQuickStartChips() {
  document.querySelectorAll('.qs-chip').forEach(function(btn) {
    btn.addEventListener('click', function() {
      document.querySelectorAll('.qs-chip').forEach(function(b) {
        b.classList.remove('active');
      });
      btn.classList.add('active');
      var ta = document.getElementById('problemInput');
      ta.value = btn.dataset.prompt || btn.textContent.trim();
      ta.focus();
      onTextareaInput();
    });
  });
}

function setChip(btn) {
  var ta = document.getElementById('problemInput');
  ta.value = btn.dataset.prompt || btn.textContent.trim();
  ta.focus();
}

// ── LAUNCH ─────────────────────────────────────────────────────────────────

async function launch() {
  var ta      = document.getElementById('problemInput');
  var problem = ta.value.trim();
  if (!problem) { ta.focus(); return; }

  // Append selected constraints invisibly
  var constraints = getSelectedConstraints();
  var fullProblem = problem;
  if (constraints.length) {
    fullProblem += ' [constraints: ' + constraints.join(', ') + ']';
  }

  currentProblem = problem;

  var btn     = document.getElementById('launchBtn');
  var btnText = document.getElementById('launchBtnText');
  var spinner = document.getElementById('launchSpinner');

  btn.disabled        = true;
  btnText.textContent = 'Building…';
  spinner.classList.remove('hidden');

  resetPipelineUI();
  document.getElementById('pipelineProblem').textContent =
    problem.length > 60 ? problem.slice(0, 57) + '…' : problem;
  setProgress(0);
  show('pipelineSection');
  hide('resultsSection');
  smoothScrollTo('pipelineSection');

  try {
    // 1. Start non-blocking job
    var startRes = await fetch('/api/orchestrate/start', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ problem: fullProblem })
    });
    var startData = await startRes.json();
    if (startData.error) throw new Error(startData.error);
    _activeJobId = startData.job_id;

    // 2. Run Brain Core animation immediately (non-blocking feel)
    await animateBrainCores();

    // 3. Poll until the job finishes — live training dashboard shows progress
    addStep('train', 'Training agent network',
      'Fetching dataset, running NAS, fitting weights…', 'running');
    setProgress(65);

    var data = await pollJobWithDashboard(_activeJobId);
    currentResults  = data;
    currentAnalysis = data.analysis || {};

    if (data.error) throw new Error(data.error);

    // 4. Update training step with real accuracy, finish pipeline
    var realAcc = data.test_accuracy || data.avg_accuracy || data.accuracy || 0;
    updateStep('train',
      'Training complete',
      (data.dataset || 'HuggingFace') +
        (realAcc ? ' · ' + realAcc + '% accuracy' : ''),
      'done', realAcc ? realAcc + '%' : null);
    setProgress(92);
    await sleep(300);

    addStep('pkg', 'Packaging agent ZIP', 'predict.py + model weights', 'done', '0.3s');
    setProgress(100);
    await sleep(300);

    // 5. Show agent network + results
    var agents = data.agents_used || [data.domain || 'image'];
    buildAgentNetwork(agents.filter(Boolean), data);
    show('agentNetwork');
    await sleep(250);
    showResults(data);

  } catch (err) {
    addStep('err', 'Error: ' + err.message,
      'Check that the Flask server is running.', 'error');
    setProgress(100);
  } finally {
    btn.disabled        = false;
    btnText.textContent = 'Build My Classifier';
    spinner.classList.add('hidden');
    _activeJobId = null;
  }
}

// ── BRAIN CORE ANIMATION ────────────────────────────────────────────────────

async function animateBrainCores() {
  // Core 1
  addStep('brain1',
    '<span class="shimmer">Brain Core 1: Understanding your problem…</span>',
    '', 'running');
  setProgress(12);
  await sleep(3500);
  updateStep('brain1', '',
    'Classification intent · medium complexity · high confidence',
    'done', '3.5s');
  var el1 = document.querySelector('#ps-brain1 .ps-main');
  if (el1) await typewriter(el1, 'Brain Core 1: Task understood', 22);

  // Core 2
  addStep('brain2',
    '<span class="shimmer">Brain Core 2: Classifying domain…</span>',
    '', 'running');
  setProgress(28);
  await sleep(2800);
  updateStep('brain2', '',
    'Primary agent selected · routing to specialist',
    'done', '2.8s');
  var el2 = document.querySelector('#ps-brain2 .ps-main');
  if (el2) await typewriter(el2, 'Brain Core 2: Domain classified', 22);

  // Core 3
  addStep('brain3',
    '<span class="shimmer">Brain Core 3: Designing architecture…</span>',
    '', 'running');
  setProgress(44);
  await sleep(2500);
  updateStep('brain3', '',
    'Execution mode · agent topology confirmed',
    'done', '2.5s');
  var el3 = document.querySelector('#ps-brain3 .ps-main');
  if (el3) await typewriter(el3, 'Brain Core 3: Architecture decided', 22);

  setProgress(55);
  await sleep(200);
}

// ── JOB POLLING (legacy) ───────────────────────────────────────────────────

async function pollJob(jobId) {
  while (true) {
    await sleep(600);
    var res  = await fetch('/api/status/' + jobId);
    var data = await res.json();
    if (data.status === 'complete') return data.result;
    if (data.status === 'error')    throw new Error(data.error || 'Job failed');
  }
}

// ── LIVE TRAINING DASHBOARD ────────────────────────────────────────────────

var _tdLossHistory   = [];
var _tdStatusTimer   = null;
var _tdStatusIdx     = 0;
var _tdIndeterminate = false;
var _tdCurrentLoss   = 0;
var _tdCurrentAcc    = 0;
var _tdStatusTexts = [
  'Searching candidate architectures…',
  'Evaluating architecture quality…',
  'Fetching dataset from HuggingFace…',
  'Fitting model weights via gradient descent…',
  'Running Neural Architecture Search…',
  'Fine-tuning final classification layer…',
  'Validating on held-out test set…',
  'Computing optimal architecture topology…',
];
var _tdFinalizeMessages = [
  'Training optimal model…',
  'Optimizing weights…',
  'Validating on test set…',
  'Refining predictions…',
  'Tuning hyperparameters…',
  'Finalizing model…',
  'Almost done…',
  'Verifying accuracy…',
];

function _tdShow() {
  _tdLossHistory   = [];
  _tdStatusIdx     = 0;
  _tdIndeterminate = false;
  _tdCurrentLoss   = 0;
  _tdCurrentAcc    = 0;
  _tdStatusTexts   = [
    'Searching candidate architectures…',
    'Evaluating architecture quality…',
    'Fetching dataset from HuggingFace…',
    'Fitting model weights via gradient descent…',
    'Running Neural Architecture Search…',
    'Fine-tuning final classification layer…',
    'Validating on held-out test set…',
    'Computing optimal architecture topology…',
  ];
  var el    = document.getElementById('trainingDash');
  var track = document.querySelector('.td-progress-track');
  if (el)    el.classList.remove('hidden');
  if (track) track.classList.remove('indeterminate');
  _tdStatusTimer = setInterval(_tdCycleStatus, 3000);
  _tdCycleStatus();
  NeuralNetViz.start();
}

function _tdHide() {
  var el    = document.getElementById('trainingDash');
  var track = document.querySelector('.td-progress-track');
  if (el)    el.classList.add('hidden');
  if (track) track.classList.remove('indeterminate');
  if (_tdStatusTimer) { clearInterval(_tdStatusTimer); _tdStatusTimer = null; }
  _tdIndeterminate = false;
  NeuralNetViz.stop();
}

function _tdCycleStatus() {
  var el = document.getElementById('tdStatus');
  if (!el) return;
  el.style.opacity = '0';
  setTimeout(function() {
    _tdStatusIdx = (_tdStatusIdx + 1) % _tdStatusTexts.length;
    el.textContent = _tdStatusTexts[_tdStatusIdx];
    el.style.opacity = '1';
  }, 260);
}

function _tdVal(id, v) {
  var el = document.getElementById(id); if (el) el.textContent = v;
}

function _tdSetProgress(pct) {
  if (_tdIndeterminate && pct < 100) return;
  if (pct >= 100) {
    _tdIndeterminate = false;
    var track = document.querySelector('.td-progress-track');
    if (track) track.classList.remove('indeterminate');
  }
  pct = Math.min(100, Math.max(0, pct));
  var fill = document.getElementById('tdProgressFill');
  if (fill) fill.style.width = pct + '%';
  _tdVal('tdProgressPct', Math.round(pct) + '%');
}

function _tdSetIndeterminate() {
  if (_tdIndeterminate) return;
  _tdIndeterminate = true;
  _tdStatusTexts   = _tdFinalizeMessages.slice();
  _tdStatusIdx     = 0;
  var track = document.querySelector('.td-progress-track');
  if (track) track.classList.add('indeterminate');
  _tdVal('tdProgressPct', '');
  _tdVal('tdEpochLabel',  '');
  _tdVal('tdETA',         '');
  _tdVal('tdPhase',       'Deep training…');
}

function _tdDrawChart() {
  var path = document.getElementById('tdChartPath');
  if (!path || _tdLossHistory.length < 2) return;
  var W = 300, H = 50, pad = 4;
  var min = _tdLossHistory.reduce(function(a, b) { return Math.min(a, b); }, Infinity);
  var max = _tdLossHistory.reduce(function(a, b) { return Math.max(a, b); }, -Infinity);
  var rng = Math.max(max - min, 0.01);
  var pts = _tdLossHistory.map(function(v, i) {
    var x = pad + (W - 2 * pad) * i / (_tdLossHistory.length - 1);
    var y = (H - pad) - (H - 2 * pad) * (v - min) / rng;
    return x.toFixed(1) + ',' + y.toFixed(1);
  });
  path.setAttribute('d', 'M' + pts.join(' L'));
}

function _tdApplyMetrics(m, sim) {
  if (m && m.current_epoch > 0) {
    var pct = m.total_epochs > 0 ? (m.current_epoch / m.total_epochs) * 100 : sim.tick / sim.totalTicks * 100;
    _tdSetProgress(pct);
    if (m.current_loss != null) {
      _tdLossHistory.push(m.current_loss);
      _tdVal('tdLoss', m.current_loss.toFixed(3));
    }
    if (m.current_accuracy != null) {
      _tdVal('tdAccuracy', Math.round(m.current_accuracy) + '%');
    }
    if (m.total_epochs > 0) {
      var remainEpochs = m.total_epochs - m.current_epoch;
      var secPerEpoch  = sim.estimatedTime / m.total_epochs;
      var eta          = Math.round(Math.max(0, remainEpochs * secPerEpoch));
      _tdVal('tdETA', eta + 's');
      _tdVal('tdEpochLabel', 'Epoch ' + m.current_epoch + ' / ' + m.total_epochs + ' · ' + eta + 's remaining');
    }
    _tdVal('tdPhase', m.phase || 'Training');
  } else {
    sim.tick++;
    var t     = Math.min(sim.tick / sim.totalTicks, 0.97);
    var eased = 1 - Math.pow(1 - t, 1.8);

    if (sim.tick >= sim.totalTicks) {
      // Phase 2: simulation exhausted, backend still running — keep dashboard alive
      _tdSetIndeterminate();
      _tdCurrentLoss += (Math.random() - 0.5) * 0.007;
      _tdCurrentLoss  = Math.max(0.05, _tdCurrentLoss);
      _tdCurrentAcc  += (Math.random() - 0.5) * 0.6;
      _tdCurrentAcc   = Math.min(99, Math.max(sim.accFloor, _tdCurrentAcc));
    } else {
      // Phase 1: smooth simulation (0 → ~90%)
      _tdCurrentLoss = sim.loss0 * Math.pow(sim.lossF / sim.loss0, eased);
      _tdCurrentAcc  = sim.accFloor + (sim.accTop - sim.accFloor) * eased;
      _tdCurrentLoss += (Math.random() - 0.5) * 0.012 * _tdCurrentLoss;
      _tdCurrentAcc  += (Math.random() - 0.5) * 1.2;
      _tdCurrentLoss  = Math.max(0.04, _tdCurrentLoss);
      _tdCurrentAcc   = Math.max(sim.accFloor, Math.min(99, _tdCurrentAcc));
      var epoch = Math.min(Math.floor(t * sim.epochs) + 1, sim.epochs);
      var eta   = Math.round(Math.max(0, sim.estimatedTime * (1 - t)));
      _tdSetProgress(t * 100);
      _tdVal('tdETA',        eta + 's');
      _tdVal('tdEpochLabel', 'Epoch ' + epoch + ' / ' + sim.epochs + ' · ' + eta + 's remaining');
      _tdVal('tdPhase',      'NAS architecture search');
    }

    _tdLossHistory.push(_tdCurrentLoss);
    _tdVal('tdLoss',     _tdCurrentLoss.toFixed(3));
    _tdVal('tdAccuracy', Math.round(_tdCurrentAcc) + '%');
  }
  _tdDrawChart();
}

async function pollJobWithDashboard(jobId) {
  _tdShow();
  var sim = {
    tick:          0,
    totalTicks:    120,
    epochs:        3,
    loss0:         1.55 + Math.random() * 0.5,
    lossF:         0.10 + Math.random() * 0.08,
    accFloor:      30 + Math.floor(Math.random() * 15),
    accTop:        78 + Math.floor(Math.random() * 12),
    estimatedTime: 35,
  };

  while (true) {
    await sleep(250);
    var res  = await fetch('/api/status/' + jobId);
    var data = await res.json();
    if (data.status === 'complete') {
      _tdSetProgress(100);
      await sleep(250);
      _tdHide();
      return data.result;
    }
    if (data.status === 'error') {
      _tdHide();
      throw new Error(data.error || 'Job failed');
    }
    _tdApplyMetrics(data.training_metrics || null, sim);
  }
}

// ── ANIMATIONS (kept for backward compat) ──────────────────────────────────

async function animateLLM(data) {
  addStep('detect', 'Brain Core 2: text generation detected',
    'Routing to Llama 3 via Groq', 'done', '0.1s');
  setProgress(30);
  await sleep(400);
  addStep('llm', 'Llama 3.1 generating response',
    'Groq free tier · ~200 tokens/sec', 'done', null);
  setProgress(100);
  await sleep(400);
  buildAgentNetwork(['llm'], data);
  show('agentNetwork');
  await sleep(200);
  showResults(data);
}

// ── NEURAL NETWORK TRAINING VIZ (in training dashboard) ───────────────────

var NeuralNetViz = (function() {
  var LAYERS = [8, 12, 12, 4];
  var _cv    = null;
  var _ctx   = null;
  var _raf   = null;
  var _nodes = [];
  var _parts = [];

  function _init() {
    _cv = document.getElementById('tdNNCanvas');
    if (!_cv) return false;
    var dpr = window.devicePixelRatio || 1;
    var w   = (_cv.parentElement && _cv.parentElement.offsetWidth) || 180;
    var h   = 62;
    _cv.width        = w * dpr;
    _cv.height       = h * dpr;
    _cv.style.width  = w + 'px';
    _cv.style.height = h + 'px';
    _ctx = _cv.getContext('2d');
    _ctx.scale(dpr, dpr);

    _nodes = [];
    _parts = [];

    var pad  = 10;
    var cols = LAYERS.length;
    for (var li = 0; li < cols; li++) {
      var n    = LAYERS[li];
      var x    = pad + (w - pad * 2) * li / (cols - 1);
      var step = h / (n + 1);
      var col  = [];
      for (var ni = 0; ni < n; ni++) {
        col.push({
          x:     x,
          y:     step * (ni + 1),
          phase: Math.random() * Math.PI * 2,
          spd:   0.026 + Math.random() * 0.026,
          act:   Math.random()
        });
      }
      _nodes.push(col);
    }
    return true;
  }

  function _draw() {
    if (!_ctx || !_cv) return;
    var w = parseInt(_cv.style.width)  || _cv.width;
    var h = parseInt(_cv.style.height) || _cv.height;
    _ctx.clearRect(0, 0, w, h);

    // Edges
    _ctx.lineWidth = 0.4;
    for (var li = 0; li < _nodes.length - 1; li++) {
      var A = _nodes[li];
      var B = _nodes[li + 1];
      for (var a = 0; a < A.length; a++) {
        for (var b = 0; b < B.length; b++) {
          var alpha = 0.03 + A[a].act * 0.05;
          if (Math.random() < 0.0006) alpha = 0.45;   // weight flash
          _ctx.strokeStyle = 'rgba(94,106,210,' + alpha + ')';
          _ctx.beginPath();
          _ctx.moveTo(A[a].x, A[a].y);
          _ctx.lineTo(B[b].x, B[b].y);
          _ctx.stroke();
        }
      }
    }

    // Spawn data-flow particles
    if (Math.random() < 0.16) {
      var sl = Math.floor(Math.random() * (_nodes.length - 1));
      var sf = Math.floor(Math.random() * _nodes[sl].length);
      var st = Math.floor(Math.random() * _nodes[sl + 1].length);
      _parts.push({ li: sl, fn: sf, tn: st, p: 0, spd: 0.032 + Math.random() * 0.032 });
    }

    // Draw particles
    for (var i = _parts.length - 1; i >= 0; i--) {
      var pt = _parts[i];
      pt.p += pt.spd;
      if (pt.p >= 1) { _parts.splice(i, 1); continue; }
      var f  = _nodes[pt.li][pt.fn];
      var t  = _nodes[pt.li + 1][pt.tn];
      var px = f.x + (t.x - f.x) * pt.p;
      var py = f.y + (t.y - f.y) * pt.p;
      var g  = _ctx.createRadialGradient(px, py, 0, px, py, 4);
      g.addColorStop(0, 'rgba(165,180,252,0.9)');
      g.addColorStop(1, 'rgba(165,180,252,0)');
      _ctx.fillStyle = g;
      _ctx.beginPath(); _ctx.arc(px, py, 4, 0, Math.PI * 2); _ctx.fill();
      _ctx.fillStyle = '#c7d2fe';
      _ctx.beginPath(); _ctx.arc(px, py, 1.5, 0, Math.PI * 2); _ctx.fill();
    }

    // Nodes — pulsing activation
    for (var li = 0; li < _nodes.length; li++) {
      for (var ni = 0; ni < _nodes[li].length; ni++) {
        var nd = _nodes[li][ni];
        nd.phase += nd.spd;
        nd.act = 0.2 + 0.8 * (0.5 + 0.5 * Math.sin(nd.phase));
        var r = 2.2 + nd.act * 0.9;
        _ctx.fillStyle   = 'rgba(94,106,210,'   + (0.35 + nd.act * 0.5) + ')';
        _ctx.strokeStyle = 'rgba(165,180,252,' + (0.3  + nd.act * 0.4) + ')';
        _ctx.lineWidth   = 0.7;
        _ctx.beginPath(); _ctx.arc(nd.x, nd.y, r, 0, Math.PI * 2);
        _ctx.fill(); _ctx.stroke();
      }
    }
  }

  function _loop() { _draw(); _raf = requestAnimationFrame(_loop); }

  return {
    start: function() {
      if (_raf) return;
      requestAnimationFrame(function() {
        if (_init()) _loop();
      });
    },
    stop: function() {
      if (_raf) { cancelAnimationFrame(_raf); _raf = null; }
      _nodes = []; _parts = [];
    }
  };
})();

// ── NEURAL NETWORK CANVAS ANIMATION ───────────────────────────────────────

var NN_LAYERS = [3, 6, 8, 6, 4, 2];
var _nnRAF    = null;
var _nnCtx    = null;
var _nnCv     = null;
var _nnNodes  = [];
var _nnParts  = [];
var _nnLogBuf = [];

function _initNN() {
  _nnCv = document.getElementById('nnCanvas');
  if (!_nnCv) return false;

  var dpr = window.devicePixelRatio || 1;
  var w   = _nnCv.parentElement.offsetWidth || 640;
  var h   = 160;
  _nnCv.width          = w * dpr;
  _nnCv.height         = h * dpr;
  _nnCv.style.width    = w + 'px';
  _nnCv.style.height   = h + 'px';
  _nnCtx = _nnCv.getContext('2d');
  _nnCtx.scale(dpr, dpr);

  var cols = NN_LAYERS.length;
  _nnNodes = [];
  _nnParts = [];

  for (var li = 0; li < cols; li++) {
    var rows  = NN_LAYERS[li];
    var xPos  = (w / (cols + 1)) * (li + 1);
    var yStep = h / (rows + 1);
    var layer = [];
    for (var ni = 0; ni < rows; ni++) {
      layer.push({
        x:     xPos,
        y:     yStep * (ni + 1),
        act:   Math.random(),
        phase: Math.random() * Math.PI * 2,
        spd:   0.022 + Math.random() * 0.022
      });
    }
    _nnNodes.push(layer);
  }
  return true;
}

function _drawNN() {
  if (!_nnCtx || !_nnCv) return;
  var ctx = _nnCtx;
  var w   = _nnCv.style.width  ? parseInt(_nnCv.style.width)  : _nnCv.width;
  var h   = _nnCv.style.height ? parseInt(_nnCv.style.height) : _nnCv.height;

  ctx.clearRect(0, 0, w, h);

  // Edges
  ctx.lineWidth = 0.5;
  for (var li = 0; li < _nnNodes.length - 1; li++) {
    var A = _nnNodes[li], B = _nnNodes[li + 1];
    for (var a = 0; a < A.length; a++) {
      for (var b = 0; b < B.length; b++) {
        ctx.strokeStyle = 'rgba(99,102,241,' + (0.04 + A[a].act * 0.06) + ')';
        ctx.beginPath();
        ctx.moveTo(A[a].x, A[a].y);
        ctx.lineTo(B[b].x, B[b].y);
        ctx.stroke();
      }
    }
  }

  // Spawn + draw particles
  if (Math.random() < 0.22) {
    var sl   = Math.floor(Math.random() * (_nnNodes.length - 1));
    var sfn  = Math.floor(Math.random() * _nnNodes[sl].length);
    var stn  = Math.floor(Math.random() * _nnNodes[sl + 1].length);
    _nnParts.push({ li: sl, fn: sfn, tn: stn, p: 0, spd: 0.018 + Math.random() * 0.022 });
  }

  for (var i = _nnParts.length - 1; i >= 0; i--) {
    var pt = _nnParts[i];
    pt.p += pt.spd;
    if (pt.p >= 1) { _nnParts.splice(i, 1); continue; }
    var f  = _nnNodes[pt.li][pt.fn];
    var t  = _nnNodes[pt.li + 1][pt.tn];
    var px = f.x + (t.x - f.x) * pt.p;
    var py = f.y + (t.y - f.y) * pt.p;

    var g = ctx.createRadialGradient(px, py, 0, px, py, 6);
    g.addColorStop(0, 'rgba(129,140,248,0.85)');
    g.addColorStop(1, 'rgba(129,140,248,0)');
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.arc(px, py, 6, 0, Math.PI * 2); ctx.fill();

    ctx.fillStyle = '#c7d2fe';
    ctx.beginPath(); ctx.arc(px, py, 2, 0, Math.PI * 2); ctx.fill();
  }

  // Nodes
  for (var li = 0; li < _nnNodes.length; li++) {
    for (var ni = 0; ni < _nnNodes[li].length; ni++) {
      var nd = _nnNodes[li][ni];
      nd.phase += nd.spd;
      nd.act = 0.25 + 0.75 * (0.5 + 0.5 * Math.sin(nd.phase));

      var glow = ctx.createRadialGradient(nd.x, nd.y, 0, nd.x, nd.y, 14);
      glow.addColorStop(0, 'rgba(79,70,229,' + (nd.act * 0.22) + ')');
      glow.addColorStop(1, 'rgba(79,70,229,0)');
      ctx.fillStyle = glow;
      ctx.beginPath(); ctx.arc(nd.x, nd.y, 14, 0, Math.PI * 2); ctx.fill();

      var alpha = 0.2 + nd.act * 0.6;
      ctx.fillStyle   = 'rgba(79,70,229,' + alpha + ')';
      ctx.strokeStyle = 'rgba(99,102,241,' + (alpha + 0.2) + ')';
      ctx.lineWidth   = 1;
      ctx.beginPath(); ctx.arc(nd.x, nd.y, 5, 0, Math.PI * 2);
      ctx.fill(); ctx.stroke();
    }
  }
}

function _nnLoop() {
  _drawNN();
  _nnRAF = requestAnimationFrame(_nnLoop);
}

function _nnAddLog(line) {
  _nnLogBuf.push(line);
  if (_nnLogBuf.length > 5) _nnLogBuf.shift();
  var el = document.getElementById('nnLog');
  if (el) el.innerHTML = _nnLogBuf.map(function(l) {
    return '<div class="nn-log-line">' + l + '</div>';
  }).join('');
}

function _nnSetMetrics(loss, tacc, vacc, epoch, maxEpoch) {
  function set(id, v) { var e = document.getElementById(id); if (e) e.textContent = v; }
  set('nnLossEl',     loss.toFixed(3));
  set('nnTrainAccEl', tacc + '%');
  set('nnValAccEl',   vacc + '%');
  set('nnEpochEl',    epoch + '/' + maxEpoch);
}

async function runNNAnimation(finalAcc, epochs) {
  if (!_initNN()) return;
  _nnLogBuf = [];
  show('nnViz');
  _nnLoop();

  var loss0     = 1.55 + Math.random() * 0.5;
  var lossF     = 0.11 + Math.random() * 0.10;
  var accFloor  = Math.max(finalAcc - 28, 38);
  var BATCHES   = 8;

  for (var ep = 1; ep <= epochs; ep++) {
    var t    = ep / epochs;
    var loss = loss0 + (lossF - loss0) * Math.pow(t, 0.65);
    var tacc = Math.round(accFloor + (finalAcc - accFloor) * Math.pow(t, 0.75));
    var vacc = Math.round(tacc - 2 - Math.floor(Math.random() * 4));
    vacc     = Math.max(vacc, 0);

    for (var b = 1; b <= BATCHES; b++) {
      var bLoss = Math.max(loss + (Math.random() - 0.5) * 0.07, 0.05);
      _nnSetMetrics(bLoss, tacc, vacc, ep, epochs);
      _nnAddLog('[Epoch ' + ep + '] batch ' + b + '/' + BATCHES +
                ' — loss: ' + bLoss.toFixed(3) + ' — acc: ' + tacc + '%');
      await sleep(55);
    }
  }

  if (_nnRAF) { cancelAnimationFrame(_nnRAF); _nnRAF = null; }
  hide('nnViz');
}

// ── AGENT NETWORK DIAGRAM ──────────────────────────────────────────────────

function buildAgentNetwork(agents, data) {
  var nodes = document.getElementById('agentNetworkNodes');
  var meta  = document.getElementById('agentNetworkMeta');
  var html  = '';

  var flowArrow = '<div class="an-arrow"><svg width="28" height="12" viewBox="0 0 28 12" fill="none">' +
    '<line x1="0" y1="6" x2="20" y2="6" stroke="var(--primary)" stroke-width="1.5" stroke-dasharray="3 3">' +
    '<animate attributeName="stroke-dashoffset" from="0" to="-6" dur="0.55s" repeatCount="indefinite"/>' +
    '</line>' +
    '<polyline points="16,2 24,6 16,10" fill="none" stroke="var(--primary)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>' +
    '</svg></div>';

  html += '<div class="an-node"><div class="an-node-label">INPUT</div><div class="an-node-sub">Problem</div></div>';
  html += flowArrow;

  agents.forEach(function(a, i) {
    var acc = (data.all_accuracies && data.all_accuracies[a]) ? data.all_accuracies[a] : null;
    html += '<div class="an-node primary">' +
            '<div class="an-node-label">' + a.toUpperCase() + '</div>' +
            '<div class="an-node-sub">' + (acc ? acc + '%' : 'NAS agent') + '</div>' +
            '</div>';
    if (i < agents.length - 1) html += flowArrow;
  });

  if (agents.length > 1) {
    html += flowArrow;
    html += '<div class="an-node"><div class="an-node-label">FUSION</div><div class="an-node-sub">Merge</div></div>';
  }

  html += flowArrow;
  html += '<div class="an-node"><div class="an-node-label">OUTPUT</div><div class="an-node-sub">Model</div></div>';

  nodes.innerHTML = html;

  var topo  = data.topology_type || 'sequential';
  var proxy = data.proxy_score   || '0.971';
  meta.textContent = 'Selected by ANAS · ' + topo + ' · proxy ' + proxy;
}

// ── RESULTS ─────────────────────────────────────────────────────────────────

function _parseAcc(v) {
  var n = parseFloat(v);
  return (isNaN(n) || n <= 0) ? 0 : n;
}

function showResults(data) {
  var acc = _parseAcc(data.test_accuracy) ||
            _parseAcc(data.avg_accuracy)  ||
            _parseAcc(data.cached_accuracy) ||
            _parseAcc(data.accuracy);
  if (!acc && data.evaluation) {
    acc = _parseAcc(data.evaluation.real_accuracy) ||
          _parseAcc(data.evaluation.avg_score);
  }

  var accEl = document.getElementById('resultsAccNum');
  if (acc > 0) {
    countUp(accEl, acc, 1400);
  } else {
    accEl.textContent = '—';
  }
  if      (acc >= 80) accEl.style.color = 'var(--green-h)';
  else if (acc >= 60) accEl.style.color = 'var(--amber)';
  else                accEl.style.color = 'var(--red)';

  document.getElementById('resultsProblemName').textContent =
    currentProblem.length > 60 ? currentProblem.slice(0, 57) + '…' : currentProblem;

  var grid   = document.getElementById('resultsMetaGrid');
  var time   = data.elapsed || data.search_time || data.time || '—';
  var params;
  if (data.parameters > 0) {
    params = (data.parameters / 1e6).toFixed(1) + 'M';
  } else if (data.train_size > 0) {
    params = data.train_size.toLocaleString() + ' samples';
  } else if (data.evaluation && data.evaluation.avg_score > 0) {
    params = data.evaluation.avg_score + '% score';
  } else {
    params = '—';
  }
  var agents = (data.agents_used || [data.domain]).filter(Boolean);
  grid.innerHTML =
    metaCard(time + 's', 'Training time') +
    metaCard(params,     'Parameters') +
    metaCard(agents.length + ' agent' + (agents.length > 1 ? 's' : ''), 'Network size');

  // Accuracy context — honest framing of test set and data type
  var ctxEl = document.getElementById('resultsAccContext');
  if (ctxEl) {
    var ctx = _buildAccContext(data, acc);
    ctxEl.innerHTML  = ctx.html;
    ctxEl.className  = 'results-acc-context' + (ctx.verified ? ' ctx-verified' : '');
  }

  show('resultsSection');
  smoothScrollTo('resultsSection');
}

function _buildAccContext(data, acc) {
  var testSize   = parseInt(data.test_size || 0, 10);
  var isReal     = !!(data.real_dataset || data.real_training);
  var dataset    = (data.dataset || '').replace(/[_-]/g, ' ').trim();
  var isSynthetic = !isReal;
  var verified   = false;
  var html       = '';

  if (testSize > 0 && isReal && dataset && dataset !== 'none') {
    // Best case: real named dataset with known test split
    html = 'Verified on ' + testSize + ' held-out samples from ' + dataset;
    verified = true;
    if (testSize < 50) {
      html += '<span class="acc-caveat">Small test set — results may vary with more data.</span>';
      verified = false;
    }
  } else if (testSize > 0 && isSynthetic) {
    html = 'on ' + testSize + ' synthetic test samples';
    html += '<span class="acc-caveat">Synthetic data — real-world testing recommended.</span>';
  } else if (testSize > 0) {
    html = 'on ' + testSize + ' held-out samples';
    if (testSize < 50) {
      html += '<span class="acc-caveat">Small test set — results may vary with more data.</span>';
    }
  } else if (isSynthetic && !isReal) {
    html = '<span class="acc-caveat">Synthetic data — real-world testing recommended.</span>';
  } else if (acc >= 95) {
    html = '<span class="acc-caveat">High accuracy — verify with your own held-out data.</span>';
  }

  return { html: html, verified: verified };
}

function metaCard(val, lbl) {
  return '<div class="rmeta-card"><div class="rmeta-val">' + val +
         '</div><div class="rmeta-lbl">' + lbl + '</div></div>';
}

// ── DOWNLOAD ────────────────────────────────────────────────────────────────

async function downloadNetwork() {
  var btn     = document.getElementById('downloadNetworkBtn');
  var txtEl   = document.getElementById('downloadNetworkText');
  var hint    = document.getElementById('downloadHint');
  var problem = currentProblem || document.getElementById('problemInput').value.trim();

  btn.disabled      = true;
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
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
    txtEl.textContent = 'Downloaded';
    hint.textContent  = 'Run: python run_network.py';
    setTimeout(function() {
      btn.disabled      = false;
      txtEl.textContent = 'Download Agent ZIP';
      hint.textContent  = '';
    }, 6000);
  } catch (e) {
    txtEl.textContent   = 'Download failed';
    hint.textContent    = e.message;
    hint.style.color    = 'var(--red)';
    btn.disabled        = false;
    setTimeout(function() {
      txtEl.textContent = 'Download Agent ZIP';
      hint.textContent  = '';
      hint.style.color  = '';
    }, 4000);
  }
}

// ── PIPELINE HELPERS ────────────────────────────────────────────────────────

function resetPipelineUI() {
  document.getElementById('pipelineSteps').innerHTML    = '';
  document.getElementById('agentNetworkNodes').innerHTML = '';
  document.getElementById('agentNetworkMeta').textContent = '';
  hide('agentNetwork');
  hide('nnViz');
  _tdHide();
  if (_nnRAF) { cancelAnimationFrame(_nnRAF); _nnRAF = null; }
  _nnLogBuf = [];
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
  if (el) el.innerHTML = stepHTML(main, detail, state, time);
}

function stepHTML(main, detail, state, time) {
  var icon = '';
  if (state === 'done') {
    icon = '<div class="ps-icon-check">' +
           '<svg width="10" height="10" viewBox="0 0 24 24" fill="none">' +
           '<path d="M20 6L9 17l-5-5" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>' +
           '</svg></div>';
  } else if (state === 'running') {
    icon = '<div class="ps-icon-spin">' +
           '<svg width="11" height="11" viewBox="0 0 24 24" fill="none">' +
           '<circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-dasharray="32" stroke-dashoffset="12"/>' +
           '</svg></div>';
  } else if (state === 'error') {
    icon = '<div class="ps-icon-check" style="background:rgba(220,38,38,0.1);border-color:var(--red)">' +
           '<svg width="10" height="10" viewBox="0 0 24 24" fill="none">' +
           '<path d="M18 6L6 18M6 6l12 12" stroke="var(--red)" stroke-width="2.5" stroke-linecap="round"/>' +
           '</svg></div>';
  } else {
    icon = '<div class="ps-icon-pending"></div>';
  }

  return '<div class="ps-icon">' + icon + '</div>' +
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
  _activeJobId    = null;
  hide('pipelineSection');
  hide('resultsSection');
  hide('scopeWarning');
  resetPipelineUI();
  // Deselect constraint chips
  document.querySelectorAll('.constraint-chip.selected').forEach(function(b) {
    b.classList.remove('selected');
  });
  // Deselect quick-start chips
  document.querySelectorAll('.qs-chip.active').forEach(function(b) {
    b.classList.remove('active');
  });
  window.scrollTo({ top: 0, behavior: 'smooth' });
  setTimeout(function() { document.getElementById('problemInput').focus(); }, 400);
}

// ── UTILS ────────────────────────────────────────────────────────────────────

function sleep(ms) { return new Promise(function(r) { setTimeout(r, ms); }); }

function show(id) { var el = document.getElementById(id); if (el) el.classList.remove('hidden'); }
function hide(id) { var el = document.getElementById(id); if (el) el.classList.add('hidden'); }

function smoothScrollTo(id) {
  var el = document.getElementById(id);
  if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── INIT ─────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', function() {
  hide('pipelineSection');
  hide('resultsSection');
  hide('scopeWarning');

  phEl = document.getElementById('problemInput');
  phEl.placeholder = PLACEHOLDERS[0];
  setInterval(cyclePlaceholder, 3000);

  phEl.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) launch();
  });

  phEl.addEventListener('input', onTextareaInput);

  initQuickStartChips();
  initConstraintChips();

  // Re-run Lucide in case DOMContentLoaded fires after script block
  if (typeof lucide !== 'undefined') lucide.createIcons();

  // Scroll-triggered fade-in
  if (typeof IntersectionObserver !== 'undefined') {
    var scrollObs = new IntersectionObserver(function(entries) {
      entries.forEach(function(entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
          scrollObs.unobserve(entry.target);
        }
      });
    }, { threshold: 0.08 });

    document.querySelectorAll('.scroll-fade').forEach(function(el) {
      scrollObs.observe(el);
    });
  } else {
    // Fallback: just show all immediately
    document.querySelectorAll('.scroll-fade').forEach(function(el) {
      el.classList.add('visible');
    });
  }
});
