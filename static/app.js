/* ==========================================================================
   WESAD Stress Detection – SPA Client
   No Jinja2 — all rendering done client-side via vanilla JS.
   ========================================================================== */

(function () {
  'use strict';

  /* ── cached DOM refs ──────────────────────────────────────────────── */
  var $home    = document.getElementById('page-home');
  var $result  = document.getElementById('page-result');
  var $compare = document.getElementById('page-compare');

  var $alertBox       = document.getElementById('alertBox');
  var $modelsBanner   = document.getElementById('modelsInfoBanner');
  var $modelsListText = document.getElementById('modelsListText');
  var $manualBanner   = document.getElementById('manualInfoBanner');
  var $chestGrid      = document.getElementById('chestGrid');
  var $wristGrid      = document.getElementById('wristGrid');
  var $labelGrid      = document.getElementById('labelGrid');
  var $subjectSelect  = document.getElementById('subject_id');
  var $uploadTracking = document.getElementById('tracking_id_upload');
  var $manualTracking = document.getElementById('tracking_id_manual');

  var $uploadForm = document.getElementById('uploadForm');
  var $manualForm = document.getElementById('manualForm');
  var $compareBtn = document.getElementById('compareBtn');
  var $uploadBtn  = document.getElementById('uploadBtn');
  var $manualBtn  = document.getElementById('manualBtn');

  var $fileInput    = document.getElementById('pkl_file');
  var $dropZone     = document.getElementById('dropZone');
  var $dropPrompt   = document.getElementById('dropPrompt');
  var $dropSelected = document.getElementById('dropSelected');
  var $fileNameText = document.getElementById('fileNameText');
  var $clearFile    = document.getElementById('clearFile');

  var $progressBar  = document.getElementById('progressBar');
  var $savedSection = document.getElementById('savedSection');
  var $savedList    = document.getElementById('savedFilesList');
  var $themeToggle = document.getElementById('themeToggle');
  var $toastContainer = document.getElementById('toastContainer');
  var $scrollTop = document.getElementById('scrollTop');

  /* ── app state ────────────────────────────────────────────────────── */
  var CFG = null; // populated by /api/config
  var _charts = []; // active Chart.js instances — destroyed before re-render
  var _historyChart = null; // dedicated history chart instance (re-created on filter)
  var RESULTS_CACHE = {}; // pre-fetched prediction results keyed by subject ID
  var HISTORY_CACHE = {}; // fetched history keyed by tracking ID
  var ACTIVE_HISTORY = { trackingId: '', start: '', end: '' };

  /* ── theme toggling ───────────────────────────────────────────────── */
  function getTheme() { return localStorage.getItem('wesad-theme') || 'dark'; }
  function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('wesad-theme', theme);
    var meta = document.getElementById('metaThemeColor');
    if (meta) meta.setAttribute('content', theme === 'light' ? '#f4f2ff' : '#060612');
  }
  setTheme(getTheme());
  if ($themeToggle) {
    $themeToggle.addEventListener('click', function () {
      setTheme(getTheme() === 'dark' ? 'light' : 'dark');
    });
  }

  /* ── toast notification system ────────────────────────────────────── */
  var _toastQueue = [];
  var TOAST_ICONS = {
    success: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
    danger: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
    warning: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    info: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
  };
  function showToast(msg, cat, duration) {
    cat = cat || 'info';
    duration = duration || 5000;
    if (!$toastContainer) return;
    var toast = document.createElement('div');
    toast.className = 'toast toast-' + cat;
    toast.innerHTML =
      '<span class="toast-icon">' + (TOAST_ICONS[cat] || TOAST_ICONS.info) + '</span>' +
      '<span class="toast-body">' + esc(msg) + '</span>' +
      '<button class="toast-close" aria-label="Close">&times;</button>' +
      '<div class="toast-progress" style="width:100%;"></div>';
    $toastContainer.appendChild(toast);
    _toastQueue.push(toast);
    if (_toastQueue.length > 4) {
      var oldest = _toastQueue.shift();
      dismissToast(oldest);
    }
    var progress = toast.querySelector('.toast-progress');
    progress.style.transitionDuration = duration + 'ms';
    requestAnimationFrame(function () { progress.style.width = '0%'; });
    toast.querySelector('.toast-close').addEventListener('click', function () { dismissToast(toast); });
    var timer = setTimeout(function () { dismissToast(toast); }, duration);
    toast._timer = timer;
  }
  function dismissToast(toast) {
    if (!toast || !toast.parentNode) return;
    clearTimeout(toast._timer);
    toast.classList.add('toast-out');
    setTimeout(function () { if (toast.parentNode) toast.parentNode.removeChild(toast); }, 300);
    var idx = _toastQueue.indexOf(toast);
    if (idx >= 0) _toastQueue.splice(idx, 1);
  }

  /* ── scroll-to-top button ─────────────────────────────────────────── */
  if ($scrollTop) {
    window.addEventListener('scroll', function () {
      if (window.scrollY > 400) $scrollTop.classList.add('visible');
      else $scrollTop.classList.remove('visible');
    }, { passive: true });
    $scrollTop.addEventListener('click', function () {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  }

  /* ── helpers ──────────────────────────────────────────────────────── */
  function esc(s) {
    var d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }
  function pct(v) { return (v * 100).toFixed(1); }
  function formatDateTime(value) {
    if (!value) return 'Unknown time';
    var dt = new Date(value);
    if (isNaN(dt.getTime())) return value;
    return dt.toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });
  }
  function formatDelta(delta) {
    if (delta == null) return 'No previous entry yet';
    var sign = delta > 0 ? '+' : '';
    return sign + pct(delta) + '% vs previous entry';
  }
  function formatDeltaHTML(delta, trend) {
    if (delta == null) return '<span class="trend-arrow trend-stable">No previous entry</span>';
    var sign = delta > 0 ? '+' : '';
    var arrowSvg = trend === 'up'
      ? '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="18 15 12 9 6 15"/></svg>'
      : trend === 'down'
        ? '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 9 12 15 18 9"/></svg>'
        : '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="5" y1="12" x2="19" y2="12"/></svg>';
    return '<span class="trend-arrow trend-' + (trend || 'stable') + '">' + arrowSvg + ' ' + sign + pct(delta) + '% vs previous</span>';
  }
  function toHistoryIso(value, isEnd) {
    if (!value) return '';
    var normalized = value.length === 16 ? value + ':00' : value;
    var date = new Date(normalized);
    if (isNaN(date.getTime())) return '';
    if (isEnd) date.setSeconds(59, 999);
    return date.toISOString();
  }
  function toLocalDateTimeInputValue(isoValue) {
    if (!isoValue) return '';
    var date = new Date(isoValue);
    if (isNaN(date.getTime())) return '';
    // datetime-local inputs expect local wall-clock time (no timezone suffix).
    return new Date(date.getTime() - date.getTimezoneOffset() * 60000)
      .toISOString()
      .slice(0, 16);
  }
  function buildHistoryQuery(start, end, format) {
    var params = new URLSearchParams();
    if (start) params.set('start', start);
    if (end) params.set('end', end);
    if (format) params.set('format', format);
    var qs = params.toString();
    return qs ? ('?' + qs) : '';
  }
  function showPage(page) {
    $home.style.display    = page === 'home'    ? '' : 'none';
    $result.style.display  = page === 'result'  ? '' : 'none';
    $compare.style.display = page === 'compare' ? '' : 'none';
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }
  function showAlert(msg, cat) {
    cat = cat || 'danger';
    var icon = cat === 'danger'
      ? '<svg class="alert-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>'
      : '<svg class="alert-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>';
    $alertBox.innerHTML = '<div class="alert alert-' + cat + '">' + icon + '<span>' + esc(msg) + '</span></div>';
    $alertBox.style.display = '';
    setTimeout(function () { $alertBox.style.display = 'none'; }, 8000);
    showToast(msg, cat);
  }
  function setLoading(btn, on) {
    if (on) { btn.classList.add('loading'); btn.disabled = true; }
    else    { btn.classList.remove('loading'); btn.disabled = false; }
  }

  function destroyCharts() {
    _charts.forEach(function (c) { try { c.destroy(); } catch (_) {} });
    _charts = [];
    if (_historyChart) { try { _historyChart.destroy(); } catch (_) {} _historyChart = null; }
    document.querySelectorAll('.tooltip').forEach(function (t) { t.remove(); });
  }

  function startProgress() {
    $progressBar.classList.remove('done');
    $progressBar.classList.add('active');
  }
  function stopProgress() {
    $progressBar.classList.remove('active');
    $progressBar.classList.add('done');
    setTimeout(function () { $progressBar.classList.remove('done'); }, 600);
  }

  /* ── initialise from /api/config ──────────────────────────────────── */
  function init() {
    fetch('/api/config')
      .then(function (r) { return r.json(); })
      .then(function (cfg) {
        CFG = cfg;
        buildSubjectDropdown(cfg.subjects, cfg.general_model_ready);
        buildSensorInputs(cfg.feature_columns, cfg.sensor_meta);
        buildLabelGrid(cfg.label_map);
        if (cfg.subjects.length) {
          $modelsBanner.style.display = '';
          var cache = cfg.subject_cache || {};
          var instant = cfg.instant_subjects || [];
          $modelsListText.innerHTML = '<span class="subject-chips">' +
            cfg.subjects.map(function (s) {
              var cached = cache[s] ? ' has-cache' : '';
              var isInstant = instant.indexOf(s) >= 0 ? ' instant' : '';
              return '<span class="subject-chip' + cached + isInstant + '" data-sid="' + esc(s) + '" title="Click to analyse instantly">' + esc(s) + '</span>';
            }).join('') +
            '</span>';
          buildChipActions();
        }
        $manualBanner.style.display = '';
        loadSavedFiles();
        // Prefetch all pre-computed results for instant single-click access
        prefetchResults();
      })
      .catch(function (err) { showAlert('Failed to load configuration: ' + err); });
  }

  function prefetchResults() {
    fetch('/api/batch-results')
      .then(function (r) { return r.json(); })
      .then(function (data) {
        RESULTS_CACHE = data || {};
        // Mark chips that now have instant results
        Object.keys(RESULTS_CACHE).forEach(function (sid) {
          var chip = $modelsListText.querySelector('.subject-chip[data-sid="' + sid + '"]');
          if (chip) chip.classList.add('instant');
        });
      })
      .catch(function () { /* silently ignore — chips still work via API */ });
  }

  function refreshSubjectChips() {
    fetch('/api/config')
      .then(function (r) { return r.json(); })
      .then(function (cfg) {
        CFG = cfg;
        if (cfg.subjects.length) {
          $modelsBanner.style.display = '';
          var cache = cfg.subject_cache || {};
          var instant = cfg.instant_subjects || [];
          $modelsListText.innerHTML = '<span class="subject-chips">' +
            cfg.subjects.map(function (s) {
              var cached = cache[s] ? ' has-cache' : '';
              var isInstant = (instant.indexOf(s) >= 0 || RESULTS_CACHE[s]) ? ' instant' : '';
              return '<span class="subject-chip' + cached + isInstant + '" data-sid="' + esc(s) + '" title="Click to analyse instantly">' + esc(s) + '</span>';
            }).join('') +
            '</span>';
          buildChipActions();
        }
      })
      .catch(function () {});
  }

  function buildSubjectDropdown(subjects, generalReady) {
    $subjectSelect.innerHTML = '';
    /* Default option: general model (subject-independent) */
    var def = document.createElement('option');
    def.value = '';
    def.selected = true;
    def.textContent = generalReady
      ? '\u2728 General Model (recommended)'
      : 'General model not available';
    $subjectSelect.appendChild(def);
    /* Per-subject overrides */
    if (subjects.length) {
      var sep = document.createElement('option');
      sep.disabled = true;
      sep.textContent = '\u2500\u2500 or pick a subject-specific model \u2500\u2500';
      $subjectSelect.appendChild(sep);
      subjects.forEach(function (sid) {
        var o = document.createElement('option');
        o.value = sid; o.textContent = sid;
        $subjectSelect.appendChild(o);
      });
    }
  }

  function buildSensorInputs(columns, meta) {
    var chestCols = columns.slice(0, 8);
    var wristCols = columns.slice(8);
    $chestGrid.innerHTML = chestCols.map(function (col) { return sensorFieldHTML(col, meta[col]); }).join('');
    $wristGrid.innerHTML = wristCols.map(function (col) { return sensorFieldHTML(col, meta[col]); }).join('');
  }

  function sensorFieldHTML(col, m) {
    m = m || {};
    var unit = m.unit ? ' <span class="unit-badge">' + esc(m.unit) + '</span>' : '';
    return '<div class="form-group">' +
      '<label for="' + esc(col) + '">' + esc(col) + unit + '</label>' +
      '<input type="text" inputmode="decimal" pattern="-?[0-9]*\\.?[0-9]*"' +
      ' id="' + esc(col) + '" name="' + esc(col) + '"' +
      ' placeholder="' + esc(m.hint || '0') + '"' +
      ' data-low="' + (m.low || 0) + '"' +
      ' data-high="' + (m.high || 0) + '"' +
      ' data-critical="' + (m.critical || 0) + '"' +
      ' required>' +
      '</div>';
  }

  function buildLabelGrid(labelMap) {
    var html = '';
    Object.keys(labelMap).forEach(function (key) {
      var val = labelMap[key];
      var cls = 'lc-' + val.toLowerCase().replace(/ /g, '-');
      html += '<div class="label-chip ' + cls + '">' +
        '<span class="label-key">' + esc(key) + '</span>' +
        '<span class="label-val">' + esc(val) + '</span>' +
        (key === '2' ? '<span class="badge stress">Stress</span>' : '') +
        '</div>';
    });
    $labelGrid.innerHTML = html;
  }

  /* ── input range colour feedback ──────────────────────────────────── */
  document.addEventListener('input', function (e) {
    var inp = e.target;
    if (inp.tagName !== 'INPUT' || !inp.closest('.sensor-grid')) return;
    inp.classList.remove('in-baseline', 'in-stress', 'in-critical');
    var v = parseFloat(inp.value);
    if (isNaN(v)) return;
    var crit = parseFloat(inp.getAttribute('data-critical')) || 0;
    var high = parseFloat(inp.getAttribute('data-high')) || 0;
    if (crit && v >= crit) { inp.classList.add('in-critical'); }
    else if (high && v >= high) { inp.classList.add('in-stress'); }
    else { inp.classList.add('in-baseline'); }
  });

  /* ── file upload drop-zone ────────────────────────────────────────── */
  function showFile(name) {
    $fileNameText.textContent = name;
    $dropPrompt.style.display = 'none';
    $dropSelected.style.display = 'flex';
    $dropZone.classList.add('has-file');
  }
  function resetFile() {
    $fileInput.value = '';
    $dropPrompt.style.display = '';
    $dropSelected.style.display = 'none';
    $dropZone.classList.remove('has-file');
  }
  $fileInput.addEventListener('change', function () {
    if (this.files.length) showFile(this.files[0].name); else resetFile();
  });
  $clearFile.addEventListener('click', function (e) { e.preventDefault(); e.stopPropagation(); resetFile(); });
  ['dragenter', 'dragover'].forEach(function (ev) {
    $dropZone.addEventListener(ev, function (e) { e.preventDefault(); $dropZone.classList.add('drag-over'); });
  });
  ['dragleave', 'drop'].forEach(function (ev) {
    $dropZone.addEventListener(ev, function (e) { e.preventDefault(); $dropZone.classList.remove('drag-over'); });
  });
  $dropZone.addEventListener('drop', function (e) {
    if (e.dataTransfer.files.length) { $fileInput.files = e.dataTransfer.files; showFile(e.dataTransfer.files[0].name); }
  });
  $dropZone.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $fileInput.click(); }
  });

  /* ── presets ──────────────────────────────────────────────────────── */
  document.querySelectorAll('.preset-btn').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var preset = this.getAttribute('data-preset');
      document.querySelectorAll('#manualForm .sensor-grid input').forEach(function (inp) {
        inp.value = preset === 'clear' ? '' : (inp.getAttribute('data-' + preset) || '0');
      });
      document.querySelectorAll('.preset-btn').forEach(function (b) { b.classList.remove('active'); });
      this.classList.add('active');
    });
  });

  /* ── saved files ────────────────────────────────────────────────── */
  function loadSavedFiles() {
    fetch('/api/saved-files')
      .then(function (r) { return r.json(); })
      .then(function (data) {
        var files = data.files || [];
        if (!files.length) { $savedSection.style.display = 'none'; return; }
        $savedSection.style.display = '';
        $savedList.innerHTML = files.map(function (f) {
          var badges = '';
          if (f.has_model) badges += '<span class="saved-file-badge has-model">Model ready</span>';
          if (f.has_cache) badges += '<span class="saved-file-badge has-cache">Cached</span>';
          return '<div class="saved-file-card" data-sid="' + esc(f.subject_id) + '">' +
            '<div class="saved-file-name">' +
              '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--accent-light)" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg> ' +
              esc(f.filename) +
            '</div>' +
            '<div class="saved-file-meta">' + esc(f.size_mb + ' MB') + ' ' + badges + '</div>' +
            '<div class="saved-file-actions">' +
              '<button class="btn btn-analyse" data-sid="' + esc(f.subject_id) + '">Analyse</button>' +
              '<button class="btn btn-compare-saved" data-sid="' + esc(f.subject_id) + '">Compare</button>' +
              '<button class="btn btn-delete-saved" data-sid="' + esc(f.subject_id) + '" title="Remove saved file">&times;</button>' +
            '</div>' +
          '</div>';
        }).join('');
        bindSavedFileButtons();
      })
      .catch(function () { /* silently ignore */ });
  }

  function bindSavedFileButtons() {
    $savedList.querySelectorAll('.btn-analyse').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var sid = this.getAttribute('data-sid');
        // Instant path: render from client-side cache
        if (RESULTS_CACHE[sid]) {
          renderResult(RESULTS_CACHE[sid]);
          return;
        }
        setLoading(this, true);
        startProgress();
        var self = this;
        fetch('/api/analyze/' + encodeURIComponent(sid), { method: 'POST' })
          .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, data: d }; }); })
          .then(function (res) {
            setLoading(self, false); stopProgress();
            if (!res.ok) { showAlert(res.data.error || 'Analysis failed.'); return; }
            RESULTS_CACHE[sid] = res.data;
            renderResult(res.data);
          })
          .catch(function (err) { setLoading(self, false); stopProgress(); showAlert('Network error: ' + err); });
      });
    });

    $savedList.querySelectorAll('.btn-compare-saved').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var sid = this.getAttribute('data-sid');
        setLoading(this, true);
        startProgress();
        var self = this;
        fetch('/api/compare-saved/' + encodeURIComponent(sid), { method: 'POST' })
          .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, data: d }; }); })
          .then(function (res) {
            setLoading(self, false); stopProgress();
            if (!res.ok) { showAlert(res.data.error || 'Comparison failed.'); return; }
            renderCompare(res.data);
          })
          .catch(function (err) { setLoading(self, false); stopProgress(); showAlert('Network error: ' + err); });
      });
    });

    $savedList.querySelectorAll('.btn-delete-saved').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var sid = this.getAttribute('data-sid');
        if (!confirm('Delete saved file ' + sid + '.pkl?')) return;
        fetch('/api/saved-files/' + encodeURIComponent(sid), { method: 'DELETE' })
          .then(function () { loadSavedFiles(); })
          .catch(function (err) { showAlert('Delete failed: ' + err); });
      });
    });
  }

  /* ── pre-trained model chip actions ─────────────────────────────── */
  var _activePopover = null;
  function closeChipPopover() {
    if (_activePopover) { _activePopover.remove(); _activePopover = null; }
    $modelsListText.querySelectorAll('.subject-chip').forEach(function (c) { c.classList.remove('active'); });
  }

  document.addEventListener('click', function (e) {
    if (_activePopover && !_activePopover.contains(e.target) && !e.target.classList.contains('subject-chip')) {
      closeChipPopover();
    }
  });

  function buildChipActions() {
    $modelsListText.querySelectorAll('.subject-chip').forEach(function (chip) {
      // Single click = instant analyse
      chip.addEventListener('click', function (e) {
        e.stopPropagation();
        closeChipPopover();
        var sid = this.getAttribute('data-sid');
        // Shift+click opens compare instead
        if (e.shiftKey) {
          comparePretrainedSubject(sid);
          return;
        }
        analysePretrainedSubject(sid);
      });
      // Right-click / context menu = compare
      chip.addEventListener('contextmenu', function (e) {
        e.preventDefault();
        e.stopPropagation();
        closeChipPopover();
        var sid = this.getAttribute('data-sid');
        comparePretrainedSubject(sid);
      });
    });
  }

  function analysePretrainedSubject(sid) {
    // Instant path: render from client-side cache (zero network latency)
    if (RESULTS_CACHE[sid]) {
      renderResult(RESULTS_CACHE[sid]);
      return;
    }
    // Fallback: fetch from server (still fast if server cache is warm)
    startProgress();
    showAlert('Analysing ' + sid + '\u2026', 'success');
    fetch('/api/analyze-pretrained/' + encodeURIComponent(sid), { method: 'POST' })
      .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, data: d }; }); })
      .then(function (res) {
        stopProgress();
        if (!res.ok) { showAlert(res.data.error || 'Analysis failed.'); return; }
        RESULTS_CACHE[sid] = res.data; // cache for next click
        renderResult(res.data);
      })
      .catch(function (err) { stopProgress(); showAlert('Network error: ' + err); });
  }

  function comparePretrainedSubject(sid) {
    startProgress();
    showAlert('Comparing classifiers for ' + sid + '…', 'success');
    fetch('/api/compare-pretrained/' + encodeURIComponent(sid), { method: 'POST' })
      .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, data: d }; }); })
      .then(function (res) {
        stopProgress();
        if (!res.ok) { showAlert(res.data.error || 'Comparison failed.'); return; }
        renderCompare(res.data);
      })
      .catch(function (err) { stopProgress(); showAlert('Network error: ' + err); });
  }

  /* ── upload & analyse ─────────────────────────────────────────────── */
  $uploadForm.addEventListener('submit', function (e) {
    e.preventDefault();
    if (!$fileInput.files.length) { showAlert('Please select a WESAD subject .pkl file.'); return; }
    var fd = new FormData();
    fd.append('pkl_file', $fileInput.files[0]);
    if ($uploadTracking && $uploadTracking.value.trim()) {
      fd.append('tracking_id', $uploadTracking.value.trim());
    }
    setLoading($uploadBtn, true);
    startProgress();
    fetch('/api/upload', { method: 'POST', body: fd })
      .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, data: d }; }); })
      .then(function (res) {
        setLoading($uploadBtn, false); stopProgress();
        if (!res.ok) { showAlert(res.data.error || 'Upload failed.'); return; }
        // Cache result for instant access and refresh subjects list
        if (res.data.subject_id) RESULTS_CACHE[res.data.subject_id] = res.data;
        loadSavedFiles();
        refreshSubjectChips();
        renderResult(res.data);
      })
      .catch(function (err) { setLoading($uploadBtn, false); stopProgress(); showAlert('Network error: ' + err); });
  });

  /* ── compare 6 models ─────────────────────────────────────────────── */
  $compareBtn.addEventListener('click', function () {
    if (!$fileInput.files.length) { showAlert('Please select a WESAD subject .pkl file first.'); return; }
    var fd = new FormData();
    fd.append('pkl_file', $fileInput.files[0]);
    setLoading($compareBtn, true);
    startProgress();
    fetch('/api/compare', { method: 'POST', body: fd })
      .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, data: d }; }); })
      .then(function (res) {
        setLoading($compareBtn, false); stopProgress();
        if (!res.ok) { showAlert(res.data.error || 'Comparison failed.'); return; }
        renderCompare(res.data);
      })
      .catch(function (err) { setLoading($compareBtn, false); stopProgress(); showAlert('Network error: ' + err); });
  });

  /* ── manual predict ───────────────────────────────────────────────── */
  $manualForm.addEventListener('submit', function (e) {
    e.preventDefault();
    var sid = $subjectSelect.value;  /* empty string = general model */
    var sensors = {};
    if (CFG) {
      CFG.feature_columns.forEach(function (col) {
        var inp = document.getElementById(col);
        sensors[col] = inp ? parseFloat(inp.value) || 0 : 0;
      });
    }
    var payload = { sensors: sensors };
    if (sid) { payload.subject_id = sid; } /* optional override */
    if ($manualTracking && $manualTracking.value.trim()) { payload.tracking_id = $manualTracking.value.trim(); }
    setLoading($manualBtn, true);
    startProgress();
    fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
      .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, data: d }; }); })
      .then(function (res) {
        setLoading($manualBtn, false); stopProgress();
        if (!res.ok) { showAlert(res.data.error || 'Prediction failed.'); return; }
        renderResult(res.data);
      })
      .catch(function (err) { setLoading($manualBtn, false); stopProgress(); showAlert('Network error: ' + err); });
  });

  /* ================================================================
     RENDER: RESULT PAGE
     ================================================================ */
  function renderResult(d) {
    destroyCharts();
    var isManual = d.is_manual;
    var stressed = d.overall_stress;
    var color    = d.stress_level_color;
    var ratio    = d.stress_ratio;
    var capturedAt = d.captured_at ? formatDateTime(d.captured_at) : '';

    var verdictIcon = stressed
      ? '<svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="var(--red)" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
      : '<svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="var(--green)" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>';

    var verdictTitle = isManual
      ? 'Predicted: <strong>' + esc(d.label_names[0]) + '</strong>'
      : (stressed ? 'Stress Detected' : 'No Significant Stress');
    var verdictMeta = isManual
      ? esc(d.subject_id) + ' &mdash; single reading classification'
      : 'Subject <strong>' + esc(d.subject_id) + '</strong> &mdash; ' + esc(d.method) + ' model';
    if (capturedAt) verdictMeta += ' &mdash; recorded ' + esc(capturedAt);

    var gaugeUnit = isManual ? 'stress confidence' : 'stress ratio';
    var trackingHTML = '';
    if (d.tracking_id) {
      trackingHTML =
        '<section class="card animate-in">' +
          '<div class="history-meta">' +
            '<span class="history-meta-id">History ID: <strong>' + esc(d.tracking_id) + '</strong></span>' +
            '<span class="history-meta-count">' + esc(String(d.history_length || 1)) + ' recorded entr' + ((d.history_length || 1) === 1 ? 'y' : 'ies') + '</span>' +
          '</div>' +
        '</section>';
    }

    /* summary cards */
    var summaryCards = '';
    if (isManual) {
      summaryCards =
        '<div class="stat-grid">' +
          '<div class="stat-card"><span class="stat-num" style="color:' + color + ';">' + pct(ratio) + '%</span><span class="stat-lbl">Stress Confidence</span></div>' +
          '<div class="stat-card"><span class="stat-num">' + esc(d.label_names[0]) + '</span><span class="stat-lbl">Predicted State</span></div>' +
        '</div>' +
        '<div class="info-banner" style="margin-top:1rem;">' +
          '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>' +
          '<span>This is a single-reading prediction. For more reliable results, upload a full <code>.pkl</code> recording file.</span>' +
        '</div>';
    } else {
      summaryCards =
        '<div class="stat-grid">' +
          '<div class="stat-card stat-card--total"><span class="stat-num">' + d.total_windows + '</span><span class="stat-lbl">Windows</span></div>' +
          '<div class="stat-card stat-card--ratio"><span class="stat-num" style="color:' + color + ';">' + pct(ratio) + '%</span><span class="stat-lbl">Stress Ratio</span></div>' +
          '<div class="stat-card stat-card--stress"><span class="stat-num stress-num">' + (d.label_counts.Stress || 0) + '</span><span class="stat-lbl">Stress Windows</span></div>' +
          '<div class="stat-card stat-card--baseline"><span class="stat-num baseline-num">' + (d.label_counts.Baseline || 0) + '</span><span class="stat-lbl">Baseline Windows</span></div>' +
        '</div>';
    }

    /* distribution / timeline / table (file-upload only) */
    var tabsHTML = '';
    var historyHTML = '';
    if (!isManual) {
      var total = d.total_windows || 1;

      /* distribution bars */
      var bars = '';
      Object.keys(d.label_counts).forEach(function (label) {
        var count = d.label_counts[label];
        var p = (count / total * 100).toFixed(1);
        bars += '<div class="bar-row">' +
          '<span class="bar-name">' + esc(label) + '</span>' +
          '<div class="bar-track"><div class="bar-fill bar-' + label.toLowerCase().replace(/ /g, '-') + '" data-width="' + p + '" style="width:0%;"></div></div>' +
          '<span class="bar-value">' + count + '<small> (' + p + '%)</small></span></div>';
      });

      /* timeline blocks */
      var timeline = '';
      for (var i = 0; i < d.predictions.length; i++) {
        var cls = 'tl-' + d.label_names[i].toLowerCase().replace(/ /g, '-');
        timeline += '<div class="tl-block ' + cls + '" data-tip="Window ' + (i + 1) + ': ' + esc(d.label_names[i]) + '"></div>';
      }

      /* table rows */
      var trows = '';
      Object.keys(d.label_counts).forEach(function (label) {
        var count = d.label_counts[label];
        var badge = label === 'Stress' ? '<span class="badge stress">Stress</span>'
                  : label === 'Baseline' ? '<span class="badge baseline">OK</span>' : '';
        trows += '<tr><td>' + esc(label) + ' ' + badge + '</td><td>' + count + '</td><td>' + (count / total * 100).toFixed(1) + '%</td></tr>';
      });

      tabsHTML =
        '<section class="card animate-in">' +
          '<div class="tabs" role="tablist">' +
            '<button class="tab-btn active" data-tab="distribution" role="tab" aria-selected="true">' +
              '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="12" width="4" height="9"/><rect x="10" y="7" width="4" height="14"/><rect x="17" y="2" width="4" height="19"/></svg> Distribution</button>' +
            '<button class="tab-btn" data-tab="timeline" role="tab" aria-selected="false">' +
              '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="17" y1="10" x2="3" y2="10"/><line x1="21" y1="6" x2="3" y2="6"/><line x1="21" y1="14" x2="3" y2="14"/><line x1="17" y1="18" x2="3" y2="18"/></svg> Timeline</button>' +
            '<button class="tab-btn" data-tab="table" role="tab" aria-selected="false">' +
              '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="3" x2="9" y2="21"/></svg> Table</button>' +
          '</div>' +
          '<div class="tab-panel active" id="tab-distribution" role="tabpanel"><div class="bar-chart">' + bars + '</div></div>' +
          '<div class="tab-panel" id="tab-timeline" role="tabpanel">' +
            '<p class="timeline-info">Each block = one ' + d.window_sec + 's window. Hover for details.</p>' +
            '<div class="timeline-track">' + timeline + '</div>' +
            '<div class="timeline-legend">' +
              '<span class="legend-item"><span class="legend-dot" style="background:var(--green);"></span> Baseline</span>' +
              '<span class="legend-item"><span class="legend-dot" style="background:var(--red);"></span> Stress</span>' +
              '<span class="legend-item"><span class="legend-dot" style="background:var(--orange);"></span> Amusement</span>' +
              '<span class="legend-item"><span class="legend-dot" style="background:var(--blue, #60a5fa);"></span> Meditation</span>' +
            '</div>' +
          '</div>' +
          '<div class="tab-panel" id="tab-table" role="tabpanel">' +
            '<table class="data-table"><thead><tr><th>State</th><th>Windows</th><th>Percentage</th></tr></thead>' +
            '<tbody>' + trows + '</tbody></table>' +
          '</div>' +
        '</section>';
    }

    if (d.tracking_id) {
      historyHTML =
        '<section class="card animate-in history-card" id="historySection">' +
          '<h2 class="section-title">Tracking History</h2>' +
          '<div class="history-controls">' +
            '<div class="history-filter-grid">' +
              '<div class="form-group"><label for="historyStart">From</label><input type="datetime-local" id="historyStart"></div>' +
              '<div class="form-group"><label for="historyEnd">To</label><input type="datetime-local" id="historyEnd"></div>' +
            '</div>' +
            '<div class="history-actions">' +
              '<button class="btn btn-back history-btn" id="historyApplyBtn" type="button">Apply Filter</button>' +
              '<button class="btn btn-back history-btn" id="historyResetBtn" type="button">Clear Filter</button>' +
              '<button class="btn btn-back history-btn" id="historyExportCsvBtn" type="button">Export CSV</button>' +
            '</div>' +
          '</div>' +
          '<div class="info-banner" id="historySummary">Loading history for ' + esc(d.tracking_id) + '…</div>' +
          '<div class="history-chart-wrap" id="historyChartWrap" style="display:none;">' +
            '<canvas id="historyChart"></canvas>' +
          '</div>' +
          '<div class="history-list" id="historyEntries"></div>' +
        '</section>';
    }

    /* evaluation metrics (trained on-the-fly) */
    var metricsHTML = '';
    if (d.accuracy) {
      var rocCard = d.roc_auc != null
        ? '<div class="stat-card"><span class="stat-num" style="color:#c084fc;">' + pct(d.roc_auc) + '%</span><span class="stat-lbl">ROC AUC</span></div>'
        : '';
      var fsNote = d.feature_selection_applied
        ? '<div class="info-banner" style="margin-top:.5rem;"><span>Feature selection applied: ' + (d.n_features_used || '?') + ' features used.</span></div>'
        : '';
      metricsHTML =
        '<section class="card animate-in">' +
          '<h2 class="section-title">Model Evaluation &mdash; ' + esc(d.classifier_name || 'Random Forest') + '</h2>' +
          '<div class="stat-grid">' +
            '<div class="stat-card"><span class="stat-num" style="color:var(--accent-light);">' + pct(d.accuracy) + '%</span><span class="stat-lbl">Accuracy</span></div>' +
            '<div class="stat-card"><span class="stat-num" style="color:var(--green);">' + pct(d.f1_weighted) + '%</span><span class="stat-lbl">F1 (weighted)</span></div>' +
            '<div class="stat-card"><span class="stat-num" style="color:var(--orange);">' + pct(d.precision) + '%</span><span class="stat-lbl">Precision</span></div>' +
            '<div class="stat-card"><span class="stat-num" style="color:#60a5fa;">' + pct(d.recall) + '%</span><span class="stat-lbl">Recall</span></div>' +
            rocCard +
            (d.cv_mean ? '<div class="stat-card"><span class="stat-num" style="color:var(--accent-light);">' + pct(d.cv_mean) + '%</span><span class="stat-lbl">CV Mean &plusmn;' + pct(d.cv_std) + '%</span></div>' : '') +
          '</div>' +
          fsNote +
        '</section>';

      if (d.confusion_matrix && d.confusion_matrix.length) {
        metricsHTML += '<section class="card animate-in"><h2 class="section-title">Confusion Matrix</h2>' +
          '<div style="position:relative;max-width:380px;margin:0 auto;"><canvas id="cmChart" width="380" height="380"></canvas></div></section>';
      }
      if (d.feature_importance_labels && d.feature_importance_labels.length) {
        metricsHTML += '<section class="card animate-in"><h2 class="section-title">Top 15 Feature Importances</h2>' +
          '<div style="position:relative;max-width:700px;margin:0 auto;"><canvas id="featChart"></canvas></div></section>';
      }
    }

    /* assemble page */
    $result.innerHTML =
      '<header class="hero hero-sm animate-in"><h1>Analysis Results</h1></header>' +

      '<section class="verdict-card animate-in ' + (stressed ? 'stressed' : 'relaxed') + '">' +
        '<div class="verdict-icon">' + verdictIcon + '</div>' +
        '<h2>' + verdictTitle + '</h2>' +
        '<p class="verdict-meta">' + verdictMeta + '</p>' +
        '<div class="stress-level-pill" style="background:' + color + '20;color:' + color + ';border:1px solid ' + color + '44;">' + esc(d.stress_level) + '</div>' +
      '</section>' +

      trackingHTML +

      '<section class="card animate-in">' +
        '<h2 class="section-title">Stress Gauge</h2>' +
        '<div class="gauge-wrap">' +
          '<svg class="gauge-svg" viewBox="0 0 260 155">' +
            '<text x="22" y="152" class="gauge-mark">0%</text><text x="118" y="18" class="gauge-mark">50%</text><text x="228" y="152" class="gauge-mark">100%</text>' +
            '<path class="gauge-bg" d="M 30 140 A 100 100 0 0 1 230 140"/>' +
            '<path class="gauge-fill ' + d.gauge_class + '" id="gaugeFill" d="M 30 140 A 100 100 0 0 1 230 140" stroke-dasharray="314.16" stroke-dashoffset="314.16"/>' +
            '<text class="gauge-value" x="130" y="100" id="gaugeText">0%</text>' +
            '<text class="gauge-unit" x="130" y="125">' + gaugeUnit + '</text>' +
          '</svg>' +
          '<div class="stress-level-pill gauge-pill" style="background:' + color + '20;color:' + color + ';border:1px solid ' + color + '44;">' + esc(d.stress_level) + '</div>' +
        '</div>' +
      '</section>' +

      '<section class="card animate-in"><h2 class="section-title">Summary</h2>' + summaryCards + '</section>' +

  tabsHTML + historyHTML + metricsHTML +

      '<div class="actions animate-in">' +
        '<button class="btn btn-back" id="backBtn">' +
          '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="19" y1="12" x2="5" y2="12"/><polyline points="12 19 5 12 12 5"/></svg> Analyse Another' +
        '</button>' +
      '</div>' +
      '<footer class="footer animate-in"><p>WESAD Stress Detection &middot; Machine Learning Pipeline</p>' +
        '<div class="footer-links">' +
          '<a href="https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection" target="_blank" rel="noopener">' +
            '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg> WESAD Dataset</a>' +
          '<span style="color:var(--text-dim);">&copy; 2026</span>' +
        '</div></footer>';

    showPage('result');
    animateResult(d);
    if (d.tracking_id) {
      ACTIVE_HISTORY = { trackingId: d.tracking_id, start: '', end: '' };
      loadHistory(d.tracking_id);
    }
  }

  function loadHistory(trackingId, options) {
    options = options || {};
    if (!trackingId) return;
    var start = options.start || '';
    var end = options.end || '';
    var cacheKey = trackingId + '|' + start + '|' + end;
    ACTIVE_HISTORY = { trackingId: trackingId, start: start, end: end };
    if (HISTORY_CACHE[cacheKey]) {
      renderHistory(HISTORY_CACHE[cacheKey]);
      return;
    }
    fetch('/api/history/' + encodeURIComponent(trackingId) + buildHistoryQuery(start, end))
      .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, data: d }; }); })
      .then(function (res) {
        if (!res.ok) { showHistoryError(res.data.error || 'Unable to load history.'); return; }
        HISTORY_CACHE[cacheKey] = res.data;
        renderHistory(res.data);
      })
      .catch(function (err) { showHistoryError('Unable to load history: ' + err); });
  }

  function showHistoryError(message) {
    var summary = document.getElementById('historySummary');
    if (summary) summary.textContent = message;
  }

  function renderHistory(data) {
    var summary = document.getElementById('historySummary');
    var chartWrap = document.getElementById('historyChartWrap');
    var entriesWrap = document.getElementById('historyEntries');
    var startInput = document.getElementById('historyStart');
    var endInput = document.getElementById('historyEnd');
    if (!summary || !chartWrap || !entriesWrap) return;

    var entries = data.entries || [];
    var info = data.summary || {};
    if (!entries.length) {
      summary.textContent = 'No history has been stored for this ID yet.';
      entriesWrap.innerHTML = '';
      return;
    }

    summary.innerHTML =
      '<span><strong>' + esc(data.tracking_id) + '</strong> has ' + esc(String(info.count || entries.length)) + ' recorded entr' + ((info.count || entries.length) === 1 ? 'y' : 'ies') + '.</span>' +
      '<span>' + formatDeltaHTML(info.delta, info.trend) + '</span>';

    chartWrap.style.display = '';
    // Destroy previous history chart to prevent memory leak on re-filter
    if (_historyChart) { try { _historyChart.destroy(); } catch (_) {} _historyChart = null; }
    _historyChart = new Chart(document.getElementById('historyChart'), {
      type: 'line',
      data: {
        labels: entries.map(function (entry, index) {
          var dt = new Date(entry.captured_at || '');
          if (isNaN(dt.getTime())) return 'Entry ' + (index + 1);
          return dt.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
        }),
        datasets: [{
          label: 'Stress %',
          data: entries.map(function (entry) { return Number((entry.stress_ratio || 0) * 100); }),
          borderColor: 'rgba(124,92,252,1)',
          backgroundColor: 'rgba(124,92,252,0.18)',
          tension: 0.28,
          fill: true,
          pointRadius: 4,
          pointHoverRadius: 5,
        }]
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: '#8585a8' }, grid: { color: 'rgba(255,255,255,0.04)' } },
          y: {
            min: 0,
            max: 100,
            ticks: { color: '#8585a8', callback: function (value) { return value + '%'; } },
            grid: { color: 'rgba(255,255,255,0.05)' }
          }
        }
      }
    });

    entriesWrap.innerHTML = entries.slice().reverse().slice(0, 8).map(function (entry, index) {
      var stressed = entry.overall_stress;
      var borderCls = stressed ? ' entry-stressed' : ' entry-relaxed';
      var ratio = entry.stress_ratio || 0;
      return '<div class="history-entry' + borderCls + '">' +
        '<div>' +
          '<div class="history-entry-time">' + esc(formatDateTime(entry.captured_at)) + '</div>' +
          '<div class="history-entry-meta">' + esc(entry.predicted_label || entry.stress_level || 'Recorded') + ' &middot; ' + esc(entry.method || 'analysis') + '</div>' +
        '</div>' +
        '<div class="history-entry-score" style="color:' + (ratio > 0.4 ? 'var(--red)' : ratio > 0.2 ? 'var(--orange)' : 'var(--green)') + ';">' + pct(ratio) + '%</div>' +
      '</div>';
    }).join('');
  }

  /* ── post-render animations & bindings for result page ─────────── */
  function animateResult(d) {
    /* gauge */
    var gauge = document.getElementById('gaugeFill');
    var gText = document.getElementById('gaugeText');
    var ratio = d.stress_ratio;
    var target = Math.round(ratio * 100);
    var arcLen = 314.16;
    setTimeout(function () {
      gauge.style.strokeDashoffset = arcLen * (1 - ratio);
      var cur = 0, step = Math.max(target / 50, 0.5);
      var iv = setInterval(function () {
        cur += step;
        if (cur >= target) { cur = target; clearInterval(iv); }
        gText.textContent = Math.round(cur) + '%';
      }, 25);
    }, 400);

    /* bar chart animation */
    document.querySelectorAll('.bar-fill').forEach(function (bar) {
      var w = bar.getAttribute('data-width');
      setTimeout(function () { bar.style.width = w + '%'; }, 700);
    });

    /* tabs */
    document.querySelectorAll('.tab-btn').forEach(function (btn) {
      btn.addEventListener('click', function () {
        document.querySelectorAll('.tab-btn').forEach(function (b) { b.classList.remove('active'); b.setAttribute('aria-selected', 'false'); });
        document.querySelectorAll('.tab-panel').forEach(function (p) { p.classList.remove('active'); });
        btn.classList.add('active'); btn.setAttribute('aria-selected', 'true');
        document.getElementById('tab-' + btn.getAttribute('data-tab')).classList.add('active');
      });
    });

    /* timeline tooltips */
    var tip = null;
    document.querySelectorAll('.tl-block').forEach(function (block) {
      block.addEventListener('mouseenter', function (e) {
        tip = document.createElement('div'); tip.className = 'tooltip';
        tip.textContent = block.getAttribute('data-tip'); document.body.appendChild(tip); posTip(e);
      });
      block.addEventListener('mousemove', posTip);
      block.addEventListener('mouseleave', function () { if (tip) { tip.remove(); tip = null; } });
    });
    function posTip(e) { if (!tip) return; tip.style.left = (e.clientX - tip.offsetWidth / 2) + 'px'; tip.style.top = (e.clientY - 36) + 'px'; }

    /* confusion matrix */
    if (d.confusion_matrix && d.confusion_matrix.length) {
      drawCM('cmChart', d.confusion_matrix, d.target_names || [], 380, 55);
    }

    /* feature importance chart */
    if (d.feature_importance_labels && d.feature_importance_labels.length) {
      _charts.push(new Chart(document.getElementById('featChart'), {
        type: 'bar',
        data: {
          labels: d.feature_importance_labels,
          datasets: [{ label: 'Importance', data: d.feature_importance_values,
            backgroundColor: 'rgba(124,92,252,0.7)', borderColor: 'rgba(124,92,252,1)', borderWidth: 1, borderRadius: 4 }]
        },
        options: {
          indexAxis: 'y', responsive: true,
          scales: { x: { beginAtZero: true, ticks: { color: '#8585a8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { ticks: { color: '#8585a8', font: { size: 10 } }, grid: { display: false } } },
          plugins: { legend: { display: false } }
        }
      }));
    }

    /* back button */
    document.getElementById('backBtn').addEventListener('click', function () { showPage('home'); });

    if (d.tracking_id) {
      var applyBtn = document.getElementById('historyApplyBtn');
      var resetBtn = document.getElementById('historyResetBtn');
      var exportCsvBtn = document.getElementById('historyExportCsvBtn');
      var startInput = document.getElementById('historyStart');
      var endInput = document.getElementById('historyEnd');

      if (applyBtn) {
        applyBtn.addEventListener('click', function () {
          loadHistory(d.tracking_id, {
            start: toHistoryIso(startInput && startInput.value, false),
            end: toHistoryIso(endInput && endInput.value, true),
          });
        });
      }

      if (resetBtn) {
        resetBtn.addEventListener('click', function () {
          if (startInput) startInput.value = '';
          if (endInput) endInput.value = '';
          loadHistory(d.tracking_id, { start: '', end: '' });
        });
      }

      if (exportCsvBtn) {
        exportCsvBtn.addEventListener('click', function () {
          var query = buildHistoryQuery(
            toHistoryIso(startInput && startInput.value, false),
            toHistoryIso(endInput && endInput.value, true),
            'csv'
          );
          window.open('/api/history/' + encodeURIComponent(d.tracking_id) + '/export' + query, '_blank');
        });
      }
    }
  }

  /* ================================================================
     RENDER: COMPARE PAGE
     ================================================================ */
  function renderCompare(data) {
    destroyCharts();
    var results = data.results;
    var subjectId = data.subject_id;
    var best = results[0];
    var hasBest = best && !best.error;

    var winnerHTML = '';
    if (hasBest) {
      winnerHTML =
        '<div class="winner-banner animate-in">' +
          '<div class="winner-badge">\uD83C\uDFC6</div>' +
          '<div class="winner-info">' +
            '<h2>' + esc(best.classifier_name) + '</h2>' +
            '<p>Accuracy: <strong>' + (best.accuracy * 100).toFixed(2) + '%</strong> &middot; ' +
              'F1 (weighted): <strong>' + (best.f1_weighted * 100).toFixed(2) + '%</strong> &middot; ' +
              'Training time: <strong>' + best.train_time_sec + 's</strong></p>' +
          '</div>' +
        '</div>';
    }

    /* comparison table */
    var thead = '<tr><th>Rank</th><th>Classifier</th><th>Accuracy</th><th>F1 (weighted)</th><th>F1 (macro)</th><th>Precision</th><th>Recall</th><th>ROC AUC</th><th>CV Mean</th><th>CV Std</th><th>Time (s)</th></tr>';
    var tbody = '';
    results.forEach(function (r, idx) {
      var cls = (idx === 0 && !r.error) ? 'best-row' : '';
      var rank = idx + 1;
      var bestBadge = (idx === 0 && !r.error) ? ' <span class="badge baseline">Best</span>' : '';
      var errBadge = r.error ? ' <span class="badge" style="background:#ef4444;color:#fff;font-size:.7rem;padding:.15rem .45rem;border-radius:.4rem;">Error</span>' : '';
      if (r.error) {
        tbody += '<tr class="' + cls + '"><td><strong>' + rank + '</strong></td><td>' + esc(r.classifier_name) + errBadge + '</td><td colspan="9" style="color:#f87171;font-size:.82rem;">' + esc(r.error) + '</td></tr>';
      } else {
        var aucCell = r.roc_auc != null ? (r.roc_auc * 100).toFixed(2) + '%' : 'N/A';
        tbody += '<tr class="' + cls + '">' +
          '<td><strong>' + rank + '</strong></td>' +
          '<td>' + esc(r.classifier_name) + bestBadge + '</td>' +
          '<td>' + (r.accuracy * 100).toFixed(2) + '%</td>' +
          '<td>' + (r.f1_weighted * 100).toFixed(2) + '%</td>' +
          '<td>' + (r.f1_macro * 100).toFixed(2) + '%</td>' +
          '<td>' + (r.precision * 100).toFixed(2) + '%</td>' +
          '<td>' + (r.recall * 100).toFixed(2) + '%</td>' +
          '<td>' + aucCell + '</td>' +
          '<td>' + (r.cv_mean * 100).toFixed(2) + '%</td>' +
          '<td>&plusmn;' + (r.cv_std * 100).toFixed(2) + '%</td>' +
          '<td>' + r.train_time_sec + '</td></tr>';
      }
    });

    /* confusion matrix canvases */
    var cmItems = '';
    results.forEach(function (r, idx) {
      if (!r.error && r.confusion_matrix && r.confusion_matrix.length) {
        cmItems += '<div class="cm-item"><h3 style="font-size:.95rem;margin-bottom:.5rem;">' + esc(r.classifier_name) + '</h3><canvas id="cm-' + idx + '" width="260" height="260"></canvas></div>';
      }
    });

    $compare.innerHTML =
      '<header class="hero hero-sm animate-in"><h1>Model Comparison</h1>' +
        '<p class="subtitle">Subject <strong>' + esc(subjectId) + '</strong> &mdash; ' + results.length + ' ML classifiers evaluated</p></header>' +
      winnerHTML +
      '<section class="card card-glow-top animate-in"><h2 class="section-title">Performance Metrics</h2>' +
        '<div class="compare-table-wrap"><table class="compare-table"><thead>' + thead + '</thead><tbody>' + tbody + '</tbody></table></div></section>' +
      '<section class="card animate-in"><h2 class="section-title">Accuracy Comparison</h2>' +
        '<div style="position:relative;max-width:700px;margin:0 auto;"><canvas id="accChart"></canvas></div></section>' +
      '<section class="card animate-in"><h2 class="section-title">Multi-Metric Radar</h2>' +
        '<div style="position:relative;max-width:500px;margin:0 auto;"><canvas id="radarChart"></canvas></div></section>' +
      '<section class="card animate-in"><h2 class="section-title">Confusion Matrices</h2><div class="cm-grid">' + cmItems + '</div></section>' +
      '<div class="actions animate-in"><button class="btn btn-back" id="backBtnCompare">' +
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="19" y1="12" x2="5" y2="12"/><polyline points="12 19 5 12 12 5"/></svg> Back to Home</button></div>' +
      '<footer class="footer animate-in"><p>WESAD Stress Detection &middot; Multi-Classifier Comparison</p>' +
        '<div class="footer-links">' +
          '<a href="https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection" target="_blank" rel="noopener">' +
            '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg> WESAD Dataset</a>' +
          '<span style="color:var(--text-dim);">&copy; 2026</span>' +
        '</div></footer>';

    showPage('compare');
    animateCompare(results);
  }

  function animateCompare(results) {
    var COLORS = [
      'rgba(52,211,153,0.8)', 'rgba(124,92,252,0.8)', 'rgba(248,113,113,0.8)',
      'rgba(251,191,36,0.8)', 'rgba(96,165,250,0.8)',  'rgba(167,139,250,0.8)'
    ];
    var BORDERS = COLORS.map(function (c) { return c.replace('0.8', '1'); });
    var names = results.map(function (r) { return r.classifier_name; });

    /* accuracy bar chart */
    _charts.push(new Chart(document.getElementById('accChart'), {
      type: 'bar',
      data: {
        labels: names,
        datasets: [{ label: 'Accuracy (%)', data: results.map(function (r) { return (r.accuracy * 100).toFixed(2); }),
          backgroundColor: COLORS, borderColor: BORDERS, borderWidth: 1, borderRadius: 6 }]
      },
      options: {
        responsive: true,
        scales: { y: { beginAtZero: true, max: 100, ticks: { color: '#8585a8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                  x: { ticks: { color: '#8585a8' }, grid: { display: false } } },
        plugins: { legend: { display: false }, tooltip: { callbacks: { label: function (ctx) { return ctx.parsed.y + '%'; } } } }
      }
    }));

    /* radar chart */
    var radarLabels = ['Accuracy', 'F1 (weighted)', 'F1 (macro)', 'Precision', 'Recall', 'ROC AUC', 'CV Mean'];
    var radarDS = results.map(function (r, i) {
      return {
        label: r.classifier_name,
        data: [r.accuracy * 100, r.f1_weighted * 100, r.f1_macro * 100, r.precision * 100, r.recall * 100, (r.roc_auc || 0) * 100, r.cv_mean * 100],
        borderColor: BORDERS[i % BORDERS.length],
        backgroundColor: COLORS[i % COLORS.length].replace('0.8', '0.15'),
        pointBackgroundColor: BORDERS[i % BORDERS.length],
        borderWidth: 2
      };
    });
    _charts.push(new Chart(document.getElementById('radarChart'), {
      type: 'radar',
      data: { labels: radarLabels, datasets: radarDS },
      options: {
        responsive: true,
        scales: { r: { beginAtZero: true, max: 100, ticks: { color: '#8585a8', backdropColor: 'transparent' },
                       grid: { color: 'rgba(255,255,255,0.08)' }, pointLabels: { color: '#e2e2f0', font: { size: 11 } } } },
        plugins: { legend: { labels: { color: '#e2e2f0', padding: 12 } } }
      }
    }));

    /* confusion matrices */
    results.forEach(function (r, idx) {
      if (!r.error && r.confusion_matrix && r.confusion_matrix.length) {
        drawCM('cm-' + idx, r.confusion_matrix, r.target_names || [], 260, 40);
      }
    });

    /* back button */
    document.getElementById('backBtnCompare').addEventListener('click', function () { showPage('home'); });
  }

  /* ── shared: draw confusion matrix on canvas ──────────────────────── */
  function drawCM(canvasId, cm, targetNames, size, pad) {
    var canvas = document.getElementById(canvasId);
    if (!canvas) return;
    var ctx = canvas.getContext('2d');
    var n = cm.length;
    var cellSize = (size - pad) / n;
    var maxVal = 0;
    cm.forEach(function (row) { row.forEach(function (v) { if (v > maxVal) maxVal = v; }); });

    for (var i = 0; i < n; i++) {
      for (var j = 0; j < n; j++) {
        var intensity = maxVal > 0 ? cm[i][j] / maxVal : 0;
        var r = Math.round(124 * intensity + 20);
        var g = Math.round(92 * intensity + 20);
        var b = Math.round(252 * intensity + 40);
        ctx.fillStyle = 'rgb(' + r + ',' + g + ',' + b + ')';
        ctx.fillRect(pad + j * cellSize, pad + i * cellSize, cellSize - 2, cellSize - 2);
        ctx.fillStyle = intensity > 0.5 ? '#fff' : '#aaa';
        ctx.font = 'bold 14px Inter, sans-serif';
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(cm[i][j], pad + j * cellSize + cellSize / 2, pad + i * cellSize + cellSize / 2);
      }
    }
    ctx.fillStyle = '#8585a8'; ctx.font = '11px Inter, sans-serif';
    for (var k = 0; k < n; k++) {
      var lbl = targetNames[k] || k;
      ctx.textAlign = 'center';
      ctx.fillText(lbl, pad + k * cellSize + cellSize / 2, pad - 10);
      ctx.textAlign = 'right';
      ctx.fillText(lbl, pad - 6, pad + k * cellSize + cellSize / 2);
    }
  }

  /* ── keyboard shortcuts ─────────────────────────────────────────── */
  document.addEventListener('keydown', function (e) {
    // Escape → go back to home from result or compare page
    if (e.key === 'Escape') {
      if ($result.style.display !== 'none' || $compare.style.display !== 'none') {
        e.preventDefault();
        showPage('home');
      }
    }
    // Ctrl+Enter → submit whichever form is in focus
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      var active = document.activeElement;
      if (active && active.closest('#manualForm')) {
        e.preventDefault();
        $manualForm.dispatchEvent(new Event('submit', { cancelable: true }));
      } else if (active && active.closest('#uploadForm')) {
        e.preventDefault();
        $uploadForm.dispatchEvent(new Event('submit', { cancelable: true }));
      }
    }
  });

  /* ── boot ─────────────────────────────────────────────────────────── */
  init();
})();
