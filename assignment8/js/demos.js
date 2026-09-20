/* Live demos. Every number on screen is computed in the browser from the
   values shown. Nothing is a stored result or an illustration. */

var DEMOS = (function () {
  'use strict';

  /* ---------- small helpers ---------- */

  function el(tag, cls, html) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html != null) n.innerHTML = html;
    return n;
  }
  function ctrlRow(parent) { var d = el('div', 'ctrls'); parent.appendChild(d); return d; }

  function slider(row, label, min, max, step, val, fmt, on) {
    var w = el('div', 'ctrl');
    var id = 'i' + Math.random().toString(36).slice(2, 8);
    var lb = el('label', null, label); lb.setAttribute('for', id);
    var inp = document.createElement('input');
    inp.type = 'range'; inp.id = id; inp.min = min; inp.max = max; inp.step = step; inp.value = val;
    var out = document.createElement('output');
    out.textContent = fmt(+val);
    inp.addEventListener('input', function () { out.textContent = fmt(+inp.value); on(+inp.value); });
    w.appendChild(lb); w.appendChild(inp); w.appendChild(out); row.appendChild(w);
    return { get: function () { return +inp.value; }, set: function (v) { inp.value = v; out.textContent = fmt(+v); } };
  }

  function seg(row, labels, idx, on) {
    var w = el('div', 'seg'), btns = [];
    labels.forEach(function (t, i) {
      var b = el('button', null, t);
      b.type = 'button';
      b.setAttribute('aria-pressed', i === idx ? 'true' : 'false');
      b.addEventListener('click', function () {
        btns.forEach(function (x, j) { x.setAttribute('aria-pressed', i === j ? 'true' : 'false'); });
        on(i);
      });
      btns.push(b); w.appendChild(b);
    });
    row.appendChild(w);
    return { set: function (i) { btns[i].click(); } };
  }

  function check(row, label, val, on) {
    var w = el('label', 'toggle');
    var inp = document.createElement('input');
    inp.type = 'checkbox'; inp.checked = val;
    inp.addEventListener('change', function () { on(inp.checked); });
    w.appendChild(inp); w.appendChild(document.createTextNode(label));
    row.appendChild(w);
    return { get: function () { return inp.checked; } };
  }

  function readout(parent) { var d = el('div', 'readout'); parent.appendChild(d); return d; }

  function css(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  /* A canvas that redraws on resize and on a theme change. */
  function canvasPanel(parent, aspect) {
    var box = el('div', 'cvbox');
    var cv = document.createElement('canvas');
    box.appendChild(cv); parent.appendChild(box);
    var ctx = cv.getContext('2d');
    var drawFn = null;

    function resize() {
      var w = box.clientWidth || parent.clientWidth || 600;
      var h = Math.round(w * aspect);
      var dpr = Math.min(window.devicePixelRatio || 1, 2);
      cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
      cv.style.width = w + 'px'; cv.style.height = h + 'px';
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      if (drawFn) drawFn(ctx, w, h);
    }
    var ro = window.ResizeObserver ? new ResizeObserver(resize) : null;
    if (ro) ro.observe(box); else window.addEventListener('resize', resize);
    document.addEventListener('themechange', resize);

    return {
      draw: function (fn) { drawFn = fn; resize(); },
      redraw: resize
    };
  }

  function fmt(x, n) {
    if (!isFinite(x)) return '—';
    return x.toFixed(n == null ? 2 : n);
  }
  function gb(bytes) {
    if (bytes >= 1e9) return (bytes / 1e9).toFixed(2) + ' GB';
    if (bytes >= 1e6) return (bytes / 1e6).toFixed(1) + ' MB';
    return (bytes / 1e3).toFixed(0) + ' kB';
  }

  /* ================= 1. scaled dot-product attention ================= */

  function demoSdpa(root) {
    var toks = ['The', 'cat', 'sat', 'on', 'the', 'mat'];
    /* Fixed embeddings and fixed projections. Small, readable, and real. */
    var E = [
      [0.9, 0.1, -0.4, 0.2], [0.2, 1.0, 0.3, -0.5], [-0.3, 0.6, 0.9, 0.1],
      [0.5, -0.2, 0.1, 0.8], [0.9, 0.1, -0.4, 0.2], [-0.1, 0.7, 0.8, -0.2]
    ];
    var Wq = [[0.7, -0.2, 0.1, 0.3], [0.1, 0.8, -0.3, 0.2], [-0.4, 0.3, 0.6, 0.1], [0.2, 0.1, 0.2, 0.7]];
    var Wk = [[0.6, 0.3, -0.1, 0.2], [-0.2, 0.7, 0.4, -0.1], [0.3, -0.2, 0.8, 0.2], [0.1, 0.4, 0.1, 0.6]];
    var Wv = [[0.5, 0.1, 0.3, -0.2], [0.2, 0.6, -0.1, 0.4], [-0.3, 0.2, 0.7, 0.1], [0.4, -0.1, 0.2, 0.5]];

    function mv(W, x) {
      return W.map(function (row) {
        return row.reduce(function (s, w, i) { return s + w * x[i]; }, 0);
      });
    }
    var Q = E.map(function (x) { return mv(Wq, x); });
    var K = E.map(function (x) { return mv(Wk, x); });
    var V = E.map(function (x) { return mv(Wv, x); });
    var dk = 4;

    var step = 3, masked = true, qsel = 4;

    var row = ctrlRow(root);
    seg(row, ['1 score', '2 scale', '3 mask', '4 softmax', '5 output'], 3, function (i) { step = i; render(); });
    check(row, 'causal mask on', true, function (v) { masked = v; render(); });
    slider(row, 'query token', 0, 5, 1, 4, function (v) { return toks[v]; }, function (v) { qsel = v; render(); });

    var host = el('div', 'scrollx'); root.appendChild(host);
    var out = readout(root);

    function render() {
      var S = [], i, j;
      for (i = 0; i < 6; i++) {
        S[i] = [];
        for (j = 0; j < 6; j++) {
          var d = 0;
          for (var t = 0; t < dk; t++) d += Q[i][t] * K[j][t];
          S[i][j] = d;
        }
      }
      var M = S.map(function (r) { return r.map(function (v) { return step >= 1 ? v / Math.sqrt(dk) : v; }); });
      if (step >= 2 && masked) {
        M = M.map(function (r, a) { return r.map(function (v, b) { return b > a ? -Infinity : v; }); });
      }
      var W = M;
      if (step >= 3) {
        W = M.map(function (r) {
          var mx = Math.max.apply(null, r.filter(isFinite));
          var ex = r.map(function (v) { return isFinite(v) ? Math.exp(v - mx) : 0; });
          var s = ex.reduce(function (a, b) { return a + b; }, 0);
          return ex.map(function (v) { return v / s; });
        });
      }

      var title = ['Raw dot products Q·K', 'Divided by √dₖ = 2',
        (masked ? 'Future positions set to −∞' : 'Mask switched off'),
        'Softmax across each row', 'Softmax weights, and the output vectors'][step];

      var h = '<table class="grid"><tr><th class="lab">' + title + '</th>';
      for (j = 0; j < 6; j++) h += '<th>' + toks[j] + '</th>';
      h += '</tr>';
      for (i = 0; i < 6; i++) {
        h += '<tr><th class="lab">' + toks[i] + '</th>';
        for (j = 0; j < 6; j++) {
          var v = W[i][j];
          var txt = !isFinite(v) ? '−∞' : fmt(v, step >= 3 ? 2 : 2);
          var a = step >= 3 ? Math.min(v * 1.4, 1) : 0;
          var bgc = (step >= 3 && isFinite(v)) ? 'background:color-mix(in srgb, var(--found) ' + (a * 70) + '%, transparent);' : '';
          var em = (i === qsel) ? 'color:var(--ink);font-weight:700;' : '';
          h += '<td style="' + bgc + em + '">' + txt + '</td>';
        }
        h += '</tr>';
      }
      h += '</table>';
      host.innerHTML = h;

      var r = W[qsel];
      var txt;
      if (step < 3) {
        txt = '<b>' + toks[qsel] + '</b> scores every key. These are not yet weights: they are not positive and they do not add to one.';
        if (step >= 2 && masked) txt += '\nPositions after ' + toks[qsel] + ' are −∞, so the softmax will give them exactly zero.';
      } else {
        var o = [0, 0, 0, 0];
        for (j = 0; j < 6; j++) for (var t2 = 0; t2 < 4; t2++) o[t2] += r[j] * V[j][t2];
        var top = r.map(function (v, k) { return [v, k]; }).sort(function (a, b) { return b[0] - a[0]; });
        txt = '<b>' + toks[qsel] + '</b> weights: ' +
          r.map(function (v, k) { return toks[k] + ' ' + fmt(v, 2); }).join('   ') +
          '\nsum = <b>' + fmt(r.reduce(function (a, b) { return a + b; }, 0), 3) + '</b>' +
          '   strongest: <b>' + toks[top[0][1]] + '</b>' +
          '\noutput vector = Σ weight × value = [' + o.map(function (x) { return fmt(x, 3); }).join(', ') + ']';
        if (!masked) {
          var leak = 0;
          for (j = qsel + 1; j < 6; j++) leak += r[j];
          txt += '\n<span class="bad">Mask off: ' + fmt(leak * 100, 1) + '% of this token’s attention is on tokens that have not happened yet.</span>';
        }
      }
      out.innerHTML = txt;
    }
    render();
  }

  /* ================= 2. position schemes ================= */

  function demoPosition(root) {
    var L = 32, slope = 0.25, base = 10000, dmodel = 64;
    var showLearned = true, showSin = true, showAlibi = true, showRope = true;

    var row = ctrlRow(root);
    slider(row, 'trained length', 8, 64, 1, 32, function (v) { return v + ' tok'; }, function (v) { L = v; panel.redraw(); });
    slider(row, 'ALiBi slope', 0.02, 0.6, 0.02, 0.25, function (v) { return fmt(v, 2); }, function (v) { slope = v; panel.redraw(); });
    slider(row, 'RoPE base', 1000, 100000, 1000, 10000, function (v) { return String(v); }, function (v) { base = v; panel.redraw(); });
    var row2 = ctrlRow(root);
    check(row2, 'learned absolute', true, function (v) { showLearned = v; panel.redraw(); });
    check(row2, 'sinusoidal', true, function (v) { showSin = v; panel.redraw(); });
    check(row2, 'ALiBi', true, function (v) { showAlibi = v; panel.redraw(); });
    check(row2, 'RoPE', true, function (v) { showRope = v; panel.redraw(); });

    var panel = canvasPanel(root, 0.36);
    var out = readout(root);

    function sinusoidalSim(d) {
      /* dot product of two sinusoidal encodings separated by d, normalised */
      var s = 0, n = 0;
      for (var i = 0; i < dmodel / 2; i++) {
        var w = 1 / Math.pow(10000, (2 * i) / dmodel);
        s += Math.cos(w * d); n += 1;
      }
      return s / n;
    }
    function ropeSim(d) {
      var s = 0, n = 0;
      for (var i = 0; i < dmodel / 2; i++) {
        var th = 1 / Math.pow(base, (2 * i) / dmodel);
        s += Math.cos(th * d); n += 1;
      }
      return s / n;
    }

    panel.draw(function (ctx, w, h) {
      var pad = { l: 46, r: 14, t: 14, b: 30 };
      var W = w - pad.l - pad.r, H = h - pad.t - pad.b;
      var maxD = 64;
      var X = function (d) { return pad.l + (d / maxD) * W; };
      var Y = function (v) { return pad.t + (1 - (v + 0.25) / 1.3) * H; };

      ctx.clearRect(0, 0, w, h);

      /* untrained region */
      ctx.fillStyle = css('--bg-sunken');
      ctx.fillRect(X(L), pad.t, X(maxD) - X(L), H);
      ctx.strokeStyle = css('--rule-firm'); ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(X(L), pad.t); ctx.lineTo(X(L), pad.t + H); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = css('--ink-faint');
      ctx.font = '500 11px ui-monospace, monospace';
      ctx.fillText('never trained here', X(L) + 6, pad.t + 13);

      /* axes */
      ctx.strokeStyle = css('--rule'); ctx.beginPath();
      ctx.moveTo(pad.l, Y(0)); ctx.lineTo(pad.l + W, Y(0)); ctx.stroke();
      ctx.fillStyle = css('--ink-faint');
      ctx.fillText('1.0', 8, Y(1) + 4); ctx.fillText('0', 8, Y(0) + 4);
      ctx.fillText('distance between query and key →', pad.l, h - 8);

      function line(fn, colour, opt) {
        opt = opt || {};
        ctx.strokeStyle = colour; ctx.lineWidth = opt.width || 2;
        ctx.setLineDash(opt.dash || []);
        ctx.beginPath();
        for (var d = 0; d <= (opt.limit || maxD); d += 0.25) {
          var v = fn(d);
          if (d === 0) ctx.moveTo(X(d), Y(v)); else ctx.lineTo(X(d), Y(v));
        }
        ctx.stroke(); ctx.setLineDash([]);
        if (opt.label) {
          ctx.fillStyle = colour;
          ctx.font = '600 11px ui-monospace, monospace';
          ctx.textAlign = 'right';
          ctx.fillText(opt.label, X(opt.limit || maxD) - 4, Y(fn(opt.limit || maxD)) + (opt.up ? -9 : 16));
          ctx.textAlign = 'left';
        }
      }

      /* Sinusoidal and RoPE are the same function of distance at the same base.
         They are drawn overlaid on purpose: that coincidence is the lesson. */
      if (showSin) line(sinusoidalSim, css('--mem'), { width: 4, label: showRope ? '' : 'sinusoidal' });
      if (showRope) line(ropeSim, css('--found'), { dash: [6, 5], up: true, label: showSin ? 'RoPE, lying on top of sinusoidal' : 'RoPE' });
      if (showAlibi) line(function (d) { return Math.exp(-slope * d); }, css('--comp'), { label: 'ALiBi' });
      if (showLearned) {
        line(function () { return 1; }, css('--pos'), { limit: L });
        ctx.fillStyle = css('--pos');
        ctx.beginPath(); ctx.arc(X(L), Y(1), 4, 0, 7); ctx.fill();
        ctx.font = '600 11px ui-monospace, monospace';
        ctx.fillText('learned table ends', X(L) + 8, Y(1) - 6);
      }
    });

    out.innerHTML =
      'Each curve is the factor that position alone applies to an attention weight, set to 1 at distance zero. ' +
      'The shapes are the point.\n' +
      '<b>Learned absolute</b> is flat and then stops. It carries no distance signal, and past the last row of the table there is nothing at all.\n' +
      '<b>Sinusoidal</b> decays and then wobbles. The signal is there but it is indirect, and it shares dimensions with content.\n' +
      '<b>ALiBi</b> is a clean decay that never stops decaying. That is why it extrapolates, and also why a distant token can never win.\n' +
      '<b>RoPE</b> lies exactly on top of the sinusoidal curve, and that is not a drawing fault. At the same base the two are the same function of distance. The difference is where the number is used. Sinusoidal adds it to the embedding, where it becomes one cross term among four and competes with content. RoPE puts it inside the score, where nothing dilutes it. The same signal, delivered somewhere it survives.\n' +
      'Push the RoPE base up and only the dashed line stretches. That one slider is what position interpolation, NTK-aware scaling and YaRN all argue about.';
  }

  /* ================= 3. RoPE rotation ================= */

  function demoRope(root) {
    var i = 8, j = 2, theta = 0.35;

    var row = ctrlRow(root);
    var si = slider(row, 'query at position', 0, 20, 1, 8, function (v) { return String(v); }, function (v) { i = v; panel.redraw(); upd(); });
    var sj = slider(row, 'key at position', 0, 20, 1, 2, function (v) { return String(v); }, function (v) { j = v; panel.redraw(); upd(); });
    slider(row, 'θ per step', 0.05, 0.8, 0.05, 0.35, function (v) { return fmt(v, 2); }, function (v) { theta = v; panel.redraw(); upd(); });
    var b = el('button', null, 'move both forward 5'); b.type = 'button';
    b.style.cssText = 'font:500 12px var(--sans);padding:7px 11px;border-radius:7px;border:1px solid var(--rule-firm);background:var(--bg-panel);color:var(--ink);cursor:pointer';
    b.addEventListener('click', function () {
      if (i + 5 <= 20 && j + 5 <= 20) { i += 5; j += 5; si.set(i); sj.set(j); panel.redraw(); upd(); }
    });
    row.appendChild(b);

    var panel = canvasPanel(root, 0.42);
    var out = readout(root);

    panel.draw(function (ctx, w, h) {
      ctx.clearRect(0, 0, w, h);
      var cx = w / 2, cy = h / 2 + 6, R = Math.min(w, h) * 0.34;

      ctx.strokeStyle = css('--rule'); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(cx, cy, R, 0, 7); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(cx - R - 10, cy); ctx.lineTo(cx + R + 10, cy);
      ctx.moveTo(cx, cy - R - 10); ctx.lineTo(cx, cy + R + 10); ctx.stroke();

      function arrow(ang, colour, label) {
        var x = cx + R * Math.cos(-ang), y = cy + R * Math.sin(-ang);
        ctx.strokeStyle = colour; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(x, y); ctx.stroke();
        ctx.fillStyle = colour;
        ctx.beginPath(); ctx.arc(x, y, 5, 0, 7); ctx.fill();
        ctx.font = '600 12px ui-monospace, monospace';
        ctx.fillText(label, x + 9 * Math.cos(-ang), y + 9 * Math.sin(-ang) + 4);
      }

      /* the wedge between them */
      ctx.fillStyle = css('--found-bg');
      ctx.beginPath(); ctx.moveTo(cx, cy);
      ctx.arc(cx, cy, R * 0.45, -Math.max(i, j) * theta, -Math.min(i, j) * theta, false);
      ctx.closePath(); ctx.fill();

      arrow(i * theta, css('--comp'), 'q @ ' + i);
      arrow(j * theta, css('--mem'), 'k @ ' + j);

      ctx.fillStyle = css('--ink-faint');
      ctx.font = '500 11px ui-monospace, monospace';
      ctx.fillText('the shaded angle is (i − j)θ, and it is all the score can see', 12, h - 10);
    });

    function upd() {
      var gap = i - j;
      out.innerHTML =
        'query rotated by iθ = ' + fmt(i * theta, 2) + ' rad, key rotated by jθ = ' + fmt(j * theta, 2) + ' rad\n' +
        'angle between them = (i − j)θ = <b>' + fmt(gap * theta, 3) + '</b> rad,   cos = <b>' + fmt(Math.cos(gap * theta), 4) + '</b>\n' +
        'distance i − j = <b>' + gap + '</b>\n' +
        'Move both tokens forward together. Both arrows turn, the shaded angle does not, and neither does the score. ' +
        'That is the whole mechanism: absolute rotation cancels in the dot product and relative rotation survives.';
    }
    upd();
  }

  /* ================= 4. context extension ================= */

  function demoExtend(root) {
    var L = 4096, s = 8, dim = 64, base = 10000, alpha = 1, beta = 32;

    var row = ctrlRow(root);
    slider(row, 'trained length', 1024, 8192, 1024, 4096, function (v) { return (v / 1024) + 'K'; }, function (v) { L = v; panel.redraw(); upd(); });
    slider(row, 'extension factor', 2, 32, 1, 8, function (v) { return v + '×'; }, function (v) { s = v; panel.redraw(); upd(); });

    var panel = canvasPanel(root, 0.36);
    var out = readout(root);

    function theta(i) { return 1 / Math.pow(base, (2 * i) / dim); }
    function turns(i) { return L * theta(i) / (2 * Math.PI); }        /* full turns inside the window */
    function ramp(i) {
      var r = turns(i);
      if (r > beta) return 0;      /* fast pair: leave alone */
      if (r < alpha) return 1;     /* slow pair: interpolate fully */
      return (beta - r) / (beta - alpha);
    }
    var nbTerm = Math.pow(s, dim / (dim - 2));

    panel.draw(function (ctx, w, h) {
      var pad = { l: 52, r: 14, t: 16, b: 34 };
      var W = w - pad.l - pad.r, H = h - pad.t - pad.b;
      var n = dim / 2;
      var X = function (i) { return pad.l + (i / (n - 1)) * W; };
      var Y = function (v) { return pad.t + (1 - v) * H; };

      ctx.clearRect(0, 0, w, h);
      ctx.strokeStyle = css('--rule'); ctx.lineWidth = 1;
      [0, 0.5, 1].forEach(function (v) {
        ctx.beginPath(); ctx.moveTo(pad.l, Y(v)); ctx.lineTo(pad.l + W, Y(v)); ctx.stroke();
      });
      ctx.fillStyle = css('--ink-faint');
      ctx.font = '500 11px ui-monospace, monospace';
      ctx.fillText('1.0', 14, Y(1) + 4);
      ctx.fillText('1/' + s, 14, Y(0) + 4);
      ctx.fillText('fast pairs (local order)', pad.l, h - 9);
      ctx.textAlign = 'right';
      ctx.fillText('slow pairs (long range)', pad.l + W, h - 9);
      ctx.textAlign = 'left';

      function line(f, colour, label, at, above) {
        ctx.strokeStyle = colour; ctx.lineWidth = 2.2;
        ctx.beginPath();
        for (var i = 0; i < n; i++) {
          var v = f(i);
          if (i === 0) ctx.moveTo(X(i), Y(v)); else ctx.lineTo(X(i), Y(v));
        }
        ctx.stroke();
        var li = Math.round(n * at);
        ctx.fillStyle = colour;
        ctx.font = '600 11px ui-monospace, monospace';
        ctx.fillText(label, X(li) + 5, Y(f(li)) + (above ? -7 : 15));
      }

      /* normalised so 1 = untouched frequency, 0 = fully interpolated (divided by s) */
      var norm = function (scaleFactor) {
        return (Math.log(scaleFactor) - Math.log(1 / s)) / (0 - Math.log(1 / s));
      };
      line(function () { return norm(1 / s); }, css('--pos'), 'position interpolation', 0.06, true);
      line(function (i) { return norm(Math.pow(nbTerm, -(2 * i) / dim)); }, css('--mem'), 'NTK-aware', 0.62, false);
      line(function (i) { var g = ramp(i); return norm((1 - g) * 1 + g * (1 / s)); }, css('--found'), 'YaRN, by parts', 0.06, false);
    });

    function upd() {
      var n = dim / 2, fast = 0, slow = 0, mid = 0;
      for (var i = 0; i < n; i++) {
        var r = turns(i);
        if (r > beta) fast++; else if (r < alpha) slow++; else mid++;
      }
      out.innerHTML =
        'The vertical axis is how much each dimension pair is squeezed. 1.0 means untouched. The bottom line means divided by the full ' + s + '×.\n' +
        '<b>Position interpolation</b> is flat at the bottom. Every pair is squeezed equally, including the fast ones that encode which word came first.\n' +
        '<b>NTK-aware</b> changes the base instead, so the squeeze slides across the dimensions and the fastest pairs keep their resolution.\n' +
        '<b>YaRN</b> makes the split explicit. At a trained length of ' + (L / 1024) + 'K, <b>' + fast + '</b> pairs complete more than ' + beta +
        ' turns inside the window and are left alone, <b>' + slow + '</b> pairs complete fewer than ' + alpha +
        ' turn and are interpolated fully, and <b>' + mid + '</b> pairs are ramped between the two.\n' +
        'Raise the extension factor and watch all three sink. That sinking is the quality you are paying, and it is why the family has a ceiling.';
    }
    upd();
  }

  /* ================= 5. KV cache ================= */

  function demoCache(root) {
    var layers = 48, heads = 64, hd = 128, T = 32768, batch = 8, bytes = 2, groups = 8, latent = 512, rope = 64;

    var row = ctrlRow(root);
    slider(row, 'context', 1024, 262144, 1024, 32768, function (v) { return (v / 1024) + 'K'; }, function (v) { T = v; upd(); });
    slider(row, 'users at once', 1, 64, 1, 8, function (v) { return String(v); }, function (v) { batch = v; upd(); });
    slider(row, 'layers', 8, 96, 4, 48, function (v) { return String(v); }, function (v) { layers = v; upd(); });
    var row2 = ctrlRow(root);
    slider(row2, 'query heads', 8, 128, 8, 64, function (v) { return String(v); }, function (v) { heads = v; upd(); });
    slider(row2, 'GQA key/value heads', 1, 32, 1, 8, function (v) { return String(v); }, function (v) { groups = v; upd(); });
    seg(row2, ['bf16', 'fp8'], 0, function (i) { bytes = i === 0 ? 2 : 1; upd(); });

    var host = el('div', 'scrollx'); root.appendChild(host);
    var out = readout(root);

    function upd() {
      var kvh = Math.min(groups, heads);
      var rows = [
        ['Multi-head, one K/V per head', 2 * layers * heads * hd * T * bytes, 'every head keeps its own view'],
        ['Grouped-query, ' + kvh + ' K/V heads', 2 * layers * kvh * hd * T * bytes, heads / kvh + ' query heads share one K/V head'],
        ['Multi-query, 1 K/V head', 2 * layers * 1 * hd * T * bytes, 'all heads share one view'],
        ['Latent, ' + latent + ' + ' + rope + ' per token', layers * (latent + rope) * T * bytes, 'one compressed vector, not a key and a value']
      ];
      var base = rows[0][1];
      var maxv = base;

      var h = '<table class="grid" style="width:100%"><tr><th class="lab">method</th><th>one user</th><th>' +
        batch + ' users</th><th>vs multi-head</th><th class="lab"></th></tr>';
      rows.forEach(function (r) {
        var pctOf = (r[1] / base) * 100;
        var barw = Math.max((r[1] / maxv) * 100, 0.4);
        h += '<tr><td class="lab">' + r[0] + '</td><td>' + gb(r[1]) + '</td><td>' + gb(r[1] * batch) +
          '</td><td>' + fmt(pctOf, 1) + '%</td>' +
          '<td class="lab" style="width:34%"><span style="display:block;height:9px;border-radius:3px;background:var(--found);width:' + barw + '%"></span></td></tr>';
      });
      h += '</table>';
      host.innerHTML = h;

      var mhaTot = base * batch, mlaTot = rows[3][1] * batch;
      out.innerHTML =
        'cache bytes = 2 × layers × kv_heads × head_dim × context × users × bytes_per_number\n' +
        'At ' + (T / 1024) + 'K context with ' + batch + ' conversations open, multi-head needs <b>' + gb(mhaTot) +
        '</b> and the latent form needs <b>' + gb(mlaTot) + '</b>.\n' +
        'Now double the context and look again. <b>Every line doubles.</b> Head sharing changes the slope of the line. ' +
        'It does not make the line stop rising, because something is still stored for every token. That is why the compute row and the state row exist.';
    }
    upd();
  }

  /* ================= 6. window and sinks ================= */

  function demoWindow(root) {
    var T = 28, w = 6, sinks = 0, q = 22;

    var row = ctrlRow(root);
    slider(row, 'window', 2, 28, 1, 6, function (v) { return v + ' tok'; }, function (v) { w = v; render(); });
    slider(row, 'kept sink tokens', 0, 4, 1, 0, function (v) { return String(v); }, function (v) { sinks = v; render(); });
    slider(row, 'query position', 0, 27, 1, 22, function (v) { return String(v); }, function (v) { q = v; render(); });

    var host = el('div', 'scrollx'); root.appendChild(host);
    var out = readout(root);

    function render() {
      var h = '<table class="grid" style="font-size:10px"><tr><th class="lab">query ↓ &nbsp; key →</th>';
      for (var k = 0; k < T; k++) h += '<th style="padding:2px 3px">' + k + '</th>';
      h += '</tr>';
      for (var i = 0; i < T; i++) {
        h += '<tr><th class="lab" style="padding:2px 6px">' + i + '</th>';
        for (var j = 0; j < T; j++) {
          var causal = j <= i;
          var inWin = j > i - w;
          var isSink = j < sinks;
          var on = causal && (inWin || isSink);
          var bg = !causal ? 'var(--bg-sunken)'
            : on ? (isSink && !inWin ? 'var(--pos)' : 'var(--found)')
            : 'var(--bg-panel)';
          var ring = (i === q && on) ? 'outline:2px solid var(--comp);outline-offset:-2px;' : '';
          h += '<td style="padding:0;width:13px;height:13px;background:' + bg + ';' + ring + '"></td>';
        }
        h += '</tr>';
      }
      h += '</table>';
      host.innerHTML = h;

      var seen = 0, lost = 0;
      for (var b = 0; b <= q; b++) {
        if (b > q - w || b < sinks) seen++; else lost++;
      }
      var cache = Math.min(w + sinks, T);
      out.innerHTML =
        'Query at position <b>' + q + '</b> can read <b>' + seen + '</b> keys and cannot read <b>' + lost + '</b> that came before it.\n' +
        'Cache held: <b>' + cache + '</b> entries, and it stops growing. A full causal cache at this position would hold ' + (q + 1) + '.\n' +
        (sinks === 0
          ? '<span class="bad">With no sink kept, the first tokens leave the cache as soon as the window passes them. That is the configuration where perplexity explodes.</span>'
          : '<span class="good">The first ' + sinks + ' token' + (sinks > 1 ? 's stay' : ' stays') + ' in the cache permanently. Softmax has somewhere to put the weight it does not want, and the stream stays stable.</span>') +
        '\nThe empty cells to the left of the blue band are the ones to watch. Those keys are inside the sequence, already written, and unreadable. ' +
        'Grey is the future, which the causal mask removes. This buys stream length, not context. What left the window is gone.';
    }
    render();
  }

  /* ================= 7. linear attention regrouping ================= */

  function demoLinear(root) {
    var q = 2, softmax = false;
    var k = [0.5, 1.0, 1.5], v = [10, 20, 30];

    var row = ctrlRow(root);
    slider(row, 'query q', 0.5, 4, 0.1, 2, function (x) { return fmt(x, 1); }, function (x) { q = x; upd(); });
    check(row, 'softmax on', false, function (x) { softmax = x; upd(); });
    var row2 = ctrlRow(root);
    k.forEach(function (_, i) {
      slider(row2, 'k' + (i + 1), 0.1, 3, 0.1, k[i], function (x) { return fmt(x, 1); }, function (x) { k[i] = x; upd(); });
    });

    var out = readout(root);

    function upd() {
      var scores = k.map(function (ki) { return q * ki; });
      var wts = scores.slice();
      if (softmax) {
        var mx = Math.max.apply(null, scores);
        var ex = scores.map(function (s) { return Math.exp(s - mx); });
        var sum = ex.reduce(function (a, b) { return a + b; }, 0);
        wts = ex.map(function (e) { return e / sum; });
      }
      var direct = wts.reduce(function (a, wi, i) { return a + wi * v[i]; }, 0);

      var S = k.reduce(function (a, ki, i) { return a + ki * v[i]; }, 0);
      var regrouped = q * S;

      var match = Math.abs(direct - regrouped) < 1e-9;

      out.innerHTML =
        'keys ' + k.map(function (x) { return fmt(x, 1); }).join(', ') + '   values ' + v.join(', ') + '   query ' + fmt(q, 1) + '\n\n' +
        '<b>Direct route</b> — visit every stored key and value:\n' +
        '  ' + wts.map(function (wi, i) { return fmt(wi, softmax ? 3 : 2) + '×' + v[i]; }).join(' + ') + ' = <b>' + fmt(direct, 3) + '</b>\n\n' +
        '<b>Regrouped route</b> — fold the past into one state before the query arrives:\n' +
        '  S = Σ kᵢvᵢ = ' + fmt(S, 2) + '   then output = q × S = <b>' + fmt(regrouped, 3) + '</b>\n\n' +
        (match
          ? '<span class="good">The two routes agree exactly. With no softmax the query factors out, so the whole history can be summed into one fixed-size state and the individual keys can be thrown away. That is linear attention.</span>'
          : '<span class="bad">The two routes no longer agree. Softmax divides every score by a total that depends on all the other scores, so the query cannot be factored out. Exact softmax needs the individual old keys to still be there when the new query arrives, which is exactly why the cache has to keep growing.</span>');
    }
    upd();
  }

  /* ================= 8. delta rule ================= */

  function demoDelta(root) {
    var targets = [40, 55, 20, 75];
    var beta = 1.0;

    var row = ctrlRow(root);
    slider(row, 'write strength β', 0.1, 1, 0.1, 1, function (v) { return fmt(v, 1); }, function (v) { beta = v; upd(); });
    targets.forEach(function (_, i) {
      if (i === 0) return;
      slider(row, 'write ' + (i + 1) + ' wants', 5, 100, 5, targets[i], function (v) { return String(v); }, function (v) { targets[i] = v; upd(); });
    });

    var host = el('div', 'scrollx'); root.appendChild(host);
    var out = readout(root);

    function upd() {
      var addOnly = 0, delta = 0;
      var rows = '<table class="grid" style="width:100%"><tr><th class="lab">write</th><th>wanted</th>' +
        '<th>add-only reads back</th><th>delta rule reads back</th><th>delta written</th></tr>';
      targets.forEach(function (t, i) {
        addOnly += t;
        var d = beta * (t - delta);
        delta += d;
        rows += '<tr><td class="lab">' + (i + 1) + '</td><td>' + t + '</td>' +
          '<td style="color:' + (Math.abs(addOnly - t) < 0.01 ? 'var(--state)' : 'var(--comp)') + '">' + fmt(addOnly, 1) + '</td>' +
          '<td style="color:' + (Math.abs(delta - t) < 0.01 ? 'var(--state)' : 'var(--pos)') + '">' + fmt(delta, 1) + '</td>' +
          '<td>' + (d >= 0 ? '+' : '') + fmt(d, 1) + '</td></tr>';
      });
      rows += '</table>';
      host.innerHTML = rows;

      var last = targets[targets.length - 1];
      out.innerHTML =
        'One key is written four times. Each write replaces what that key should return.\n' +
        'The add-only state ends at <b>' + fmt(addOnly, 1) + '</b> when the answer should be <b>' + last +
        '</b>. It never forgot anything, and that is the problem: every old answer is still inside the sum.\n' +
        'The delta rule reads the state first, writes only the gap, and lands on <b>' + fmt(delta, 1) + '</b>.\n' +
        (beta < 1
          ? 'At β = ' + fmt(beta, 1) + ' the correction is partial, so the state approaches the target instead of reaching it. A learned β is how a real model trades stability against speed of overwrite.'
          : 'At β = 1 the correction is complete and the state is exact after every write. Gated DeltaNet adds a forget gate on top of this, so the state can also be cleared rather than only corrected.');
    }
    upd();
  }

  /* ================= 9. sparse attention cost ================= */

  function demoSparse(root) {
    var T = 8192, k = 64, m = 32;

    var row = ctrlRow(root);
    slider(row, 'context', 1024, 65536, 1024, 8192, function (v) { return (v / 1024) + 'K'; }, function (v) { T = v; upd(); });
    slider(row, 'k kept', 8, 256, 8, 64, function (v) { return String(v); }, function (v) { k = v; upd(); });
    slider(row, 'block size', 4, 128, 4, 32, function (v) { return String(v); }, function (v) { m = v; upd(); });

    var host = el('div', 'scrollx'); root.appendChild(host);
    var out = readout(root);

    function upd() {
      var blocks = Math.ceil(T / m);
      var methods = [
        ['Full attention', T, T, 'score everything, read everything'],
        ['Naive top-k', T, Math.min(k, T), 'score everything, then read k'],
        ['Compressed blocks, then top-k', blocks, Math.min(k, T), 'score ' + blocks + ' summaries, then read k']
      ];
      var worst = 2 * T;

      var h = '<table class="grid" style="width:100%"><tr><th class="lab">method</th><th>keys scored</th>' +
        '<th>values read</th><th>total per query</th><th class="lab"></th></tr>';
      methods.forEach(function (r) {
        var tot = r[1] + r[2];
        h += '<tr><td class="lab">' + r[0] + '</td><td>' + r[1].toLocaleString() + '</td><td>' + r[2].toLocaleString() +
          '</td><td><b>' + tot.toLocaleString() + '</b></td>' +
          '<td class="lab" style="width:34%">' +
          '<span style="display:inline-block;height:9px;border-radius:2px 0 0 2px;background:var(--comp);width:' + (r[1] / worst * 100) + '%"></span>' +
          '<span style="display:inline-block;height:9px;border-radius:0 2px 2px 0;background:var(--found);width:' + (r[2] / worst * 100) + '%"></span>' +
          '</td></tr>';
      });
      h += '</table>';
      host.innerHTML = h;

      out.innerHTML =
        'Red is the cost of finding out which keys matter. Blue is the cost of reading them.\n' +
        'Naive top-k cuts the blue bar to <b>' + k + '</b> and leaves the red bar at <b>' + T.toLocaleString() +
        '</b>. If scoring was the expensive part, and at ' + (T / 1024) + 'K it is, then almost nothing has been saved. ' +
        'This is the catch that every serious sparse method exists to fix.\n' +
        'Compressing ' + m + ' tokens into one summary drops the red bar to <b>' + blocks.toLocaleString() +
        '</b>, a ' + fmt(T / blocks, 0) + '× reduction, and only then is the saving real.\n' +
        '<span class="bad">What it costs:</span> one summary now speaks for ' + m +
        ' tokens, so token-level detail inside a block is gone, and a block that looked unpromising is never opened.';
    }
    upd();
  }

  return {
    sdpa: demoSdpa, position: demoPosition, rope: demoRope, extend: demoExtend,
    cache: demoCache, window: demoWindow, linear: demoLinear, delta: demoDelta, sparse: demoSparse
  };
})();
