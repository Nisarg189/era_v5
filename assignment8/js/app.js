/* Render the timeline, the cards and the source table. */

(function () {
  'use strict';

  const ORDER = ['found', 'pos', 'mem', 'comp', 'state'];
  const T0 = Date.UTC(2014, 5, 1);
  const T1 = Date.UTC(2026, 5, 1);

  const DEMO_HOME = {
    position: 'sinusoidal', rope: 'rope', extend: 'yarn',
    cache: 'gqa', window: 'sinks', linear: 'linear', delta: 'delta', sparse: 'nsa'
  };
  const DEMO_TITLE = {
    sdpa: 'One attention layer, every number computed',
    position: 'Four ways to tell the score how far apart two tokens are',
    rope: 'Why a rotation encodes distance',
    extend: 'Three ways to stretch RoPE past its training length',
    cache: 'The cache bill, and what each method takes off it',
    window: 'What a query can see, and what breaks when the sink is evicted',
    linear: 'The regrouping that softmax forbids',
    delta: 'Add the answer, or write the difference',
    sparse: 'Where the cost of sparse attention actually is'
  };

  const STAGES = [
    { id: 'project',  n: '1 project', d: 'make the query, key and value' },
    { id: 'score',    n: '2 score',   d: 'compare query with key' },
    { id: 'position', n: '3 position', d: 'tell the score how far apart they are' },
    { id: 'select',   n: '4 select',  d: 'decide which keys are read' },
    { id: 'read',     n: '5 read',    d: 'weighted sum of the values' },
    { id: 'cache',    n: '6 cache',   d: 'what generation has to store' }
  ];

  const ERAS = [
    { yrs: '2014 – 2017', t: 'Making it work',
      d: 'Attention is invented to remove a bottleneck, then rebuilt so a sequence can train in parallel. Cost is not yet anybody’s problem.',
      from: '2014-01-01', to: '2018-12-31' },
    { yrs: '2019 – 2020', t: 'The bills come due',
      d: 'Models get large enough that the two bills hurt. Every answer here attacks one of them, and every one gives something up.',
      from: '2019-01-01', to: '2020-12-31' },
    { yrs: '2021 – 2022', t: 'A fixed state, a better angle, a faster kernel',
      d: 'Three separate escapes appear. One replaces the cache, one repairs position, and one refuses to approximate anything at all.',
      from: '2021-01-01', to: '2022-12-31' },
    { yrs: '2023', t: 'The year of length',
      d: 'Open weights arrive and thousands of people try to make an 8K checkpoint reach 32K on one machine. Four position entries land inside fourteen weeks.',
      from: '2023-01-01', to: '2023-12-31' },
    { yrs: '2024', t: 'Recurrence comes back',
      d: 'The fixed state returns, now able to correct itself and cheap enough to train. The cache is attacked from a new direction.',
      from: '2024-01-01', to: '2024-12-31' },
    { yrs: '2025', t: 'Sparsity ships, and position is questioned',
      d: 'Sparse attention stops being an inference patch and moves into pretraining. Then the positional embedding itself is removed.',
      from: '2025-01-01', to: '2026-12-31' }
  ];

  const byDate = MECHANISMS.slice().sort(function (a, b) {
    return a.date < b.date ? -1 : a.date > b.date ? 1 : 0;
  });

  const esc = function (s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
    });
  };
  const ms = function (iso) {
    const p = iso.split('-');
    return Date.UTC(+p[0], +p[1] - 1, +p[2]);
  };
  const pct = function (iso) { return ((ms(iso) - T0) / (T1 - T0)) * 100; };

  /* ---------------- theme ---------------- */

  const root = document.documentElement;
  let stored = null;
  try { stored = localStorage.getItem('attn-theme'); } catch (e) { stored = null; }
  if (stored === 'light' || stored === 'dark') root.setAttribute('data-theme', stored);

  document.getElementById('themebtn').addEventListener('click', function () {
    const isDark = root.getAttribute('data-theme') === 'dark' ||
      (!root.getAttribute('data-theme') && matchMedia('(prefers-color-scheme: dark)').matches);
    const next = isDark ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    try { localStorage.setItem('attn-theme', next); } catch (e) { /* private window */ }
    document.dispatchEvent(new CustomEvent('themechange'));
  });

  /* ---------------- timeline ---------------- */

  const inner = document.getElementById('tl-inner');
  const tip = document.getElementById('tl-tip');
  const legend = document.getElementById('tl-legend');
  let laneFilter = null;

  ORDER.forEach(function (lane) {
    const b = document.createElement('button');
    b.type = 'button';
    b.setAttribute('aria-pressed', 'false');
    b.style.setProperty('--c', 'var(--' + lane + ')');
    b.style.setProperty('--cbg', 'var(--' + lane + '-bg)');
    b.innerHTML = '<span class="sw"></span>' + esc(LANES[lane].name);
    b.title = LANES[lane].note;
    b.addEventListener('click', function () {
      laneFilter = laneFilter === lane ? null : lane;
      Array.prototype.forEach.call(legend.children, function (el, i) {
        el.setAttribute('aria-pressed', ORDER[i] === laneFilter ? 'true' : 'false');
      });
      Array.prototype.forEach.call(inner.querySelectorAll('.tl-dot'), function (d) {
        d.classList.toggle('dim', !!laneFilter && d.dataset.lane !== laneFilter);
      });
    });
    legend.appendChild(b);
  });

  ORDER.forEach(function (lane) {
    const row = document.createElement('div');
    row.className = 'tl-lane';
    row.style.height = '54px';

    const nm = document.createElement('span');
    nm.className = 'lane-name';
    nm.textContent = LANES[lane].name;
    row.appendChild(nm);

    const items = byDate.filter(function (m) { return m.lane === lane; });
    const subrows = [[], [], []];
    const offs = [0, -15, 15];

    items.forEach(function (m) {
      const x = pct(m.date);
      let slot = 0;
      for (let i = 0; i < 3; i++) {
        const last = subrows[i].length ? subrows[i][subrows[i].length - 1] : -99;
        if (x - last > 2.0) { slot = i; break; }
        slot = i;
      }
      subrows[slot].push(x);

      const dot = document.createElement('button');
      dot.type = 'button';
      dot.className = 'tl-dot' + (m.req ? ' req' : '');
      dot.dataset.lane = lane;
      dot.dataset.id = m.id;
      dot.style.left = x + '%';
      dot.style.top = (27 + offs[slot]) + 'px';
      dot.style.setProperty('--c', 'var(--' + lane + ')');
      dot.setAttribute('aria-label', m.name + ', ' + m.dateText);

      const show = function () {
        tip.innerHTML = '<b>' + esc(m.name) + '</b><span>' + esc(m.dateText) + '</span>';
        tip.style.left = Math.min(Math.max(x, 6), 94) + '%';
        tip.style.top = (row.offsetTop + 27 + offs[slot]) + 'px';
        tip.classList.add('show');
      };
      dot.addEventListener('mouseenter', show);
      dot.addEventListener('focus', show);
      dot.addEventListener('mouseleave', function () { tip.classList.remove('show'); });
      dot.addEventListener('blur', function () { tip.classList.remove('show'); });
      dot.addEventListener('click', function () { goTo(m.id); });
      row.appendChild(dot);
    });

    inner.appendChild(row);
  });

  const axis = document.createElement('div');
  axis.className = 'tl-axis';
  for (let y = 2014; y <= 2026; y++) {
    const x = pct(y + '-01-01');
    if (x < -1 || x > 101) continue;
    const t = document.createElement('span');
    t.className = 'tick'; t.style.left = x + '%';
    axis.appendChild(t);
    const l = document.createElement('span');
    l.className = 'yr'; l.style.left = x + '%'; l.textContent = y;
    axis.appendChild(l);
  }
  inner.appendChild(axis);

  function goTo(id) {
    const el = document.getElementById('m-' + id);
    if (!el) return;
    el.scrollIntoView({ block: 'start' });
    el.classList.add('flash');
    setTimeout(function () { el.classList.remove('flash'); }, 1600);
  }

  const byId = {};
  MECHANISMS.forEach(function (m) { byId[m.id] = m; });

  function answersLine(m) {
    if (!m.answers || !byId[m.answers]) return '';
    var t = byId[m.answers];
    return '<p class="answers">Answers a limit of ' +
      '<a href="#m-' + t.id + '">' + esc(t.name) + '</a>, ' + esc(t.dateText) + '</p>';
  }

  function stageStrip(m) {
    var touches = m.touches || [];
    if (touches.indexOf('baseline') >= 0) {
      return '<div class="strip"><h5>What it changes</h5><p class="stripnote">' +
        'Nothing. This is the baseline all six steps are defined by.</p></div>';
    }
    if (!touches.length) {
      return '<div class="strip"><h5>What it changes</h5><p class="stripnote">' +
        'None of the six steps. The mathematics is identical. Only the way it is computed changes.</p></div>';
    }
    var cells = STAGES.map(function (st) {
      var on = touches.indexOf(st.id) >= 0;
      return '<span class="st' + (on ? ' on' : '') + '" title="' + esc(st.d) + '">' +
        esc(st.n) + '</span>';
    }).join('');
    var names = touches.map(function (t) {
      for (var i = 0; i < STAGES.length; i++) if (STAGES[i].id === t) return STAGES[i].n.slice(2);
      return t;
    });
    return '<div class="strip"><h5>What it changes</h5><div class="stagebar">' + cells + '</div>' +
      '<p class="stripnote">Changes <b>' + esc(names.join(' and ')) + '</b>. The other steps are unchanged.</p></div>';
  }

  /* ---------------- cards ---------------- */

  const list = document.getElementById('cardlist');

  ERAS.forEach(function (era) {
    const items = byDate.filter(function (m) { return m.date >= era.from && m.date <= era.to; });
    if (!items.length) return;

    const h = document.createElement('div');
    h.className = 'era';
    h.innerHTML = '<span class="yrs">' + esc(era.yrs) + '</span>' +
      '<h3>' + esc(era.t) + '</h3><p>' + esc(era.d) + '</p>';
    list.appendChild(h);

    items.forEach(function (m) {
      const c = document.createElement('article');
      c.className = 'card';
      c.id = 'm-' + m.id;
      c.style.setProperty('--c', 'var(--' + m.lane + ')');
      c.style.setProperty('--cbg', 'var(--' + m.lane + '-bg)');

      let src = '<a href="' + esc(m.url) + '" target="_blank" rel="noopener">' + esc(m.sub) + '</a>' +
        ' &middot; ' + esc(m.who) + ' &middot; ' + esc(m.src);
      if (m.url2) src += ' &middot; <a href="' + esc(m.url2) + '" target="_blank" rel="noopener">second source</a>';

      let html =
        '<div class="card-head">' +
          '<span class="card-date">' + esc(m.dateText) + (m.dateApprox ? ' &#8776;' : '') + '</span>' +
          '<h4>' + esc(m.name) + '</h4>' +
          '<span class="chip">' + esc(LANES[m.lane].name) + '</span>' +
          (m.req ? '' : '<span class="chip extra">not covered in class</span>') +
        '</div>' +
        '<p class="card-src">' + src + '</p>' +
        answersLine(m) +
        '<div class="block problem"><h5>The problem it answered</h5><p>' + esc(m.problem) + '</p></div>' +
        '<div class="block"><h5>Mechanism</h5><p>' + esc(m.mechanism) + '</p></div>' +
        stageStrip(m) +
        '<div class="two">' +
          '<div class="buys"><h5>Pros</h5><ul>' +
            m.buys.map(function (s) { return '<li>' + esc(s) + '</li>'; }).join('') +
          '</ul></div>' +
          '<div class="costs"><h5>Cons</h5><ul>' +
            m.costs.map(function (s) { return '<li>' + esc(s) + '</li>'; }).join('') +
          '</ul></div>' +
        '</div>' +
        '<p class="pick"><b>Pick it when:</b> ' + esc(m.pick) + '</p>';

      if (m.dateNote) {
        html += '<div class="datenote"><b>On the date and the name.</b> ' + esc(m.dateNote) + '</div>';
      }

      c.innerHTML = html;

      if (m.demo) {
        if (DEMO_HOME[m.demo] === m.id) {
          const d = document.createElement('div');
          d.className = 'demo';
          d.id = 'demo-' + m.demo;
          d.innerHTML = '<h5>Live</h5><p class="cap">' + esc(DEMO_TITLE[m.demo]) + '</p>';
          c.appendChild(d);
          if (window.DEMOS && DEMOS[m.demo]) {
            try { DEMOS[m.demo](d); } catch (e) { d.innerHTML += '<p class="cap">This demo failed to start.</p>'; }
          }
        } else {
          const a = document.createElement('a');
          a.className = 'demolink';
          a.href = '#demo-' + m.demo;
          a.textContent = 'See the live demo: ' + DEMO_TITLE[m.demo].toLowerCase();
          c.appendChild(a);
        }
      }

      list.appendChild(c);
    });
  });

  var primer = document.getElementById('demo-sdpa');
  if (primer && window.DEMOS && DEMOS.sdpa) {
    try { DEMOS.sdpa(primer); } catch (e) { primer.innerHTML += '<p class="cap">This demo failed to start.</p>'; }
  }

  /* ---------------- sources ---------------- */

  const tbl = document.getElementById('srctable');
  let rows = '<thead><tr><th>Date</th><th>Mechanism</th><th>Primary source</th></tr></thead><tbody>';
  byDate.forEach(function (m) {
    rows += '<tr>' +
      '<td class="d">' + esc(m.dateText) + (m.dateApprox ? ' &#8776;' : '') + '</td>' +
      '<td class="n">' + esc(m.name) + '</td>' +
      '<td><a href="' + esc(m.url) + '" target="_blank" rel="noopener">' + esc(m.sub) + '</a><br>' +
        esc(m.who) + ' &middot; ' + esc(m.src) + '</td>' +
      '</tr>';
  });
  tbl.innerHTML = rows + '</tbody>';

  document.getElementById('fact-n').textContent = MECHANISMS.length;
  document.getElementById('fact-src').textContent = MECHANISMS.length;
  document.getElementById('built').textContent =
    MECHANISMS.filter(function (m) { return m.req; }).length + ' of these were on the required list. ' +
    MECHANISMS.filter(function (m) { return !m.req; }).length + ' were added.';
})();
