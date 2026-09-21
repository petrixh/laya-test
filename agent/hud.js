/* Live telemetry panel for the Laya autopilot.
 *
 * Colours come from the validated dark-mode categorical palette, checked
 * against this panel's own surface (#0d1630) rather than a generic dark:
 *   node scripts/validate_palette.js "#3987e5,#d95926,#199e70" \
 *        --mode dark --surface "#0d1630" --pairs all   -> all checks pass
 *
 * The panel is a live readout, not an explorable chart, so there is no hover
 * layer; every decision is kept in __autopilot.trace, which is the table view
 * and what the Playwright driver exports.
 */
(function () {
  'use strict';

  const ACTION_SLOT = { jump: 'var(--s1)', duck: 'var(--s2)', block: 'var(--s3)' };
  const ORDER = ['jump', 'duck', 'block'];
  const HEX = { jump: '#3987e5', duck: '#d95926', block: '#199e70' };
  const SPARK_N = 60;

  const css = `
  #laya-hud {
    --surface: #0d1630;
    --raised: #142046;
    --ink: #ffffff;
    --ink-2: #c3c2b7;
    --muted: #898781;
    --rule: #2c3358;
    --s1: #3987e5;   /* jump  */
    --s2: #d95926;   /* duck  */
    --s3: #199e70;   /* run   */
    --latency: #9085e9;
    --lane: #d55181;       /* one hue: these bars are one measure over three lanes */
    --good: #0ca30c;
    --warning: #fab219;
    --critical: #d03b3b;

    position: fixed; top: 0; right: 0; bottom: 0;
    width: 310px; z-index: 50;
    background: color-mix(in srgb, var(--surface) 92%, transparent);
    backdrop-filter: blur(8px);
    border-left: 1px solid var(--rule);
    color: var(--ink);
    font: 12px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
    display: flex; flex-direction: column;
    overflow: hidden;
  }
  #laya-hud .hd {
    flex: 0 0 auto; padding: 10px 14px 9px; border-bottom: 1px solid var(--rule);
    display: flex; align-items: baseline; justify-content: space-between; gap: 8px;
  }
  #laya-hud .hd b { font-size: 12px; letter-spacing: 2px; }
  #laya-hud .hd span { font-size: 10px; color: var(--muted); letter-spacing: 1px; }
  #laya-hud .sec { padding: 10px 14px; border-bottom: 1px solid var(--rule); flex: 0 0 auto; }
  #laya-hud .lbl {
    font-size: 9.5px; letter-spacing: 1.6px; color: var(--muted);
    text-transform: uppercase; margin-bottom: 8px;
    display: flex; justify-content: space-between; align-items: baseline;
  }
  #laya-hud .tag {
    font-size: 9px; letter-spacing: 1.2px; padding: 1px 6px; border-radius: 3px;
    background: var(--raised); color: var(--muted);
  }
  #laya-hud .tag.forced { background: var(--warning); color: #201a04; font-weight: 700; }
  #laya-hud .tag.bad { background: var(--critical); color: #fff; font-weight: 700; }

  /* stage two: probability across the three lanes. one measure, so one hue. */
  #laya-hud .lane-row {
    display: grid; grid-template-columns: 14px 1fr 34px; gap: 8px;
    align-items: center; margin-bottom: 6px;
  }
  #laya-hud .lane-row:last-child { margin-bottom: 0; }
  #laya-hud .lane-row b { font-size: 10px; color: var(--muted); font-weight: 600; }
  #laya-hud .lane-row .trk { height: 8px; background: var(--raised); border-radius: 4px; overflow: hidden; }
  #laya-hud .lane-row .fill {
    height: 100%; width: 0; border-radius: 4px; background: var(--lane); opacity: .45;
    transition: width .18s ease-out, opacity .18s;
  }
  #laya-hud .lane-row i {
    font-style: normal; font-size: 10.5px; color: var(--ink-2);
    font-variant-numeric: tabular-nums; text-align: right;
  }
  #laya-hud .lane-row.pick .fill { opacity: 1; }
  #laya-hud .lane-row.pick b, #laya-hud .lane-row.pick i { color: var(--ink); font-weight: 700; }
  #laya-hud .lane-row.wall b { color: var(--critical); }

  /* hero: the action currently being executed */
  #laya-hud .hero { display: flex; align-items: baseline; gap: 10px; }
  #laya-hud .hero .act { font-size: 30px; font-weight: 700; letter-spacing: 1px; line-height: 1; }
  #laya-hud .hero .cf { font-size: 11px; color: var(--ink-2); }

  /* lane strip: which lane the reindeer is heading for */
  #laya-hud .lanes { display: flex; gap: 6px; margin-top: 10px; }
  #laya-hud .lanes span {
    flex: 1; height: 7px; border-radius: 4px; background: var(--raised);
    transition: background .15s;
  }
  #laya-hud .lanes span.on { background: var(--ink); }
  #laya-hud .lanes span.no { background: var(--critical); }

  /* probability bars - direct-labelled, so identity never rests on colour */
  #laya-hud .bar { margin-bottom: 8px; }
  #laya-hud .bar:last-child { margin-bottom: 0; }
  #laya-hud .bar .top {
    display: flex; justify-content: space-between;
    font-size: 11px; color: var(--ink-2); margin-bottom: 3px;
  }
  #laya-hud .bar .top em { font-style: normal; color: var(--ink); letter-spacing: .5px; }
  #laya-hud .bar .top i { font-style: normal; font-variant-numeric: tabular-nums; }
  #laya-hud .bar .trk { height: 8px; background: var(--raised); border-radius: 4px; overflow: hidden; }
  #laya-hud .bar .fill {
    height: 100%; border-radius: 4px; width: 0;
    transition: width .18s ease-out;
  }
  #laya-hud .bar.win .top em { font-weight: 700; }

  /* stat tiles */
  #laya-hud .tiles { display: grid; grid-template-columns: 1fr 1fr; gap: 10px 12px; }
  #laya-hud .tile .v {
    font-size: 21px; font-weight: 600; line-height: 1.1;
    font-variant-numeric: tabular-nums;
  }
  #laya-hud .tile .k { font-size: 9.5px; letter-spacing: 1.2px; color: var(--muted); text-transform: uppercase; }
  #laya-hud .tile .v small { font-size: 11px; color: var(--muted); font-weight: 400; }

  #laya-hud canvas { display: block; width: 100%; height: 46px; }

  /* decision log - glyph + label, never colour alone */
  #laya-hud .log {
    flex: 1 1 auto; min-height: 0; overflow-y: auto;
    padding: 10px 14px 12px;
    scrollbar-width: thin; scrollbar-color: var(--rule) transparent;
  }
  #laya-hud .row {
    display: grid; grid-template-columns: 14px 1fr auto auto; gap: 8px;
    align-items: baseline; padding: 3px 0; font-size: 11px;
    border-bottom: 1px solid color-mix(in srgb, var(--rule) 50%, transparent);
  }
  #laya-hud .row .g { font-weight: 700; }
  #laya-hud .row .ok { color: var(--good); }
  #laya-hud .row .no { color: var(--critical); }
  #laya-hud .row .ob { color: var(--ink-2); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  #laya-hud .row .pr { letter-spacing: .5px; }
  #laya-hud .row .ms { color: var(--muted); font-variant-numeric: tabular-nums; }
  @media (max-width: 760px) { #laya-hud { display: none; } }
  `;

  function el(tag, cls, html) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html != null) n.innerHTML = html;
    return n;
  }

  function build() {
    const style = el('style'); style.textContent = css;
    document.head.appendChild(style);

    const root = el('div'); root.id = 'laya-hud';
    root.appendChild(el('div', 'hd', '<b>LAYA AUTOPILOT</b><span id="lh-dev">connecting</span>'));

    const hero = el('div', 'sec');
    hero.appendChild(el('div', 'lbl', 'Reading'));
    hero.appendChild(el('div', 'hero',
      '<span class="act" id="lh-act">--</span><span class="cf" id="lh-cf"></span>'));
    hero.appendChild(el('div', 'lanes', '<span data-l="0"></span><span data-l="1"></span>'
      + '<span data-l="2"></span>'));
    root.appendChild(hero);

    const bars = el('div', 'sec');
    bars.appendChild(el('div', 'lbl', 'Action probability'));
    for (const a of ORDER) {
      const b = el('div', 'bar');
      b.id = 'lh-bar-' + a;
      b.innerHTML =
        `<div class="top"><em>${a.toUpperCase()}</em><i id="lh-p-${a}">0.00</i></div>` +
        `<div class="trk"><div class="fill" id="lh-f-${a}" style="background:${ACTION_SLOT[a]}"></div></div>`;
      bars.appendChild(b);
    }
    root.appendChild(bars);

    const lane = el('div', 'sec');
    lane.id = 'lh-lanesec';
    lane.appendChild(el('div', 'lbl',
      '<span>Lane choice</span><span class="tag" id="lh-lanetag">waiting</span>'));
    for (let i = 0; i < 3; i++) {
      const r = el('div', 'lane-row');
      r.id = 'lh-lane-' + i;
      r.innerHTML = `<b>${['L', 'M', 'R'][i]}</b>`
        + `<div class="trk"><div class="fill" id="lh-lf-${i}"></div></div>`
        + `<i id="lh-lp-${i}">--</i>`;
      lane.appendChild(r);
    }
    root.appendChild(lane);

    const lat = el('div', 'sec');
    lat.appendChild(el('div', 'lbl', 'Inference latency &middot; ms'));
    const cv = el('canvas'); cv.id = 'lh-spark'; lat.appendChild(cv);
    lat.appendChild(el('div', 'tiles',
      '<div class="tile"><div class="v" id="lh-p50">--</div><div class="k">p50 ms</div></div>' +
      '<div class="tile"><div class="v" id="lh-p95">--</div><div class="k">p95 ms</div></div>'));
    root.appendChild(lat);

    const st = el('div', 'sec');
    st.appendChild(el('div', 'lbl', 'Run'));
    st.appendChild(el('div', 'tiles',
      '<div class="tile"><div class="v" id="lh-acc">--</div><div class="k">accuracy</div></div>' +
      '<div class="tile"><div class="v" id="lh-dec">0</div><div class="k">decisions</div></div>' +
      '<div class="tile"><div class="v" id="lh-dist">0<small> m</small></div><div class="k">distance</div></div>' +
      '<div class="tile"><div class="v" id="lh-crash">0</div><div class="k">crashes</div></div>'));
    root.appendChild(st);

    const log = el('div', 'log');
    log.appendChild(el('div', 'lbl', 'Decisions'));
    log.appendChild(el('div', '', '<div id="lh-log"></div>'));
    root.appendChild(log);

    document.body.appendChild(root);
    return root;
  }

  function quantile(sorted, q) {
    if (!sorted.length) return null;
    const i = Math.min(sorted.length - 1, Math.floor(q * sorted.length));
    return sorted[i];
  }

  function drawSpark(cv, values) {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const w = cv.clientWidth, h = cv.clientHeight;
    if (!w || !h) return;
    cv.width = w * dpr; cv.height = h * dpr;
    const g = cv.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    if (values.length < 2) return;

    const pad = 3;
    const max = Math.max.apply(null, values) * 1.12;
    const min = 0;                       // latency is a magnitude: baseline at zero
    const x = i => pad + (i / (values.length - 1)) * (w - pad * 2);
    const y = v => h - pad - ((v - min) / (max - min || 1)) * (h - pad * 2);

    // recessive baseline
    g.strokeStyle = '#2c3358'; g.lineWidth = 1;
    g.beginPath(); g.moveTo(0, h - pad); g.lineTo(w, h - pad); g.stroke();

    // soft area under the line, then the 2px line itself
    const grad = g.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, 'rgba(144,133,233,.28)');
    grad.addColorStop(1, 'rgba(144,133,233,0)');
    g.beginPath();
    g.moveTo(x(0), y(values[0]));
    values.forEach((v, i) => g.lineTo(x(i), y(v)));
    g.lineTo(x(values.length - 1), h - pad); g.lineTo(x(0), h - pad); g.closePath();
    g.fillStyle = grad; g.fill();

    g.beginPath();
    g.moveTo(x(0), y(values[0]));
    values.forEach((v, i) => g.lineTo(x(i), y(v)));
    g.strokeStyle = '#9085e9'; g.lineWidth = 2;
    g.lineJoin = 'round'; g.lineCap = 'round';
    g.stroke();

    // last point gets a marker with a surface ring, per mark specs
    const lx = x(values.length - 1), ly = y(values[values.length - 1]);
    g.beginPath(); g.arc(lx, ly, 4.5, 0, Math.PI * 2);
    g.fillStyle = '#9085e9'; g.fill();
    g.lineWidth = 2; g.strokeStyle = '#0d1630'; g.stroke();
  }

  function mount(autopilot, endpoint) {
    build();
    const base = endpoint || autopilot.config.endpoint;
    const $ = id => document.getElementById(id);
    const spark = $('lh-spark');
    const logEl = $('lh-log');
    let lastLogged = 0;

    fetch(base + '/info')
      .then(r => r.json())
      .then(i => { $('lh-dev').textContent = (i.subfolder || 'base') + ' · ' + String(i.device).toUpperCase(); })
      .catch(() => { $('lh-dev').textContent = 'no service'; });

    function paintDecision(d) {
      if (!d || d.status !== 'done') return;
      $('lh-act').textContent = d.action.toUpperCase();
      $('lh-act').style.color = HEX[d.action] || '#fff';
      $('lh-cf').textContent = 'confidence ' + d.confidence.toFixed(3);
      for (const a of ORDER) {
        const p = (d.probs && d.probs[a]) || 0;
        $('lh-p-' + a).textContent = p.toFixed(2);
        $('lh-f-' + a).style.width = Math.max(1.5, p * 100) + '%';
        $('lh-bar-' + a).classList.toggle('win', a === d.action);
      }
    }

    function paintStats() {
      const st = autopilot.stats;
      const sorted = st.latencies.slice().sort((a, b) => a - b);
      $('lh-p50').textContent = sorted.length ? Math.round(quantile(sorted, 0.5)) : '--';
      $('lh-p95').textContent = sorted.length ? Math.round(quantile(sorted, 0.95)) : '--';
      $('lh-dec').textContent = st.decisions;
      $('lh-crash').textContent = st.deaths;
      $('lh-acc').textContent = st.decisions
        ? (100 * st.correct / st.decisions).toFixed(1) + '%' : '--';
      if (st.laneDecisions) {
        $('lh-lanesec').querySelector('.lbl > span').textContent =
          `Lane choice · ${st.laneDecisions}`;
      }
      // say plainly when stage two is not being consulted, so the panel is not
      // mistaken for a live decision when it is only showing the last one
      // The reindeer never moves except on an answer, so the tag tracks the
      // real state: not needed, waiting on the model, or answered.
      const st2 = autopilot.laneNeed;
      const tag = $('lh-lanetag');
      if (st2 === 'not needed' && tag.textContent !== 'not needed') {
        tag.textContent = 'not needed'; tag.className = 'tag';
      } else if (st2 === 'waiting' && tag.textContent !== 'asking…') {
        tag.textContent = 'asking\u2026'; tag.className = 'tag forced';
      }
      const rj = window.__rj;
      if (rj) {
        $('lh-dist').innerHTML = Math.floor(rj.state.distance) + '<small> m</small>';
        // Lane strip: filled = heading there, red = the model called THIS
        // wave's obstacle in that lane a barrier.
        //
        // Two wrong versions preceded this one. Reading o.def.kind showed the
        // game's own answer, so walls went red before the model had spoken.
        // Scanning the whole trace for past 'block' verdicts accumulated them
        // by lane, so within about forty decisions every occupied lane was red
        // whatever the model had said. The autopilot now publishes the current
        // wave's readings and the strip shows only those.
        const need = autopilot.currentNeed;
        const blocked = new Set();
        if (need) for (let l = 0; l < need.length; l++) if (need[l] === 'block') blocked.add(l);
        for (const node of document.querySelectorAll('#laya-hud .lanes span')) {
          const l = Number(node.dataset.l);
          node.classList.toggle('on', l === rj.state.lane);
          node.classList.toggle('no', blocked.has(l) && l !== rj.state.lane);
        }
      }
      drawSpark(spark, st.latencies.slice(-SPARK_N));

      while (lastLogged < autopilot.trace.length) {
        const t = autopilot.trace[lastLogged++];
        if (t.stage === 'lane') continue;      // lane calls have their own panel
        const row = el('div', 'row');
        row.innerHTML =
          `<span class="g ${t.correct ? 'ok' : 'no'}">${t.correct ? '✓' : '✗'}</span>` +
          `<span class="ob">${t.obstacle}</span>` +
          `<span class="pr">${t.pred}</span>` +
          `<span class="ms">${Math.round(t.wall_ms)}</span>`;
        logEl.prepend(row);
        while (logEl.children.length > 9) logEl.removeChild(logEl.lastChild);
      }
    }

    function paintLane(d, need) {
      if (!d || d.status !== 'done') return;
      const tag = $('lh-lanetag');
      // "forced" = the lane the reindeer is standing in was called a barrier,
      // so stage two had to move it. Otherwise the choice was optional.
      tag.textContent = d.contradiction ? 'CHOSE A WALL' : 'MODEL PICKED';
      tag.className = 'tag' + (d.contradiction ? ' bad' : ' forced');
      for (let i = 0; i < 3; i++) {
        const p = d.probs[i] || 0;
        $('lh-lp-' + i).textContent = p.toFixed(2);
        $('lh-lf-' + i).style.width = Math.max(1.5, p * 100) + '%';
        const row = $('lh-lane-' + i);
        row.classList.toggle('pick', i === d.lane);
        row.classList.toggle('wall', need && need[i] === 'block');
      }
    }

    autopilot.onUpdate = paintDecision;
    autopilot.onLane = paintLane;
    // stop() should stop the panel too, or the canvas keeps redrawing forever
    const wrappedStop = autopilot.stop;
    autopilot.stop = function () {
      clearInterval(timer);
      return wrappedStop.apply(this, arguments);
    };

    const timer = setInterval(paintStats, 120);
    addEventListener('resize', () => drawSpark(spark, autopilot.stats.latencies.slice(-SPARK_N)));
  }

  window.__layaHud = { mount };
})();
