/* Laya autopilot for the three-lane Reindeer Jump.
 *
 * Reads window.__rj, classifies each obstacle in the oncoming wave, picks a
 * lane, and executes.
 *
 * Division of labour, unchanged in spirit from the single-lane version:
 *   the model decides WHAT each obstacle is;
 *   the harness decides WHICH lane and WHEN to act.
 * The harness reads which lane an obstacle is in and what it is called -- both
 * mechanical facts it already had before -- and applies a fixed preference
 * (an empty lane beats a manoeuvre, never enter a barrier). Nothing pre-ranks
 * the options for the model.
 *
 * The question is the `split` framing, which scored 0.933 in
 * eval/obstacle_class.py (1.000 on named objects, 0.889 on held-out ones):
 * the validated two-option ground-versus-air choice, plus a separate noul for
 * "is this a solid barrier?". Folding the barrier in as a third choice option
 * instead collapses accuracy to 0.27-0.40 -- option count is the sharpest edge
 * on this model, so the third class gets its own question rather than a third
 * label. Both ride in one forward pass.
 */
(function () {
  'use strict';

  const DEFAULTS = {
    endpoint: 'http://127.0.0.1:8000',
    decideAt: 2.6,        // seconds-to-impact at which an obstacle gets classified
    maxInFlight: 4,       // obstacles are independent questions, so pipeline them
    laneBy: 0.55,         // be in the chosen lane this many seconds before impact
    jumpAt: 0.30,
    duckFrom: 0.40,
    duckUntil: -0.12,
    autoRestart: true,
    maxDecisions: 0,
  };

  // Plain English for the game's type names. Naming a thing is not a hint
  // about what it does.
  const NOUN = {
    snowman: 'snowman', gifts: 'pile of gifts', log: 'log',
    garland: 'garland', baubles: 'string of baubles', wall: 'tall ice wall',
  };

  const GROUND_TXT = 'obstacles resting on the snow, such as a snowman, a pile of gifts '
    + 'or a log: they block the space near the ground, so leave the ground to clear them';
  const AIR_TXT = 'obstacles suspended overhead, such as a garland or a string of baubles: '
    + 'they block the space above head height, so lower yourself to pass beneath';

  const QUESTIONS = {
    manoeuvre: {
      type: 'choice',
      instructions: 'Which manoeuvre clears the obstacle described?',
      criteria: { option_a: GROUND_TXT, option_b: AIR_TXT },
    },
    barrier: {
      type: 'noul',
      instructions: 'Is this a solid barrier that fills the whole lane, too tall to leave '
        + 'the ground over and too low to pass beneath?',
    },
  };

  const PLAYER_DEPTH = 0.65;   // the game's collision half-depth
  const WAVE_Z = 3;            // obstacles within this z of each other are one wave

  const api = {
    config: Object.assign({}, DEFAULTS),
    running: false,
    trace: [],
    stats: null,
    onUpdate: null,
    start, stop, reset,
  };

  let rj = null;
  let raf = 0;
  let ids = new WeakMap();
  let nextId = 1;
  let decisions = new Map();
  let inFlight = 0;
  let runIndex = 0;
  let lastSeen = null;

  function freshStats() {
    return {
      decisions: 0, correct: 0, byAction: {}, latencies: [],
      deaths: 0, bestDistance: 0, lastDistance: 0,
      errors: 0, deathLog: [], laneChanges: 0,
    };
  }

  function reset() {
    api.trace = [];
    api.stats = freshStats();
    decisions = new Map();
    ids = new WeakMap();
    nextId = 1;
  }

  /** Obstacles are pooled, so identity is per-appearance: a z that jumped
   *  backwards means this object was respawned and needs a fresh verdict. */
  function idOf(o) {
    const z = o.mesh.position.z;
    let rec = ids.get(o);
    if (!rec || z < rec.lastZ - 1) { rec = { id: nextId++, lastZ: z }; ids.set(o, rec); }
    else rec.lastZ = z;
    return rec.id;
  }

  function truthOf(o) {
    return o.def.kind === 'air' ? 'duck' : o.def.kind === 'block' ? 'block' : 'jump';
  }

  function describe(o) {
    return `There is a ${NOUN[o.type] || o.type} on the track ahead of the running reindeer.`;
  }

  /** The most imminent wave still able to hit us, and what each lane needs. */
  function currentWave() {
    let z = null;
    for (const o of rj.obstacles) {
      const oz = o.mesh.position.z;
      if (oz > o.def.zHalf + PLAYER_DEPTH) continue;
      if (z === null || oz > z) z = oz;
    }
    if (z === null) return null;
    const group = rj.obstacles.filter(o => Math.abs(o.mesh.position.z - z) < WAVE_Z);
    return { z, group };
  }

  // ---------------------------------------------------------------- model

  async function ask(o) {
    const id = idOf(o);
    const truth = truthOf(o);
    const entry = { status: 'pending', truth };
    decisions.set(id, entry);
    inFlight++;
    try {
      const t0 = performance.now();
      const res = await fetch(api.config.endpoint + '/predict', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ state: describe(o), questions: QUESTIONS }),
      });
      if (!res.ok) throw new Error('HTTP ' + res.status + ' ' + (await res.text()).slice(0, 120));
      const body = await res.json();
      const wall = performance.now() - t0;
      const man = body.answers.manoeuvre;
      const pBlock = body.answers.barrier.noul;

      // one distribution over the three classes, for the HUD and the trace
      const probs = {
        jump: man.probabilities.option_a * (1 - pBlock),
        duck: man.probabilities.option_b * (1 - pBlock),
        block: pBlock,
      };
      const klass = pBlock >= 0.5 ? 'block'
        : (man.choice === 'option_a' ? 'jump' : 'duck');

      Object.assign(entry, {
        status: 'done', action: klass, probs,
        confidence: probs[klass],
        correct: klass === truth,
        wallMs: wall,
      });

      const st = api.stats;
      st.decisions++;
      if (entry.correct) st.correct++;
      st.byAction[klass] = (st.byAction[klass] || 0) + 1;
      st.latencies.push(wall);

      api.trace.push({
        run: runIndex,
        distance: Math.floor(rj.state.distance),
        speed: Number(rj.state.speed.toFixed(1)),
        obstacle: o.type, lane: o.lane, kind: o.def.kind,
        truth, pred: klass, correct: entry.correct,
        p_block: Number(pBlock.toFixed(4)),
        manoeuvre_choice: man.choice,
        manoeuvre_confidence: man.confidence,
        probabilities: probs,
        server_ms: body.latency_ms,
        wall_ms: Number(wall.toFixed(1)),
      });
      if (api.onUpdate) api.onUpdate(entry);
    } catch (err) {
      entry.status = 'error';
      entry.error = String((err && err.message) || err);
      entry.action = 'jump';       // a guess, recorded as an error, never graded as skill
      api.stats.errors++;
      if (api.onUpdate) api.onUpdate(entry);
    } finally {
      inFlight--;
    }
  }

  // ---------------------------------------------------------------- loop

  function reachedLimit() {
    return api.config.maxDecisions > 0 && api.stats.decisions >= api.config.maxDecisions;
  }

  function tick() {
    raf = requestAnimationFrame(tick);
    if (!api.running || !rj) return;
    const s = rj.state;

    if (s.mode !== 'playing') {
      rj.setDuck(false);
      if (s.mode === 'over') {
        const d = Math.floor(s.distance);
        if (d !== api.stats.lastDistance) {
          api.stats.deaths++;
          api.stats.lastDistance = d;
          api.stats.bestDistance = Math.max(api.stats.bestDistance, d);
          const v = lastSeen && decisions.get(lastSeen.id);
          api.stats.deathLog.push({
            run: runIndex, distance: d, lane: s.lane, x: Number(s.x.toFixed(2)),
            obstacle: lastSeen ? lastSeen.type : null,
            verdict: v ? (v.status === 'pending' ? 'pending' : (v.action || v.status)) : 'none',
            truth: v ? v.truth : null,
            correct: v ? v.action === v.truth : null,
            ttc: lastSeen ? Number(lastSeen.ttc.toFixed(3)) : null,
          });
          if (api.onUpdate) api.onUpdate(null);
        }
        if (api.config.autoRestart && !reachedLimit()) { runIndex++; rj.startGame(); }
      }
      return;
    }

    if (s.distance > api.stats.bestDistance) api.stats.bestDistance = Math.floor(s.distance);

    if (decisions.size > 256) {
      const keep = new Map();
      for (const [k, v] of decisions) if (k > nextId - 48) keep.set(k, v);
      decisions = keep;
    }

    // classify everything already inside the horizon, pipelined
    if (!reachedLimit()) {
      for (const cand of rj.obstacles) {
        if (inFlight >= api.config.maxInFlight) break;
        const cz = cand.mesh.position.z;
        if (cz > cand.def.zHalf + PLAYER_DEPTH) continue;
        if (-cz / Math.max(1e-3, s.speed) > api.config.decideAt) continue;
        if (!decisions.has(idOf(cand))) ask(cand);
      }
    }

    const wave = currentWave();
    if (!wave) { rj.setDuck(false); return; }

    const ttc = -wave.z / Math.max(1e-3, s.speed);

    // what the model says each lane holds; absent means the lane is empty,
    // which the harness can see for itself
    const need = [null, null, null];
    for (const o of wave.group) {
      const d = decisions.get(idOf(o));
      need[o.lane] = d ? (d.status === 'pending' ? 'unknown' : d.action) : 'unknown';
      if (o.lane === s.lane) {
        lastSeen = { id: idOf(o), type: o.type, ttc };
      }
    }

    // prefer an empty lane, then a manoeuvre we know, then an unclassified one;
    // never a lane the model called a barrier
    const usable = [0, 1, 2].filter(l => need[l] !== 'block');
    const rank = l => (need[l] === null ? 0 : need[l] === 'unknown' ? 2 : 1);
    const target = usable.length
      ? usable.reduce((a, b) => {
          const ra = rank(a), rb = rank(b);
          if (ra !== rb) return ra < rb ? a : b;
          return Math.abs(b - s.lane) < Math.abs(a - s.lane) ? b : a;
        })
      : s.lane;                       // every lane called blocked: hold and hope

    if (s.lane !== target) {
      rj.moveLane(Math.sign(target - s.lane));
      api.stats.laneChanges++;
    }

    // Only commit to a manoeuvre once we are actually standing in the lane it
    // belongs to -- jumping while still sliding clears the wrong obstacle.
    const inLane = Math.abs(s.x - rj.LANES[target]) < 0.12;
    const req = need[target];
    if (!inLane || req === null || req === 'unknown' || req === 'block') {
      rj.setDuck(false);
      return;
    }
    if (req === 'jump') {
      rj.setDuck(false);
      if (ttc <= api.config.jumpAt && s.onGround) rj.jump();
    } else if (req === 'duck') {
      rj.setDuck(ttc <= api.config.duckFrom && ttc > api.config.duckUntil);
    }
  }

  function start(cfg) {
    Object.assign(api.config, cfg || {});
    rj = window.__rj;
    if (!rj) throw new Error('window.__rj not found -- is the game loaded?');
    if (!rj.LANES) throw new Error('this autopilot needs the three-lane game');
    if (!api.stats) reset();
    api.running = true;
    if (rj.state.mode !== 'playing') rj.startGame();
    if (!raf) raf = requestAnimationFrame(tick);
    return api;
  }

  function stop() {
    api.running = false;
    if (rj) rj.setDuck(false);
    if (raf) { cancelAnimationFrame(raf); raf = 0; }
    return api.trace;
  }

  reset();
  window.__autopilot = api;
})();
