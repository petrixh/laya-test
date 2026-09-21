/* Laya autopilot for Reindeer Jump.
 *
 * Reads window.__rj (exposed by the game), asks the Laya service what to do
 * about the nearest obstacle, and executes the answer.
 *
 * Design note -- the model decides WHAT, this file decides WHEN.
 * Inference costs ~280ms on CPU but the collision window at top speed is only
 * 65ms wide, so reacting on arrival of the response would be hopeless. Instead
 * we classify each obstacle once, early (it is visible ~3-6s out), cache the
 * verdict, and fire the action on a timer as the obstacle arrives. That is also
 * how you would build it for real: a System 1 model picks the action, trivial
 * logic handles the reflex timing.
 *
 * Every decision is graded against ground truth derived from the game's own
 * collision geometry, so the run doubles as a self-labelling benchmark.
 */
(function () {
  'use strict';

  const DEFAULTS = {
    endpoint: 'http://127.0.0.1:8000',
    framing: 'guided',    // 'guided' | 'action'  -- see QUESTION_FOR below
    encoding: 'json',     // 'json' | 'nl'  -- state shape for the 'action' framing
    decideAt: 2.6,        // seconds-to-impact at which we ask the model
    maxInFlight: 3,       // concurrent classifications (obstacles are independent)
    jumpAt: 0.30,         // seconds-to-impact at which a 'jump' verdict fires
    duckFrom: 0.40,       // duck is held across this window, in seconds
    duckUntil: -0.12,
    autoRestart: true,
    maxDecisions: 0,      // 0 = unlimited
  };

  const ACTIONS = {
    jump: 'leap over a ground-level obstacle such as a snowman, a pile of gifts or a log',
    duck: 'crouch under a hanging obstacle such as a garland or a string of baubles',
    run: 'keep running normally, nothing is close enough to need action yet',
  };

  // The 'guided' framing, chosen by eval/prompt_sweep.py and confirmed by
  // eval/prompt_confirm.py on 36 scenes: 0.97 overall and 0.94 on objects the
  // criteria never name, against 0.35 for the 'action' framing below.
  //
  // Three things earned that, all static and instance-independent -- the state
  // still only names the object, and nothing pre-ranks the options:
  //   - neutral label names. 'jump'/'duck' as label tokens actively hurt
  //     (marginal accuracy 0.43 against 0.67 for neutral names).
  //   - criteria that give both the examples and the mechanism (0.69 against
  //     0.45 for bare criteria).
  //   - two options, not three. Offering a 'nothing to do yet' option costs
  //     0.14 accuracy, and the harness already knows when there is no obstacle.
  const GUIDED = {
    option_a: 'obstacles resting on the snow, such as a snowman, a pile of gifts or '
      + 'a log: they block the space near the ground, so leave the ground to clear them',
    option_b: 'obstacles suspended overhead, such as a garland or a string of baubles: '
      + 'they block the space above head height, so lower yourself to pass beneath',
  };
  const GUIDED_TO_ACTION = { option_a: 'jump', option_b: 'duck' };

  // Plain English for the game's internal type names. Naming the object is not
  // a hint about where it sits.
  const NOUN = {
    snowman: 'snowman', gifts: 'pile of gifts', log: 'log',
    garland: 'garland', baubles: 'string of baubles',
  };

  const QUESTION_KEY = 'action';

  // From the game's collision test: it fires while |obstacle.z| < zHalf + PLAYER_DEPTH,
  // so an obstacle stays dangerous for another 1.1m AFTER it draws level with us.
  const PLAYER_DEPTH = 0.65;

  const api = {
    config: Object.assign({}, DEFAULTS),
    running: false,
    trace: [],
    stats: null,
    onUpdate: null,     // hook for the HUD
    start, stop, reset,
  };

  let rj = null;
  let raf = 0;
  let threatIds = new WeakMap();
  let nextThreatId = 1;
  let decisions = new Map();   // threatId -> {status, action, probs, confidence, latency, truth}
  let inFlight = 0;
  let runIndex = 0;
  let lastSeen = null;      // snapshot of the threat we were acting on, for death forensics

  function freshStats() {
    return {
      decisions: 0, correct: 0, byAction: {}, latencies: [],
      deaths: 0, bestDistance: 0, lastDistance: 0,
      errors: 0, timeouts: 0, deathLog: [],
    };
  }

  function reset() {
    api.trace = [];
    api.stats = freshStats();
    decisions = new Map();
    threatIds = new WeakMap();
    nextThreatId = 1;
  }

  // ---------------------------------------------------------------- state

  /** Nearest obstacle that can still hit us.
   *
   * Note the exit test: not `z >= 0`. An obstacle level with the reindeer is
   * still inside the collision box, and dropping it there made the autopilot
   * release its duck a few centimetres too early and clip the garland it had
   * just correctly classified. It stays the active threat until it is clear. */
  function nearestThreat() {
    let best = null;
    for (const o of rj.obstacles) {
      const z = o.mesh.position.z;
      if (z > o.def.zHalf + PLAYER_DEPTH) continue;   // fully behind us
      // largest z among the candidates is the most imminent
      if (!best || z > best.mesh.position.z) best = o;
    }
    return best;
  }

  function threatId(o) {
    // The game pools and reuses obstacle objects, so object identity is not
    // appearance identity. Obstacles only ever travel forwards (z increases);
    // a z that jumped backwards means this object was respawned at SPAWN_Z and
    // deserves a fresh id, or we would reuse the previous verdict.
    const z = o.mesh.position.z;
    let rec = threatIds.get(o);
    if (!rec || z < rec.lastZ - 1) {
      rec = { id: nextThreatId++, lastZ: z };
      threatIds.set(o, rec);
    } else {
      rec.lastZ = z;
    }
    return rec.id;
  }

  /** What the game's collision geometry says is the right answer. */
  function groundTruth(o) {
    return o.def.kind === 'air' ? 'duck' : 'jump';
  }

  function describe(o, ttc) {
    const s = rj.state;
    const kind = o.def.kind === 'air' ? 'hanging in the air' : 'sitting on the ground';
    if (api.config.framing === 'guided') {
      // Names the object and nothing else. Whether it sits or hangs is exactly
      // what the model has to work out.
      return `There is a ${NOUN[o.type] || o.type} on the track ahead of the running reindeer.`;
    }
    if (api.config.encoding === 'nl') {
      return (
        `The reindeer is running at ${s.speed.toFixed(0)} metres per second and is ` +
        `${s.onGround ? 'on the ground' : 'in the air'}. ` +
        `Ahead there is a ${o.type}, ${kind}, ` +
        `${(-o.mesh.position.z).toFixed(1)} metres away, ` +
        `about ${ttc.toFixed(2)} seconds from hitting the reindeer. ` +
        `It spans from ${o.def.yMin.toFixed(2)} to ${o.def.yMax.toFixed(2)} metres above the snow. ` +
        `The reindeer is ${(1.75).toFixed(2)} metres tall standing and 0.95 metres tall crouching.`
      );
    }
    return {
      obstacle: o.type,
      obstacle_position: kind,
      obstacle_bottom_m: Number(o.def.yMin.toFixed(2)),
      obstacle_top_m: Number(o.def.yMax.toFixed(2)),
      distance_m: Number((-o.mesh.position.z).toFixed(1)),
      seconds_to_impact: Number(ttc.toFixed(2)),
      speed_mps: Number(s.speed.toFixed(1)),
      reindeer_airborne: !s.onGround,
      reindeer_height_standing_m: 1.75,
      reindeer_height_crouching_m: 0.95,
    };
  }

  const QUESTIONS = {
    guided: {
      [QUESTION_KEY]: {
        type: 'choice',
        instructions: 'Which manoeuvre clears the obstacle described?',
        criteria: GUIDED,
      },
    },
    action: {
      [QUESTION_KEY]: {
        type: 'choice',
        instructions:
          'A running reindeer must get past the obstacle ahead without touching it. ' +
          'What should it do?',
        criteria: ACTIONS,
      },
    },
  };

  // ---------------------------------------------------------------- model

  async function ask(o, ttc) {
    const id = threatId(o);
    const truth = groundTruth(o);
    const entry = { status: 'pending', truth, asked: performance.now(), ttcAtAsk: ttc };
    decisions.set(id, entry);
    inFlight++;

    try {
      const t0 = performance.now();
      const questions = QUESTIONS[api.config.framing] || QUESTIONS.action;
      const res = await fetch(api.config.endpoint + '/predict', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ state: describe(o, ttc), questions }),
      });
      if (!res.ok) throw new Error('HTTP ' + res.status + ' ' + (await res.text()).slice(0, 120));
      const body = await res.json();
      const ans = body.answers[QUESTION_KEY];
      const wall = performance.now() - t0;

      // map the neutral option names back onto game actions for the HUD and grading
      const action = GUIDED_TO_ACTION[ans.choice] || ans.choice;
      const probs = {};
      for (const k of Object.keys(ans.probabilities)) {
        probs[GUIDED_TO_ACTION[k] || k] = ans.probabilities[k];
      }

      Object.assign(entry, {
        status: 'done',
        action: action,
        probs: probs,
        confidence: ans.confidence,
        serverMs: body.latency_ms,
        wallMs: wall,
        correct: action === truth,
      });

      const st = api.stats;
      st.decisions++;
      if (entry.correct) st.correct++;
      st.byAction[action] = (st.byAction[action] || 0) + 1;
      st.latencies.push(wall);

      api.trace.push({
        run: runIndex,
        t: Number((rj.state.time).toFixed(2)),
        distance: Math.floor(rj.state.distance),
        speed: Number(rj.state.speed.toFixed(1)),
        obstacle: o.type,
        kind: o.def.kind,
        ttc_at_ask: Number(ttc.toFixed(3)),
        framing: api.config.framing,
        truth,
        pred: action,
        raw_choice: ans.choice,
        correct: entry.correct,
        confidence: ans.confidence,
        probabilities: probs,
        server_ms: body.latency_ms,
        wall_ms: Number(wall.toFixed(1)),
      });
      if (api.onUpdate) api.onUpdate(entry);
    } catch (err) {
      entry.status = 'error';
      entry.error = String(err && err.message || err);
      api.stats.errors++;
      // Falling back to ground truth would flatter the model; fall back to the
      // safest generic action instead and record it as an error.
      entry.action = 'run';
      if (api.onUpdate) api.onUpdate(entry);
    } finally {
      inFlight--;
    }
  }

  // ---------------------------------------------------------------- loop

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
          // Forensics: a death with a correct verdict means the execution
          // window was wrong, not the model. Worth telling apart.
          const v = lastSeen && decisions.get(lastSeen.id);
          api.stats.deathLog.push({
            run: runIndex,
            distance: d,
            obstacle: lastSeen ? lastSeen.type : null,
            kind: lastSeen ? lastSeen.kind : null,
            ttc: lastSeen ? Number(lastSeen.ttc.toFixed(3)) : null,
            verdict: v ? (v.status === 'pending' ? 'pending' : (v.action || v.status)) : 'none',
            truth: v ? v.truth : (lastSeen ? (lastSeen.kind === 'air' ? 'duck' : 'jump') : null),
            correct: v ? v.action === v.truth : null,
            airborne: lastSeen ? lastSeen.airborne : null,
            duck: lastSeen ? Number(lastSeen.duck.toFixed(2)) : null,
          });
          if (api.onUpdate) api.onUpdate(null);
        }
        if (api.config.autoRestart && !reachedLimit()) {
          runIndex++;
          rj.startGame();
        }
      }
      return;
    }

    if (decisions.size > 256) {
      // verdicts for long-passed obstacles are dead weight
      const keep = new Map();
      for (const [k, v] of decisions) if (k > nextThreatId - 32) keep.set(k, v);
      decisions = keep;
    }

    // track the furthest we have got, not only at the moment of death
    if (s.distance > api.stats.bestDistance) api.stats.bestDistance = Math.floor(s.distance);

    // Pre-classify every obstacle already inside the decision horizon, not just
    // the next one. Real inference is ~1s while obstacles can be 0.66s apart at
    // top speed, so waiting until an obstacle is next in line means its verdict
    // lands after it has already hit us. They are independent questions, so
    // pipeline them.
    if (!reachedLimit()) {
      for (const cand of rj.obstacles) {
        if (inFlight >= api.config.maxInFlight) break;
        const cz = cand.mesh.position.z;
        if (cz > cand.def.zHalf + PLAYER_DEPTH) continue;
        const cttc = -cz / Math.max(1e-3, s.speed);
        if (cttc > api.config.decideAt) continue;
        const cid = threatId(cand);
        if (!decisions.has(cid)) ask(cand, cttc);
      }
    }

    const o = nearestThreat();
    if (!o) { rj.setDuck(false); return; }

    const ttc = -o.mesh.position.z / Math.max(1e-3, s.speed);
    const id = threatId(o);
    lastSeen = { id, type: o.type, kind: o.def.kind, ttc,
                 airborne: !s.onGround, duck: s.duck };
    let d = decisions.get(id);

    if (!d && ttc <= api.config.decideAt && !reachedLimit()) {
      ask(o, ttc);
      d = decisions.get(id);
    }
    if (!d || d.status === 'pending') { rj.setDuck(false); return; }

    if (d.action === 'jump') {
      rj.setDuck(false);
      if (ttc <= api.config.jumpAt && s.onGround) rj.jump();
    } else if (d.action === 'duck') {
      rj.setDuck(ttc <= api.config.duckFrom && ttc > api.config.duckUntil);
    } else {
      rj.setDuck(false);
    }
  }

  function reachedLimit() {
    return api.config.maxDecisions > 0 && api.stats.decisions >= api.config.maxDecisions;
  }

  function start(cfg) {
    Object.assign(api.config, cfg || {});
    rj = window.__rj;
    if (!rj) throw new Error('window.__rj not found -- is the game loaded?');
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
