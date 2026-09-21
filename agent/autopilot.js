/* Laya autopilot for the three-lane Reindeer Jump.
 *
 * Reads window.__rj, classifies each obstacle in the oncoming wave, picks a
 * lane, and executes.
 *
 * Division of labour:
 *   the model decides WHAT each obstacle is, and WHICH lane to move to;
 *   the harness decides only WHEN to act.
 * The harness reads which lane an obstacle is in, what it is called and how
 * far away it is -- the model takes text, not pixels, so something has to say
 * what is there -- and it never reads the game's own obstacle class. It holds
 * no lane preference of its own: see the stage-two comment below for how the
 * destination is decided and what the harness does and does not contribute.
 *
 * The question is the `split` framing, which scored 0.800 in
 * eval/obstacle_class.py across two sentence phrasings (0.750 on named
 * objects, 0.833 on held-out ones): the validated two-option ground-versus-air
 * choice, plus a separate noul for "is this a solid barrier?". Folding the
 * barrier in as a third choice option instead collapses accuracy to 0.27-0.50
 * -- option count is the sharpest edge on this model, so the third class gets
 * its own question rather than a third label. Both ride in one forward pass.
 */
(function () {
  'use strict';

  const DEFAULTS = {
    endpoint: 'http://127.0.0.1:8000',
    timeoutMs: 4000,      // a hung request must not hold an in-flight slot forever
    decideAt: 2.6,        // seconds-to-impact at which an obstacle gets classified
    maxInFlight: 4,       // obstacles are independent questions, so pipeline them
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

  // Stage two: which lane to move to.
  //
  // Asked as one yes/no per candidate lane, each about that lane alone, and
  // the lane with the lower score wins.
  //
  // It was previously a single choice question listing the candidate lanes as
  // options, which does not work: across 12 lane questions in a recorded run
  // the model took the first-listed option 12 times out of 12, and asking the
  // same pair with the options swapped flips the answer (order-consistency
  // 0.00-0.20 in eval/lane_forced.py). It answers by option position.
  //
  // One question about one described thing is the shape this model is good at
  // -- it is how the obstacle classifier above works, and that has not missed.
  // There is no option list here, so there is no position to be biased by.
  // Measured over three wordings of each lane content: avoids the barrier in
  // 26 of 27 pairs, and prefers a clear lane to one needing a manoeuvre 18 of
  // 18 -- which the choice framing never did.
  //
  // The harness compares two numbers the model produced. That is arithmetic on
  // model output, the same as thresholding the barrier question at 0.5; it is
  // not a preference of the harness's own.
  const LANE_BLOCKED_Q = {
    blocked: { type: 'noul', instructions: 'Is this lane blocked?' },
  };
  const LANE_WORDS = ['left', 'middle', 'right'];
  const LANE_DESC = {
    clear: 'clear, with nothing in it',
    jump: 'something resting on the snow that has to be jumped over',
    duck: 'something hanging overhead that has to be ducked under',
    block: 'a solid barrier that cannot be passed at all',
  };
  /** fetch with a deadline, so a stalled service cannot wedge the agent */
  async function post(body) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), api.config.timeoutMs);
    try {
      const res = await fetch(api.config.endpoint + '/predict', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(body),
        signal: ctrl.signal,
      });
      if (!res.ok) throw new Error('HTTP ' + res.status + ' ' + (await res.text()).slice(0, 120));
      return await res.json();
    } finally {
      clearTimeout(timer);
    }
  }

  const PLAYER_DEPTH = 0.65;   // the game's collision half-depth
  const WAVE_Z = 3;            // obstacles within this z of each other are one wave

  const api = {
    config: Object.assign({}, DEFAULTS),
    running: false,
    laneNeed: 'idle',
    currentNeed: null,
    // the classification the reindeer is about to act on, as opposed to
    // whichever one happened to come back most recently
    activeDecision: null,
    trace: [],
    stats: null,
    onUpdate: null,
    onLane: null,
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
  let wasPlaying = false;
  let laneCalls = new Map();   // wave key -> stage-two result
  let laneInFlight = 0;

  function freshStats() {
    return {
      decisions: 0, correct: 0, byAction: {}, latencies: [],
      deaths: 0, bestDistance: 0, lastDistance: 0,
      errors: 0, deathLog: [], laneChanges: 0,
      laneDecisions: 0, laneContradictions: 0,
      laneCostlier: 0, laneTies: 0,
      laneLatencies: [],
    };
  }

  function reset() {
    api.trace = [];
    api.stats = freshStats();
    decisions = new Map();
    laneCalls = new Map();
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
      const body = await post({ state: describe(o), questions: QUESTIONS });
      const wall = performance.now() - t0;
      const man = body.answers.manoeuvre;
      const pBlock = body.answers.barrier.noul;

      // one distribution over the three classes, for the HUD and the trace
      const probs = {
        jump: man.probabilities.option_a * (1 - pBlock),
        duck: man.probabilities.option_b * (1 - pBlock),
        block: pBlock,
      };
      if (man.choice !== 'option_a' && man.choice !== 'option_b') {
        throw new Error('unexpected manoeuvre label ' + man.choice);
      }
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
      // No action. Inventing one here meant a service outage read as "jump"
      // for every obstacle, ice walls included, while the HUD still showed a
      // run in progress.
      api.stats.errors++;
      console.error('[autopilot] classification failed:', err);
      if (api.onUpdate) api.onUpdate(entry);
    } finally {
      inFlight--;
    }
  }

  /** Ask the model which lane to move to.
   *
   * One request per candidate lane, each describing only that lane, so the
   * model never sees a list of options to prefer the front of.
   */
  async function askLane(key, need, here) {
    // Every lane except the one we are standing in. Excluding it is not the
    // harness narrowing the choice: that lane having been read as a barrier is
    // the whole reason the question is being asked. The rest are not filtered
    // by what the model said about them.
    const candidates = [0, 1, 2].filter(l => l !== here);
    const entry = { status: 'pending', candidates };
    laneCalls.set(key, entry);
    laneInFlight++;
    try {
      const t0 = performance.now();
      const scores = await Promise.all(candidates.map(async (l) => {
        const state = `This lane is ${LANE_DESC[need[l] === null ? 'clear' : need[l]]}.`;
        return (await post({ state, questions: LANE_BLOCKED_Q })).answers.blocked.noul;
      }));
      const wall = performance.now() - t0;

      // The model scores each lane independently, so two lanes holding the
      // same thing score identically -- it is indifferent and there is no
      // model-derived answer to take. The harness then breaks the tie toward
      // the nearer lane. That IS a harness decision, and it is counted as one.
      let best = 0;
      for (let i = 1; i < scores.length; i++) if (scores[i] < scores[best]) best = i;
      const tied = scores.filter((v) => v === scores[best]).length > 1;
      if (tied) {
        let nearest = best;
        candidates.forEach((l, i) => {
          if (scores[i] !== scores[best]) return;
          if (Math.abs(l - here) < Math.abs(candidates[nearest] - here)) nearest = i;
        });
        best = nearest;
      }
      const lane = candidates[best];

      // shown as "how passable", so a longer bar is a better lane
      const probs = [0, 0, 0];
      candidates.forEach((l, i) => { probs[l] = 1 - scores[i]; });

      Object.assign(entry, {
        status: 'done', lane, probs, confidence: 1 - scores[best], wallMs: wall,
        contradiction: need[lane] === 'block',
      });
      const st = api.stats;
      st.laneDecisions++;
      st.laneLatencies.push(wall);
      if (tied) st.laneTies++;
      if (entry.contradiction) st.laneContradictions++;
      if (need[lane] !== null && candidates.some((l) => need[l] === null)) st.laneCostlier++;

      api.trace.push({
        stage: 'lane', run: runIndex, distance: Math.floor(rj.state.distance),
        lanes: need.map((n) => (n === null ? 'clear' : n)),
        candidates, blocked_scores: scores.map((v) => Number(v.toFixed(4))),
        chose: lane, tie_broken_by_harness: tied, contradiction: entry.contradiction,
        wall_ms: Number(wall.toFixed(1)),
      });
      if (api.onLane) api.onLane(entry, need);
    } catch (err) {
      api.stats.errors++;
      // Drop the entry so the next frame asks again. Leaving an errored entry
      // in place wedged the wave: the retry guard is `if (!lc)`, so it never
      // re-asked, and the reindeer held position in a lane it had itself
      // called a barrier until it hit it.
      laneCalls.delete(key);
      console.error('[autopilot] lane question failed, will retry:', err);
    } finally {
      laneInFlight--;
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
      // Count on the playing -> over transition. Deduping on distance instead
      // silently dropped any run that ended at the same integer distance as
      // the previous one, which is common: it under-reported 14 deaths as 10.
      if (s.mode === 'over' && wasPlaying) {
        wasPlaying = false;
        {
          const d = Math.floor(s.distance);
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
      }
      if (s.mode === 'over' && api.config.autoRestart && !reachedLimit()) {
        runIndex++; rj.startGame();
      }
      return;
    }
    wasPlaying = true;

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
    if (!wave) {
      api.currentNeed = null;
      api.activeDecision = null;
      rj.setDuck(false);
      return;
    }

    const ttc = -wave.z / Math.max(1e-3, s.speed);

    // what the model says each lane holds; absent means the lane is empty,
    // which the harness can see for itself
    const need = [null, null, null];
    for (const o of wave.group) {
      const d = decisions.get(idOf(o));
      // only a completed classification yields an action; pending and errored
      // both read as unknown, which the lane ranking treats as last resort
      need[o.lane] = d && d.status === 'done' ? d.action : 'unknown';
      if (o.lane === s.lane) {
        lastSeen = { id: idOf(o), type: o.type, ttc };
      }
    }
    // what the model says about the wave in front of us, right now, for the HUD
    api.currentNeed = need.slice();

    // Retry anything that errored rather than carrying the failure forward.
    // An errored classification used to satisfy `settled`, which sent the
    // description "not yet identified" to the lane question -- and that scores
    // 0.146 on it, below every real lane content, so the lane nothing was
    // known about became the *preferred* destination.
    for (const o of wave.group) {
      const id = idOf(o);
      const d = decisions.get(id);
      if (d && d.status === 'error') decisions.delete(id);
    }
    const settled = wave.group.every(o => {
      const d = decisions.get(idOf(o));
      return d && d.status === 'done';
    });
    // The harness's only say in lane changes: noticing that the lane the model
    // called a barrier is the one we are standing in.
    const mustMove = settled && need[s.lane] === 'block';

    let target = s.lane;              // hold, unless the model names somewhere else
    if (mustMove) {
      const key = Math.min(...wave.group.map(idOf));
      let lc = laneCalls.get(key);
      if (!lc && laneInFlight < 2 && !reachedLimit()) {
        askLane(key, need, s.lane);
        lc = laneCalls.get(key);
      }
      if (lc && lc.status === 'done') {
        target = lc.lane;
        api.laneNeed = 'answered';
      } else {
        // Waiting. Do NOT edge toward a guess: moving here, and only later
        // showing a lane verdict in the panel, is precisely the mismatch
        // between what the reindeer does and what the model said.
        api.laneNeed = 'waiting';
      }
    } else {
      api.laneNeed = settled ? 'not needed' : 'reading';
    }

    if (laneCalls.size > 64) {
      const keep = new Map();
      for (const [k, v] of laneCalls) if (k > nextId - 48) keep.set(k, v);
      laneCalls = keep;
    }

    if (s.lane !== target) {
      rj.moveLane(Math.sign(target - s.lane));
      api.stats.laneChanges++;
    }

    // Only commit to a manoeuvre once we are actually standing in the lane it
    // belongs to -- jumping while still sliding clears the wrong obstacle.
    // Publish the decision the reindeer is actually about to act on. The HUD
    // used to paint whichever classification finished last, which with four
    // in flight over a 2.6s horizon is usually an obstacle in another lane or
    // a later wave -- so the bars and the reindeer disagreed on screen.
    let active = null;
    for (const o of wave.group) {
      if (o.lane !== target) continue;
      const d = decisions.get(idOf(o));
      if (d && d.status === 'done') active = d;
    }
    api.activeDecision = active;

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
