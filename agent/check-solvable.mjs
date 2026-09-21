/* Solvability check for the three-lane game.
 *
 * Injects a perfect rule-based player -- it reads the wave directly from the
 * game state, so it never misclassifies anything -- and asserts it survives.
 * Any death here is a generator bug: a wave with no passable, reachable lane,
 * or one that cannot be executed in the time available.
 *
 *   node agent/check-solvable.mjs [--waves 200]
 */
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join, resolve } from 'node:path';
import { chromium } from 'playwright';

const ROOT = resolve(import.meta.dirname, '..');
const argOf = (n, d) => { const i = process.argv.indexOf('--' + n); return i > -1 ? Number(process.argv[i + 1]) : d; };
const WAVES = argOf('waves', 200);

const MIME = { '.html': 'text/html', '.js': 'text/javascript' };
const server = createServer(async (req, res) => {
  try {
    const path = decodeURIComponent(new URL(req.url, 'http://x').pathname);
    const file = join(ROOT, path === '/' ? 'index.html' : path);
    if (!file.startsWith(ROOT)) return res.writeHead(403).end();
    res.writeHead(200, { 'content-type': MIME[extname(file)] || 'application/octet-stream' });
    res.end(await readFile(file));
  } catch { res.writeHead(404).end(); }
});
await new Promise((ok) => server.listen(0, '127.0.0.1', ok));

const browser = await chromium.launch({
  headless: true,
  args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--disable-dev-shm-usage'],
});
const page = await browser.newPage({ viewport: { width: 640, height: 400 } });
page.on('pageerror', (e) => console.error('  [page throw]', String(e).slice(0, 200)));
await page.goto(`http://127.0.0.1:${server.address().port}/index.html`, { waitUntil: 'load' });
await page.waitForFunction(() => !!window.__rj, null, { timeout: 30000 });

const result = await page.evaluate(async (waves) => {
  const rj = window.__rj;
  const log = { deaths: [], waves: 0, maxWave: 0, distance: 0,
                openLanes: {}, clearLanes: {}, walls: {}, kinds: {} };

  function currentWave() {
    let z = null;
    for (const o of rj.obstacles) {
      const oz = o.mesh.position.z;
      if (oz > o.def.zHalf + 0.65) continue;   // zHalf + PLAYER_DEPTH, as the game's collider uses
      if (z === null || oz > z) z = oz;
    }
    if (z === null) return null;
    const group = rj.obstacles.filter((o) => Math.abs(o.mesh.position.z - z) < 3);
    const need = [null, null, null];
    for (const o of group) need[o.lane] = o.def.kind;
    return { z, need };
  }

  function chooseLane(need, here) {
    const open = [];
    for (let l = 0; l < 3; l++) if (need[l] !== 'block') open.push(l);
    if (!open.length) return null;                 // generator bug
    const clear = open.filter((l) => need[l] === null);
    const pool = clear.length ? clear : open;      // an empty lane beats a manoeuvre
    return pool.reduce((a, b) => (Math.abs(b - here) < Math.abs(a - here) ? b : a));
  }

  rj.startGame();
  let lastWave = -1;
  await new Promise((done) => {
    const tick = () => {
      const s = rj.state;
      if (s.mode !== 'playing') {
        log.deaths.push({ wave: s.wave, distance: Math.floor(s.distance),
                          lane: s.lane, x: +s.x.toFixed(2) });
        if (log.deaths.length > 8 || s.wave >= waves) { done(); return; }
        rj.startGame(); lastWave = -1;
        requestAnimationFrame(tick); return;
      }
      log.maxWave = Math.max(log.maxWave, s.wave);
      log.distance = Math.max(log.distance, Math.floor(s.distance));
      if (s.wave >= waves) { done(); return; }

      const w = currentWave();
      if (!w) { rj.setDuck(false); requestAnimationFrame(tick); return; }

      const target = chooseLane(w.need, s.lane);
      if (target === null) {
        log.deaths.push({ wave: s.wave, reason: 'every lane walled', need: w.need });
        done(); return;
      }
      if (s.wave !== lastWave) {
        lastWave = s.wave;
        log.waves++;
        const open = w.need.filter((n) => n !== 'block').length;
        const clear = w.need.filter((n) => n === null).length;
        const walls = w.need.filter((n) => n === 'block').length;
        log.openLanes[open] = (log.openLanes[open] || 0) + 1;
        log.clearLanes[clear] = (log.clearLanes[clear] || 0) + 1;
        log.walls[walls] = (log.walls[walls] || 0) + 1;
        for (const n of w.need) log.kinds[n || 'empty'] = (log.kinds[n || 'empty'] || 0) + 1;
      }

      if (s.lane !== target) rj.moveLane(Math.sign(target - s.lane));

      const ttc = -w.z / Math.max(1e-3, s.speed);
      const inLane = Math.abs(s.x - rj.LANES[target]) < 0.1;
      const req = w.need[target];
      if (req === 'ground') {
        rj.setDuck(false);
        if (inLane && ttc <= 0.30 && s.onGround) rj.jump();
      } else if (req === 'air') {
        rj.setDuck(ttc <= 0.40 && ttc > -0.12);
      } else {
        rj.setDuck(false);
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  });
  return log;
}, WAVES);

await browser.close();
server.close();

console.log(`waves survived : ${result.maxWave}`);
console.log(`distance       : ${result.distance}m`);
console.log(`deaths         : ${result.deaths.length}`);
const pct = (o, total = result.waves) => Object.entries(o).sort().map(([k, v]) =>
  `${k}:${(100 * v / total).toFixed(0)}%`).join('  ');
console.log(`walls per wave : ${pct(result.walls)}`);
console.log(`passable lanes : ${pct(result.openLanes)}`);
console.log(`wholly clear   : ${pct(result.clearLanes)}`);
console.log(`lane contents  : ${pct(result.kinds, result.waves * 3)}  (share of all lanes)`);
for (const d of result.deaths.slice(0, 8)) console.log('   ', JSON.stringify(d));
const ok = result.deaths.length === 0 && result.maxWave >= WAVES - 1;
console.log(ok ? '\nPASS - every wave had a passable, reachable lane' : '\nFAIL');
process.exit(ok ? 0 : 1);
