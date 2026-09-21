/* Drive Reindeer Jump with the Laya autopilot.
 *
 *   node agent/play.mjs --decisions 40    headless, records video
 *   node agent/play.mjs --headed          watch it locally
 *
 * Needs a running Laya service. To exercise the game and the harness without
 * one, use agent/check-solvable.mjs, which drives the game with a rule-based
 * player and no model at all.
 *
 * The page is served over HTTP rather than opened as file:// so the in-page
 * fetch to the Laya service has a real origin for CORS.
 */
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { mkdir, writeFile, rename, readdir } from 'node:fs/promises';
import { extname, join, resolve } from 'node:path';
import { chromium } from 'playwright';

const ROOT = resolve(import.meta.dirname, '..');

function arg(name, fallback) {
  const i = process.argv.indexOf('--' + name);
  if (i === -1) return fallback;
  const next = process.argv[i + 1];
  return next && !next.startsWith('--') ? next : true;
}

const opts = {
  headed: !!arg('headed', false),
  endpoint: String(arg('endpoint', 'http://127.0.0.1:8000')),

  decisions: Number(arg('decisions', 40)),
  seconds: Number(arg('seconds', 180)),
  out: String(arg('out', 'runs/latest')),
  video: arg('video', true) !== 'false',
  width: Number(arg('width', 960)),
  height: Number(arg('height', 640)),
};

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
};

function serve() {
  return new Promise((ok) => {
    const server = createServer(async (req, res) => {
      try {
        const path = decodeURIComponent(new URL(req.url, 'http://x').pathname);
        const file = join(ROOT, path === '/' ? 'index.html' : path);
        if (!file.startsWith(ROOT)) { res.writeHead(403).end(); return; }
        const body = await readFile(file);
        res.writeHead(200, { 'content-type': MIME[extname(file)] || 'application/octet-stream' });
        res.end(body);
      } catch {
        res.writeHead(404).end('not found');
      }
    });
    server.listen(0, '127.0.0.1', () => ok({ server, port: server.address().port }));
  });
}

function pct(sorted, q) {
  if (!sorted.length) return null;
  return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
}

const main = async () => {
  const { server, port } = await serve();
  const outDir = resolve(ROOT, opts.out);
  await mkdir(outDir, { recursive: true });

  const browser = await chromium.launch({
    headless: !opts.headed,
    args: [
      // headless chromium has no GPU; SwiftShader gives it WebGL2 in software
      '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
      '--disable-dev-shm-usage',
    ],
  });
  const context = await browser.newContext({
    viewport: { width: opts.width, height: opts.height },
    ...(opts.video ? { recordVideo: { dir: outDir, size: { width: opts.width, height: opts.height } } } : {}),
  });
  const page = await context.newPage();

  page.on('console', (m) => {
    if (m.type() === 'error') console.error('  [page error]', m.text().slice(0, 300));
  });
  page.on('pageerror', (e) => console.error('  [page throw]', String(e).slice(0, 300)));

  console.log(`serving ${ROOT} on :${port}`);
  await page.goto(`http://127.0.0.1:${port}/index.html`, { waitUntil: 'load' });

  const fatal = await page.locator('#fatal').isVisible().catch(() => false);
  if (fatal) {
    console.error('game refused to start:', (await page.locator('#fatal').innerText()).slice(0, 300));
    await browser.close(); server.close(); process.exit(1);
  }

  await page.waitForFunction(() => !!window.__rj, null, { timeout: 30000 });
  console.log('game ready');

  await page.addScriptTag({ path: join(ROOT, 'agent/autopilot.js') });
  await page.addScriptTag({ path: join(ROOT, 'agent/hud.js') });
  await page.evaluate((o) => {
    window.__autopilot.start({ endpoint: o.endpoint, maxDecisions: o.decisions });
    window.__layaHud.mount(window.__autopilot, o.endpoint);
  }, opts);
  console.log(`autopilot started (target ${opts.decisions} decisions)`);

  const deadline = Date.now() + opts.seconds * 1000;
  let last = -1;
  while (Date.now() < deadline) {
    const n = await page.evaluate(() => window.__autopilot.stats.decisions);
    if (n !== last) {
      const s = await page.evaluate(() => {
        const st = window.__autopilot.stats;
        return { n: st.decisions, ok: st.correct, deaths: st.deaths, err: st.errors,
                 dist: Math.floor(window.__rj.state.distance) };
      });
      process.stdout.write(`\r  decisions ${s.n}/${opts.decisions}  acc ` +
        `${s.n ? (100 * s.ok / s.n).toFixed(1) : '--'}%  crashes ${s.deaths}  ` +
        `errors ${s.err}  dist ${s.dist}m   `);
      last = n;
    }
    if (opts.decisions > 0 && n >= opts.decisions) break;
    await page.waitForTimeout(200);
  }
  process.stdout.write('\n');

  const result = await page.evaluate(() => {
    const st = window.__autopilot.stats;
    window.__autopilot.stop();
    return { stats: { ...st }, trace: window.__autopilot.trace };
  });

  const lat = result.stats.latencies.slice().sort((a, b) => a - b);
  const byTruth = {};
  for (const t of result.trace) {
    if (t.stage === 'lane') continue;      // stage-two rows carry no obstacle truth
    const k = t.truth;
    byTruth[k] = byTruth[k] || { n: 0, correct: 0 };
    byTruth[k].n++; if (t.correct) byTruth[k].correct++;
  }

  const summary = {
    generated: new Date().toISOString(),
    options: opts,
    decisions: result.stats.decisions,
    correct: result.stats.correct,
    accuracy: result.stats.decisions ? +(result.stats.correct / result.stats.decisions).toFixed(4) : null,
    crashes: result.stats.deaths,
    best_distance_m: result.stats.bestDistance,
    errors: result.stats.errors,
    lane_changes: result.stats.laneChanges,
    lane_choice: {
      decisions: result.stats.laneDecisions,
      chose_a_barrier: result.stats.laneContradictions,
      ties_broken_by_harness: result.stats.laneTies,
      passed_up_a_clear_lane: result.stats.laneDecisions
        ? +(result.stats.laneCostlier / result.stats.laneDecisions).toFixed(4) : null,
      latency_p50_ms: result.stats.laneLatencies.length
        ? +pct(result.stats.laneLatencies.slice().sort((a, b) => a - b), 0.5).toFixed(1) : null,
    },
    latency_ms: {
      p50: lat.length ? +pct(lat, 0.5).toFixed(1) : null,
      p95: lat.length ? +pct(lat, 0.95).toFixed(1) : null,
      max: lat.length ? +lat[lat.length - 1].toFixed(1) : null,
    },
    per_truth: Object.fromEntries(Object.entries(byTruth).map(
      ([k, v]) => [k, { n: v.n, accuracy: +(v.correct / v.n).toFixed(4) }])),
    // a death whose verdict was correct indicts the execution window, not the model
    deaths_with_correct_verdict: result.stats.deathLog.filter((d) => d.correct === true).length,
    deaths_by_verdict: result.stats.deathLog.reduce((a, d) => {
      const k = d.correct === true ? 'correct'
        : d.verdict === 'error' ? 'service-error'
        : d.verdict === 'pending' ? 'answer-too-late'
        : d.verdict === 'none' ? 'never-asked' : 'wrong';
      a[k] = (a[k] || 0) + 1; return a;
    }, {}),
  };

  await writeFile(join(outDir, 'summary.json'), JSON.stringify(summary, null, 2));
  await writeFile(join(outDir, 'trace.json'), JSON.stringify(result.trace, null, 2));
  await writeFile(join(outDir, 'deaths.json'), JSON.stringify(result.stats.deathLog, null, 2));

  await context.close();          // flushes the video file
  await browser.close();
  server.close();

  if (opts.video) {
    for (const f of await readdir(outDir)) {
      if (f.endsWith('.webm')) { await rename(join(outDir, f), join(outDir, 'run.webm')); break; }
    }
  }

  console.log('\n' + JSON.stringify(summary, null, 2));
  console.log(`\nartifacts in ${outDir}`);
  process.exit(summary.decisions === 0 ? 1 : 0);
};

main().catch((e) => { console.error(e); process.exit(1); });
