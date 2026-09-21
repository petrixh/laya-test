/* Drive Reindeer Jump with the Laya autopilot.
 *
 *   node agent/play.mjs --decisions 40                 headless, records video
 *   node agent/play.mjs --headed                       watch it locally
 *   node agent/play.mjs --encoding nl --out runs/nl    A/B the state encoding
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
  encoding: String(arg('encoding', 'json')),
  decisions: Number(arg('decisions', 40)),
  seconds: Number(arg('seconds', 180)),
  out: String(arg('out', 'runs/latest')),
  video: arg('video', true) !== 'false',
  width: Number(arg('width', 960)),
  height: Number(arg('height', 640)),
  // --mock stands up a fake decision service that answers correctly with a
  // known probability. It verifies the harness end to end (page, hook, HUD,
  // grading, trace, video) without needing the model loaded, and because its
  // accuracy is known it also proves the grading path measures what it claims.
  mock: arg('mock', false),
  mockAccuracy: Number(arg('mock-accuracy', 0.85)),
  mockLatency: Number(arg('mock-latency', 280)),
};

/** Fake Laya service with a known answer distribution. */
function serveMock(accuracy, latencyMs) {
  const ACTIONS = ['jump', 'duck', 'run'];
  return new Promise((ok) => {
    const server = createServer((req, res) => {
      const cors = {
        'access-control-allow-origin': '*',
        'access-control-allow-headers': '*',
        'access-control-allow-methods': 'GET,POST,OPTIONS',
      };
      if (req.method === 'OPTIONS') { res.writeHead(204, cors).end(); return; }
      if (req.url === '/info') {
        res.writeHead(200, { ...cors, 'content-type': 'application/json' });
        res.end(JSON.stringify({ checkpoint: 'mock', subfolder: 'mock', device: 'mock',
                                 ready: true, torch: 'n/a' }));
        return;
      }
      let raw = '';
      req.on('data', (c) => (raw += c));
      req.on('end', () => {
        const body = JSON.parse(raw || '{}');
        const st = body.state || {};
        const text = typeof st === 'string' ? st : JSON.stringify(st);
        const truth = /hanging in the air/.test(text) ? 'duck' : 'jump';
        const right = Math.random() < accuracy;
        const choice = right ? truth : ACTIONS[(Math.random() * 3) | 0];
        const probabilities = {};
        let rest = 1;
        for (const a of ACTIONS) {
          probabilities[a] = a === choice ? (rest = 0.62 + Math.random() * 0.34, rest) : 0;
        }
        const spare = 1 - rest;
        for (const a of ACTIONS) if (a !== choice) probabilities[a] = spare / 2;
        // jitter around the measured CPU latency so the sparkline looks real
        const wait = latencyMs * (0.75 + Math.random() * 0.6);
        setTimeout(() => {
          res.writeHead(200, { ...cors, 'content-type': 'application/json' });
          res.end(JSON.stringify({
            model: 'mock', answers: { action: {
              type: 'choice', choice, probabilities,
              confidence: probabilities[choice],
            } }, usage: { input_tokens: 0 }, latency_ms: wait,
          }));
        }, wait);
      });
    });
    server.listen(0, '127.0.0.1', () => ok({ server, port: server.address().port }));
  });
}

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

  let mockServer = null;
  if (opts.mock) {
    const m = await serveMock(opts.mockAccuracy, opts.mockLatency);
    mockServer = m.server;
    opts.endpoint = `http://127.0.0.1:${m.port}`;
    console.log(`mock decision service on :${m.port} ` +
                `(accuracy ${opts.mockAccuracy}, latency ~${opts.mockLatency}ms)`);
  }

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
    window.__autopilot.start({
      endpoint: o.endpoint, encoding: o.encoding, maxDecisions: o.decisions,
    });
    window.__layaHud.mount(window.__autopilot, o.endpoint);
  }, opts);
  console.log(`autopilot started (encoding=${opts.encoding}, target ${opts.decisions} decisions)`);

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
    if (n >= opts.decisions) break;
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
  if (mockServer) mockServer.close();

  if (opts.video) {
    for (const f of await readdir(outDir)) {
      if (f.endsWith('.webm')) { await rename(join(outDir, f), join(outDir, 'run.webm')); break; }
    }
  }

  console.log('\n' + JSON.stringify(summary, null, 2));
  console.log(`\nartifacts in ${outDir}`);
  process.exit(summary.errors > 0 && summary.decisions === 0 ? 1 : 0);
};

main().catch((e) => { console.error(e); process.exit(1); });
