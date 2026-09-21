/* Verify the in-page autopilot loader: the path a human uses, where the agent
 * is started by ?autopilot=1 rather than injected by the driver.
 *
 *   node agent/check-page.mjs [--endpoint http://host:8000]
 */
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join, resolve } from 'node:path';
import { chromium } from 'playwright';

const ROOT = resolve(import.meta.dirname, '..');
const i = process.argv.indexOf('--endpoint');
const ENDPOINT = i > -1 ? process.argv[i + 1] : 'http://127.0.0.1:8000';
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json' };

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
const port = server.address().port;

const browser = await chromium.launch({
  headless: true,
  args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--disable-dev-shm-usage'],
});
const page = await browser.newPage({ viewport: { width: 960, height: 640 } });
page.on('pageerror', (e) => console.error('  [page throw]', String(e).slice(0, 200)));

const url = `http://127.0.0.1:${port}/index.html?autopilot=1&laya=${encodeURIComponent(ENDPOINT)}&decisions=3`;
console.log('opening', url.replace(`127.0.0.1:${port}`, '<served>'));
await page.goto(url, { waitUntil: 'load' });

let ok = true;
const fail = (m) => { console.error('  FAIL ' + m); ok = false; };

try {
  await page.waitForFunction(() => !!window.__autopilot, null, { timeout: 30000 });
  console.log('  ok   autopilot loaded from the query parameter alone');
  const ep = await page.evaluate(() => window.__autopilot.config.endpoint);
  ep === ENDPOINT ? console.log(`  ok   endpoint taken from ?laya= (${ep})`)
                  : fail(`endpoint was ${ep}, expected ${ENDPOINT}`);
  await page.waitForFunction(() => !!document.getElementById('laya-hud'), null, { timeout: 10000 });
  console.log('  ok   HUD mounted');
  await page.waitForFunction(() => window.__autopilot.stats.decisions > 0, null, { timeout: 120000 });
  const s = await page.evaluate(() => ({ ...window.__autopilot.stats, trace: window.__autopilot.trace.length }));
  console.log(`  ok   live decisions against the service: ${s.decisions} (errors ${s.errors})`);
  if (s.errors) fail(`${s.errors} request errors -- check CORS and that the service is reachable`);
  const dev = await page.evaluate(() => document.getElementById('lh-dev').textContent);
  dev.includes('no service') ? fail('HUD could not reach /info') : console.log(`  ok   HUD shows backend: ${dev}`);
} catch (e) {
  fail(e.message.slice(0, 200));
}

await browser.close();
server.close();
console.log(ok ? '\nPASS' : '\nFAIL');
process.exit(ok ? 0 : 1);
