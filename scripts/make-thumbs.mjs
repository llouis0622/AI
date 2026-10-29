// Captures gallery thumbnails of every registered visualization.
//
//   npm run build && npx astro preview --port 4321 &
//   node scripts/make-thumbs.mjs [--port 4321] [--only id1,id2]
//
// Writes public/thumbs/{light,dark}/<id>.jpg (≈480×300). Needs Playwright with Chromium
// available on the machine (it is not a project dependency). The gallery falls back to
// emoji art for any viz without a thumbnail.
import { mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
let chromium;
try {
  ({ chromium } = require('playwright'));
} catch {
  ({ chromium } = require(join(process.env.PLAYWRIGHT_DIR ?? '/opt/node22/lib/node_modules', 'playwright')));
}

const arg = (name, fallback) => {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? process.argv[i + 1] : fallback;
};
const port = arg('port', '4321');
const only = arg('only', '')?.split(',').filter(Boolean);
const root = new URL('..', import.meta.url).pathname;

const ids = [];
const vizDir = join(root, 'src/data/viz');
for (const f of readdirSync(vizDir).filter((n) => n.endsWith('.ts') && !['index.ts', 'types.ts'].includes(n))) {
  const src = readFileSync(join(vizDir, f), 'utf8');
  for (const block of src.split(/\n\s*\{\s*\n/).slice(1)) {
    const id = block.match(/id:\s*'([^']+)'/)?.[1];
    if (id) ids.push({ id, is3D: /is3D:\s*true/.test(block) });
  }
}
const targets = only?.length ? ids.filter((v) => only.includes(v.id)) : ids;
console.log(`capturing ${targets.length} visualizations`);

const browser = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
for (const theme of ['light', 'dark']) {
  const outDir = join(root, 'public/thumbs', theme);
  mkdirSync(outDir, { recursive: true });
  const ctx = await browser.newContext({ viewport: { width: 1100, height: 900 }, deviceScaleFactor: 1, colorScheme: theme, reducedMotion: 'no-preference' });
  for (const { id, is3D } of targets) {
    const page = await ctx.newPage();
    try {
      await page.goto(`http://localhost:${port}/AI/playground/${id}/`, { waitUntil: 'networkidle' });
      await page.waitForTimeout(1200);
      if (is3D) {
        await page.locator('.vz-play').first().click({ timeout: 8000 });
        await page.waitForTimeout(3500);
      }
      // Nudge simulations that start with a play button so the thumbnail shows motion.
      const play = page.locator('.vz-body .vz-playbar .vz-btn-primary').first();
      if (await play.count()) {
        await play.click({ timeout: 3000 }).catch(() => {});
        await page.waitForTimeout(2500);
      }
      const body = page.locator('.vz-body').first();
      const box = await body.boundingBox();
      const clip = { x: box.x, y: box.y, width: box.width, height: Math.min(box.height, box.width * 0.62) };
      const buf = await page.screenshot({ clip, type: 'jpeg', quality: 80 });
      // downscale via a canvas in the page to keep files small
      const small = await page.evaluate(async (b64) => {
        const img = new Image();
        img.src = `data:image/jpeg;base64,${b64}`;
        await img.decode();
        const w = 480;
        const h = Math.round((img.height / img.width) * w);
        const c = document.createElement('canvas');
        c.width = w;
        c.height = h;
        c.getContext('2d').drawImage(img, 0, 0, w, h);
        return c.toDataURL('image/jpeg', 0.8).split(',')[1];
      }, buf.toString('base64'));
      writeFileSync(join(outDir, `${id}.jpg`), Buffer.from(small, 'base64'));
      console.log(`✓ ${theme}/${id}`);
    } catch (e) {
      console.log(`✗ ${theme}/${id}: ${e.message.split('\n')[0]}`);
    }
    await page.close();
  }
  await ctx.close();
}
await browser.close();
