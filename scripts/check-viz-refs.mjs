// Verifies that every <Viz id="…"> used in MDX is registered in src/data/viz/*.ts,
// that every registered component file exists, and that ids are unique.
import { readFileSync, readdirSync, existsSync } from 'node:fs';
import { join } from 'node:path';

const root = new URL('..', import.meta.url).pathname;
const vizDir = join(root, 'src/data/viz');
const compDir = join(root, 'src/components/viz');

const registered = new Map();
let problems = 0;
for (const f of readdirSync(vizDir).filter((n) => n.endsWith('.ts') && !['index.ts', 'types.ts'].includes(n))) {
  const src = readFileSync(join(vizDir, f), 'utf8');
  const entries = [...src.matchAll(/id:\s*'([^']+)'[\s\S]*?component:\s*'([^']+)'/g)];
  for (const [, id, component] of entries) {
    if (registered.has(id)) {
      console.error(`✗ duplicate viz id "${id}" in ${f} and ${registered.get(id).file}`);
      problems++;
    }
    registered.set(id, { file: f, component });
    if (!existsSync(join(compDir, `${component}.tsx`))) {
      console.error(`✗ ${f}: component "${component}.tsx" for "${id}" does not exist`);
      problems++;
    }
  }
}

const used = new Map();
for (const dir of ['book', 'practice', 'review']) {
  const base = join(root, 'src/content', dir);
  for (const f of readdirSync(base).filter((n) => n.endsWith('.mdx'))) {
    const src = readFileSync(join(base, f), 'utf8');
    for (const [, id] of src.matchAll(/<Viz\s+id="([^"]+)"/g)) {
      used.set(id, [...(used.get(id) ?? []), `${dir}/${f}`]);
      if (!registered.has(id)) {
        console.error(`✗ ${dir}/${f}: <Viz id="${id}"> is not registered`);
        problems++;
      }
    }
    if (/<Viz\s/.test(src) && !src.includes("import Viz from '@/components/viz/Viz.tsx'")) {
      console.error(`✗ ${dir}/${f}: uses <Viz> but does not import it`);
      problems++;
    }
  }
}

const unused = [...registered.keys()].filter((id) => !used.has(id));
console.log(`viz registered: ${registered.size}, referenced: ${used.size}${unused.length ? `, not placed in any page: ${unused.join(', ')}` : ''}`);
if (problems) {
  console.error(`${problems} problem(s) found`);
  process.exit(1);
}
console.log('✓ all viz references resolve');
