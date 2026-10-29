// @ts-check
import { defineConfig } from 'astro/config';
import react from '@astrojs/react';
import mdx from '@astrojs/mdx';
import { unified } from '@astrojs/markdown-remark';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeBaseLinks from './src/lib/rehype-base-links.mjs';
import rehypeFold from './src/lib/rehype-fold.mjs';
import rehypeGlossary from './src/lib/rehype-glossary.mjs';
import { glossaryTerms } from './src/data/glossary/index.ts';

const BASE = '/AI';

export default defineConfig({
  site: 'https://llouis0622.github.io',
  base: BASE,
  trailingSlash: 'always',
  output: 'static',
  // Overridable so parallel worktrees sharing node_modules keep separate caches.
  cacheDir: process.env.ASTRO_CACHE_DIR ?? './node_modules/.astro',
  build: { format: 'directory' },
  integrations: [react(), mdx()],
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [
        [rehypeKatex, { strict: 'ignore', output: 'html' }],
        [rehypeBaseLinks, { base: BASE }],
        rehypeFold,
        [rehypeGlossary, { terms: glossaryTerms, base: BASE }]
      ]
    }),
    shikiConfig: {
      themes: { light: 'github-light', dark: 'tokyo-night' },
      wrap: false
    }
  },
  vite: {
    cacheDir: process.env.ASTRO_VITE_CACHE ?? 'node_modules/.vite',
    build: { chunkSizeWarningLimit: 1600 }
  }
});
