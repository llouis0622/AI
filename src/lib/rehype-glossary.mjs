import { MDX_NODE_TYPES, el, filePath, hasClass, text } from './hast-walk.mjs';

const SKIP_TAGS = new Set([
  'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
  'a', 'code', 'pre', 'kbd', 'summary', 'script', 'style', 'button', 'svg', 'math'
]);
const WORD_CHAR = /[0-9A-Za-z가-힣]/;
const LATIN_CHAR = /[0-9A-Za-z]/;

/**
 * Wraps the first occurrence of every glossary term on a page in
 * `<span class="term">` carrying its one-line definition. A small client script
 * (GlossaryTooltip) turns these into hover / tap tooltips.
 *
 * Korean particles attach to the end of nouns ("파라미터를"), so only the left
 * edge of a Hangul term needs a word boundary.
 */
export default function rehypeGlossary({ terms = [], base = '' } = {}) {
  const entries = [];
  for (const term of terms) {
    for (const alias of new Set([term.term, ...(term.aliases ?? [])])) {
      if (alias) entries.push({ alias, term });
    }
  }
  if (entries.length === 0) return () => {};
  entries.sort((a, b) => b.alias.length - a.alias.length);
  const byAlias = new Map(entries.map((e) => [e.alias, e.term]));
  const pattern = new RegExp(entries.map((e) => escapeRegExp(e.alias)).join('|'), 'g');
  const prefix = base.replace(/\/$/, '');

  return (tree, file) => {
    if (!/\/content\/(book|practice|review)\//.test(filePath(file))) return;
    const used = new Set();
    visit(tree, (node) => {
      const value = node.value;
      const out = [];
      let last = 0;
      pattern.lastIndex = 0;
      let match;
      while ((match = pattern.exec(value))) {
        const alias = match[0];
        const term = byAlias.get(alias);
        const start = match.index;
        const end = start + alias.length;
        const before = value[start - 1] ?? '';
        const after = value[end] ?? '';
        const latinTerm = LATIN_CHAR.test(alias[alias.length - 1]);
        const boundaryOk = !WORD_CHAR.test(before) && !(latinTerm && LATIN_CHAR.test(after));
        if (!term || used.has(term.id) || !boundaryOk) continue;
        used.add(term.id);
        if (start > last) out.push(text(value.slice(last, start)));
        out.push(
          el(
            'span',
            {
              className: ['term'],
              tabIndex: 0,
              role: 'button',
              dataTerm: term.id,
              dataTitle: term.term,
              dataDef: term.short,
              dataEmoji: term.emoji ?? '',
              dataHref: `${prefix}/glossary/#${term.id}`
            },
            [text(alias)]
          )
        );
        last = end;
      }
      if (out.length === 0) return null;
      if (last < value.length) out.push(text(value.slice(last)));
      return out;
    });
  };
}

/** Visit text nodes outside skipped regions; a visitor may return replacement nodes. */
function visit(parent, fn) {
  if (!parent.children) return;
  for (let i = 0; i < parent.children.length; i++) {
    const node = parent.children[i];
    if (MDX_NODE_TYPES.has(node.type)) continue;
    if (node.type === 'text') {
      const replacement = fn(node);
      if (replacement) {
        parent.children.splice(i, 1, ...replacement);
        i += replacement.length - 1;
      }
      continue;
    }
    if (node.type !== 'element') continue;
    if (SKIP_TAGS.has(node.tagName)) continue;
    if (hasClass(node, 'katex') || hasClass(node, 'term') || hasClass(node, 'fold-summary')) continue;
    visit(node, fn);
  }
}

function escapeRegExp(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
