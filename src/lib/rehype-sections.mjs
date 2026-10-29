import { MDX_NODE_TYPES, el, filePath, text } from './hast-walk.mjs';

/** Recurring end-of-chapter sections that get their own card treatment. */
const KINDS = [
  { match: /^핵심 요약/, kind: 'summary', icon: '📌' },
  { match: /^스스로 점검/, kind: 'selfcheck', icon: '✍️' },
  { match: /^다음 장에서/, kind: 'next', icon: '🔭' }
];

const textOf = (node) =>
  node.type === 'text' ? node.value : (node.children ?? []).map(textOf).join('');

/**
 * Wraps "핵심 요약", "스스로 점검", "다음 장에서" (an h2 and everything up to the
 * next h2) in <section class="sec sec-…"> so they can be styled as cards. In the
 * self-check section each list item becomes a tickable question.
 */
export default function rehypeSections() {
  return (tree, file) => {
    if (!filePath(file).includes('/content/book/')) return;
    const kids = tree.children;
    const out = [];
    let i = 0;
    while (i < kids.length) {
      const node = kids[i];
      const spec = node.type === 'element' && node.tagName === 'h2' && KINDS.find((k) => k.match.test(textOf(node).trim()));
      if (!spec) {
        out.push(node);
        i++;
        continue;
      }
      const body = [];
      let j = i + 1;
      while (j < kids.length && !(kids[j].type === 'element' && kids[j].tagName === 'h2')) {
        body.push(kids[j]);
        j++;
      }
      if (spec.kind === 'selfcheck') markQuestions(body);
      const head = el('div', { className: ['sec-head'] }, [el('span', { className: ['sec-icon'], ariaHidden: 'true' }, [text(spec.icon)]), node]);
      out.push(el('section', { className: ['sec', `sec-${spec.kind}`], dataSec: spec.kind }, [head, ...body]));
      i = j;
    }
    tree.children = out;
  };
}

function markQuestions(nodes) {
  let n = 0;
  for (const node of nodes) {
    if (MDX_NODE_TYPES.has(node.type) || node.type !== 'element' || (node.tagName !== 'ol' && node.tagName !== 'ul')) continue;
    for (const li of node.children ?? []) {
      if (li.type !== 'element' || li.tagName !== 'li') continue;
      n += 1;
      li.properties = { ...li.properties, dataCheck: String(n) };
      li.children = [
        el('button', { type: 'button', className: ['check-box'], ariaLabel: `${n}번 질문을 생각해 봤어요`, ariaPressed: 'false' }, [text(String(n))]),
        ...li.children
      ];
    }
  }
}
