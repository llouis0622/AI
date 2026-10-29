import { MDX_NODE_TYPES, el, filePath, hasClass, text } from './hast-walk.mjs';

const LANG_NAMES = {
  python: 'Python',
  py: 'Python',
  bash: '터미널',
  sh: '터미널',
  shell: '터미널',
  json: 'JSON',
  yaml: 'YAML',
  js: 'JavaScript',
  ts: 'TypeScript',
  text: '텍스트',
  plaintext: '텍스트'
};

/**
 * Beginners should meet pictures and words first. Display math (everywhere) and
 * code blocks (book chapters only — code labs keep their code open) are wrapped
 * in a <details> toggle. Inline math stays inline.
 */
export default function rehypeFold() {
  return (tree, file) => {
    const foldCode = filePath(file).includes('/content/book/');
    transform(tree, foldCode);
  };
}

function transform(parent, foldCode) {
  if (!parent.children) return;
  for (let i = 0; i < parent.children.length; i++) {
    const node = parent.children[i];
    if (MDX_NODE_TYPES.has(node.type)) continue;
    if (node.type !== 'element') continue;
    if (node.tagName === 'details') continue;
    if (isDisplayMath(node)) {
      parent.children[i] = fold('math', node);
      continue;
    }
    if (foldCode && node.tagName === 'pre') {
      parent.children[i] = fold('code', node);
      continue;
    }
    transform(node, foldCode);
  }
}

function isDisplayMath(node) {
  return hasClass(node, 'katex-display') || hasClass(node, 'math-display');
}

function fold(kind, node) {
  const isMath = kind === 'math';
  const lang = String(node.properties?.dataLanguage ?? '').toLowerCase();
  const langName = LANG_NAMES[lang] ?? (lang ? lang.toUpperCase() : '');
  const label = isMath ? '수식으로 보기' : `코드로 보기${langName ? ` · ${langName}` : ''}`;
  return el('details', { className: ['fold', `fold-${kind}`] }, [
    el('summary', { className: ['fold-summary'] }, [
      el('span', { className: ['fold-icon'], ariaHidden: 'true' }, [text(isMath ? 'Σ' : '</>')]),
      el('span', { className: ['fold-label'] }, [text(label)]),
      el('span', { className: ['fold-hint'] }, [text('눌러서 펼치기')])
    ]),
    el('div', { className: ['fold-body'] }, [node])
  ]);
}
