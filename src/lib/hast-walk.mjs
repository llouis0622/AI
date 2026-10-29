/**
 * Tiny dependency-free hast helpers shared by the rehype plugins.
 */

/** Node types produced by MDX that must never be rewritten (JSX, ESM, expressions). */
export const MDX_NODE_TYPES = new Set([
  'mdxJsxFlowElement',
  'mdxJsxTextElement',
  'mdxjsEsm',
  'mdxFlowExpression',
  'mdxTextExpression'
]);

/** Depth-first walk. Return `false` from `visit` to skip a node's children. */
export function walk(node, visit, parent = null) {
  if (visit(node, parent) === false) return;
  if (!node.children) return;
  for (const child of [...node.children]) walk(child, visit, node);
}

export function hasClass(node, name) {
  const cls = node?.properties?.className;
  if (Array.isArray(cls)) return cls.includes(name);
  if (typeof cls === 'string') return cls.split(/\s+/).includes(name);
  return false;
}

/** Absolute file path of the vfile being processed, with forward slashes. */
export function filePath(file) {
  return String(file?.path ?? file?.history?.[0] ?? '').replace(/\\/g, '/');
}

export function el(tagName, properties = {}, children = []) {
  return { type: 'element', tagName, properties, children };
}

export function text(value) {
  return { type: 'text', value };
}
