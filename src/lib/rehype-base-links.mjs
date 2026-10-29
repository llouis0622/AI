import { walk } from './hast-walk.mjs';

/**
 * Markdown links like `/book/02-first-learning-machine` are written site-absolute.
 * Prefix them with the deploy base and add the trailing slash that
 * `trailingSlash: 'always'` expects. External links open in a new tab.
 */
export default function rehypeBaseLinks({ base = '' } = {}) {
  const prefix = base.replace(/\/$/, '');
  return (tree) => {
    walk(tree, (node) => {
      if (node.type !== 'element' || node.tagName !== 'a') return;
      const href = node.properties?.href;
      if (typeof href !== 'string') return;
      if (href.startsWith('/') && !href.startsWith('//')) {
        if (prefix && (href === prefix || href.startsWith(`${prefix}/`))) return;
        node.properties.href = prefix + withTrailingSlash(href);
      } else if (/^https?:\/\//.test(href)) {
        node.properties.target = '_blank';
        node.properties.rel = ['noopener', 'noreferrer'];
      }
    });
  };
}

function withTrailingSlash(href) {
  const [, path, rest] = href.match(/^([^?#]*)(.*)$/);
  const needsSlash = !path.endsWith('/') && !/\.[a-z0-9]+$/i.test(path);
  return (needsSlash ? `${path}/` : path) + rest;
}
