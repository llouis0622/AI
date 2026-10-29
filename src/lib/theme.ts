export type Theme = 'light' | 'dark';
const KEY = 'aiology:theme';

export function currentTheme(): Theme {
  return document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
}

export function setTheme(theme: Theme) {
  document.documentElement.setAttribute('data-theme', theme);
  try {
    localStorage.setItem(KEY, theme);
  } catch {
    /* ignore */
  }
}

/** Wire every [data-theme-toggle] button. Called once from BaseLayout. */
export function onThemeToggle() {
  document.addEventListener('click', (e) => {
    if (!(e.target as HTMLElement).closest('[data-theme-toggle]')) return;
    const next: Theme = currentTheme() === 'dark' ? 'light' : 'dark';
    // Cross-fade the whole page instead of snapping colors.
    const doc = document as Document & { startViewTransition?: (cb: () => void) => unknown };
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (doc.startViewTransition && !reduce) doc.startViewTransition(() => setTheme(next));
    else setTheme(next);
  });
}

/** Observe theme changes (used by canvas / WebGL visualizations). */
export function observeTheme(cb: (theme: Theme) => void): () => void {
  const mo = new MutationObserver(() => cb(currentTheme()));
  mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  return () => mo.disconnect();
}
