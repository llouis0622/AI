/**
 * Learning progress kept in localStorage (this browser only). Shared by React
 * islands and the small vanilla scripts in layouts.
 */
const KEY = 'aiology:progress:v1';
export const PROGRESS_EVENT = 'aiology:progress';

export interface Mark {
  visited?: number;
  done?: number;
}

export interface ProgressState {
  chapters: Record<string, Mark>;
  labs: Record<string, Mark>;
  viz: Record<string, number>;
}

const empty = (): ProgressState => ({ chapters: {}, labs: {}, viz: {} });

export function loadProgress(): ProgressState {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return empty();
    const parsed = JSON.parse(raw) as Partial<ProgressState>;
    return { chapters: parsed.chapters ?? {}, labs: parsed.labs ?? {}, viz: parsed.viz ?? {} };
  } catch {
    return empty();
  }
}

function update(mutate: (s: ProgressState) => boolean) {
  const state = loadProgress();
  if (!mutate(state)) return;
  try {
    localStorage.setItem(KEY, JSON.stringify(state));
  } catch {
    /* storage unavailable (private mode, blocked) — progress just won't persist */
  }
  window.dispatchEvent(new CustomEvent(PROGRESS_EVENT));
}

export function markChapterVisited(chapter: number) {
  update((s) => {
    const m = (s.chapters[chapter] ??= {});
    if (m.visited) return false;
    m.visited = Date.now();
    return true;
  });
}

export function setChapterDone(chapter: number, done: boolean) {
  update((s) => {
    const m = (s.chapters[chapter] ??= {});
    m.visited ??= Date.now();
    if (done) m.done = Date.now();
    else delete m.done;
    return true;
  });
}

export function markLabVisited(lab: number) {
  update((s) => {
    const m = (s.labs[lab] ??= {});
    if (m.visited) return false;
    m.visited = Date.now();
    return true;
  });
}

export function setLabDone(lab: number, done: boolean) {
  update((s) => {
    const m = (s.labs[lab] ??= {});
    m.visited ??= Date.now();
    if (done) m.done = Date.now();
    else delete m.done;
    return true;
  });
}

export function markViz(id: string) {
  update((s) => {
    if (s.viz[id]) return false;
    s.viz[id] = Date.now();
    return true;
  });
}

export function resetProgress() {
  try {
    localStorage.removeItem(KEY);
  } catch {
    /* ignore */
  }
  window.dispatchEvent(new CustomEvent(PROGRESS_EVENT));
}

/** Subscribe to progress changes from this tab and other tabs. */
export function onProgress(cb: () => void): () => void {
  const onStorage = (e: StorageEvent) => {
    if (e.key === KEY) cb();
  };
  window.addEventListener(PROGRESS_EVENT, cb);
  window.addEventListener('storage', onStorage);
  return () => {
    window.removeEventListener(PROGRESS_EVENT, cb);
    window.removeEventListener('storage', onStorage);
  };
}
