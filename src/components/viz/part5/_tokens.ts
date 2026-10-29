/** Shared helpers for part 5 visualizations (token chips, BPE). */

/** Stable hue for a token string so the same piece always gets the same color. */
export function tokenHue(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return Math.abs(h) % 360;
}

/** Chip colors for a token in the current theme. */
export function chipColors(s: string, theme: 'light' | 'dark') {
  const h = tokenHue(s);
  return theme === 'dark'
    ? { bg: `hsla(${h}, 90%, 60%, 0.16)`, border: `hsla(${h}, 95%, 66%, 0.75)`, text: `hsl(${h}, 95%, 80%)` }
    : { bg: `hsl(${h}, 85%, 92%)`, border: `hsl(${h}, 55%, 62%)`, text: `hsl(${h}, 55%, 28%)` };
}

export interface PairCount {
  a: string;
  b: string;
  count: number;
  first: number;
}

export interface BpeState {
  segs: string[][];
  ranked: PairCount[];
  merges: [string, string][];
}

export function mergePair(seg: string[], a: string, b: string): string[] {
  const out: string[] = [];
  for (let i = 0; i < seg.length; i++) {
    if (i < seg.length - 1 && seg[i] === a && seg[i + 1] === b) {
      out.push(a + b);
      i++;
    } else out.push(seg[i]);
  }
  return out;
}

/**
 * Train BPE on (word, count) pairs: repeatedly merge the most frequent adjacent
 * pair (ties → earliest occurrence). Stops when no pair repeats.
 * Returns every intermediate state so the UI can step through them.
 */
export function trainBpe(words: [string, number][], maxMerges = 40): BpeState[] {
  let segs = words.map(([w]) => Array.from(w));
  const merges: [string, string][] = [];
  const history: BpeState[] = [];
  for (let step = 0; step <= maxMerges; step++) {
    const counts = new Map<string, PairCount>();
    segs.forEach((seg, wi) => {
      for (let i = 0; i < seg.length - 1; i++) {
        const key = `${seg[i]}\u0001${seg[i + 1]}`;
        const prev = counts.get(key);
        if (prev) prev.count += words[wi][1];
        else counts.set(key, { a: seg[i], b: seg[i + 1], count: words[wi][1], first: wi * 1000 + i });
      }
    });
    const ranked = [...counts.values()].sort((x, y) => y.count - x.count || x.first - y.first);
    history.push({ segs: segs.map((s) => [...s]), ranked: ranked.slice(0, 5), merges: [...merges] });
    if (!ranked.length || ranked[0].count < 2) break;
    const { a, b } = ranked[0];
    merges.push([a, b]);
    segs = segs.map((s) => mergePair(s, a, b));
  }
  return history;
}

/** Encode text with the first `k` learned merges (classic BPE: apply rules in order). */
export function encodeBpe(text: string, merges: [string, string][], k: number): string[][] {
  return text
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .map((w) => {
      let seg = Array.from(w);
      for (let i = 0; i < Math.min(k, merges.length); i++) seg = mergePair(seg, merges[i][0], merges[i][1]);
      return seg;
    });
}
