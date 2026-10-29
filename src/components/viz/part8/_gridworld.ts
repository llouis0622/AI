/** Tiny tabular Q-learning gridworld shared by the chapter 25 visualizations. */

export type Cell = '.' | '#' | 'F' | 'G';
export const ACTIONS = [
  [0, -1],
  [1, 0],
  [0, 1],
  [-1, 0]
] as const; // up, right, down, left
export const ARROWS = ['↑', '→', '↓', '←'];

export interface World {
  w: number;
  h: number;
  cells: Cell[];
  start: number;
}

export const DEFAULT_ROWS = ['....#G', '.##.#.', '..F...', '#...F.', 'S.#...'];

export function parseWorld(rows: string[]): World {
  const h = rows.length;
  const w = rows[0].length;
  const cells: Cell[] = [];
  let start = 0;
  rows.forEach((row, y) =>
    row.split('').forEach((ch, x) => {
      if (ch === 'S') {
        start = y * w + x;
        cells.push('.');
      } else cells.push(ch as Cell);
    })
  );
  return { w, h, cells, start };
}

export const isTerminal = (w: World, s: number) => w.cells[s] === 'G' || w.cells[s] === 'F';

export function envStep(w: World, s: number, a: number): { s2: number; r: number; done: boolean } {
  const x = s % w.w;
  const y = Math.floor(s / w.w);
  const nx = x + ACTIONS[a][0];
  const ny = y + ACTIONS[a][1];
  let s2 = s;
  if (nx >= 0 && nx < w.w && ny >= 0 && ny < w.h && w.cells[ny * w.w + nx] !== '#') s2 = ny * w.w + nx;
  const cell = w.cells[s2];
  if (cell === 'G') return { s2, r: 1, done: true };
  if (cell === 'F') return { s2, r: -1, done: true };
  return { s2, r: 0, done: false };
}

export const maxQ = (Q: Float64Array, s: number) => Math.max(Q[s * 4], Q[s * 4 + 1], Q[s * 4 + 2], Q[s * 4 + 3]);

export function argmaxQ(Q: Float64Array, s: number, rand: () => number): number {
  const m = maxQ(Q, s);
  const best: number[] = [];
  for (let a = 0; a < 4; a++) if (Math.abs(Q[s * 4 + a] - m) < 1e-12) best.push(a);
  return best[Math.floor(rand() * best.length)];
}

export function chooseAction(Q: Float64Array, s: number, eps: number, rand: () => number): number {
  return rand() < eps ? Math.floor(rand() * 4) : argmaxQ(Q, s, rand);
}

/** One Q-learning update; returns the TD error. */
export function qUpdate(Q: Float64Array, s: number, a: number, r: number, s2: number, done: boolean, alpha: number, gamma: number): number {
  const target = r + (done ? 0 : gamma * maxQ(Q, s2));
  const td = target - Q[s * 4 + a];
  Q[s * 4 + a] += alpha * td;
  return td;
}
