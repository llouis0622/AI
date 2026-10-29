import type { Pt } from './_data';
import { rng } from '../core/math';

/** CART decision tree (Gini impurity, axis-aligned splits) for 2D points. */
export interface TreeNode {
  /** Feature to split on: 0 = x, 1 = y. Undefined for a leaf. */
  feature?: 0 | 1;
  threshold?: number;
  left?: TreeNode;
  right?: TreeNode;
  /** Class counts reaching this node. */
  counts: [number, number];
  gini: number;
  depth: number;
  /** Rectangle of the input space this node covers. */
  box: [number, number, number, number]; // x0, y0, x1, y1
}

export const gini = (c: [number, number]) => {
  const n = c[0] + c[1];
  if (!n) return 0;
  const p = c[1] / n;
  return 2 * p * (1 - p);
};

const countsOf = (pts: Pt[]): [number, number] => {
  let a = 0;
  let b = 0;
  for (const p of pts) {
    if (p.label) b++;
    else a++;
  }
  return [a, b];
};

export interface TreeOptions {
  maxDepth: number;
  minLeaf?: number;
  /** Random forests: consider a random subset of features at each split. */
  randomFeature?: () => number;
}

export function buildTree(pts: Pt[], opts: TreeOptions, depth = 0, box: [number, number, number, number] = [0, 0, 1, 1]): TreeNode {
  const counts = countsOf(pts);
  const node: TreeNode = { counts, gini: gini(counts), depth, box };
  const minLeaf = opts.minLeaf ?? 1;
  if (depth >= opts.maxDepth || node.gini === 0 || pts.length < 2 * minLeaf) return node;
  let best: { f: 0 | 1; t: number; score: number } | null = null;
  const feats: (0 | 1)[] = opts.randomFeature ? [opts.randomFeature() < 0.5 ? 0 : 1] : [0, 1];
  for (const f of feats) {
    const sorted = [...pts].sort((p, q) => (f ? p.y - q.y : p.x - q.x));
    const left: [number, number] = [0, 0];
    const right = countsOf(sorted);
    for (let i = 0; i < sorted.length - 1; i++) {
      const p = sorted[i];
      left[p.label]++;
      right[p.label]--;
      const v = f ? p.y : p.x;
      const vNext = f ? sorted[i + 1].y : sorted[i + 1].x;
      if (vNext - v < 1e-9) continue;
      const nl = i + 1;
      const nr = sorted.length - nl;
      if (nl < minLeaf || nr < minLeaf) continue;
      const score = (nl * gini(left) + nr * gini(right)) / sorted.length;
      if (!best || score < best.score - 1e-12) best = { f, t: (v + vNext) / 2, score };
    }
  }
  if (!best || best.score >= node.gini - 1e-9) return node;
  node.feature = best.f;
  node.threshold = best.t;
  const L = pts.filter((p) => (best!.f ? p.y : p.x) < best!.t);
  const R = pts.filter((p) => (best!.f ? p.y : p.x) >= best!.t);
  const [x0, y0, x1, y1] = box;
  const lb: [number, number, number, number] = best.f ? [x0, y0, x1, best.t] : [x0, y0, best.t, y1];
  const rb: [number, number, number, number] = best.f ? [x0, best.t, x1, y1] : [best.t, y0, x1, y1];
  node.left = buildTree(L, opts, depth + 1, lb);
  node.right = buildTree(R, opts, depth + 1, rb);
  return node;
}

/** Probability of class 1 at (x, y) (leaf frequency). */
export function predictTree(node: TreeNode, x: number, y: number): number {
  let n = node;
  while (n.feature !== undefined) n = (n.feature ? y : x) < n.threshold! ? n.left! : n.right!;
  const tot = n.counts[0] + n.counts[1];
  return tot ? n.counts[1] / tot : 0.5;
}

export function leaves(node: TreeNode): TreeNode[] {
  if (node.feature === undefined) return [node];
  return [...leaves(node.left!), ...leaves(node.right!)];
}

export function treeDepth(node: TreeNode): number {
  if (node.feature === undefined) return node.depth;
  return Math.max(treeDepth(node.left!), treeDepth(node.right!));
}

export function accuracy(predict: (x: number, y: number) => number, pts: Pt[]) {
  if (!pts.length) return 0;
  return pts.filter((p) => (predict(p.x, p.y) >= 0.5 ? 1 : 0) === p.label).length / pts.length;
}

/**
 * Bagged trees (bootstrap samples), with a random feature choice per split when
 * `randomFeatures` is on — a tiny random forest. Each tree keeps its sample.
 */
export function buildForest(pts: Pt[], nTrees: number, maxDepth: number, seed = 1, randomFeatures = true): { tree: TreeNode; sample: Pt[] }[] {
  // separate streams: the bootstrap samples are the same with or without random features
  const r = rng(seed);
  const rf = rng(seed + 1000);
  const out: { tree: TreeNode; sample: Pt[] }[] = [];
  for (let t = 0; t < nTrees; t++) {
    const sample = Array.from({ length: pts.length }, () => pts[r.int(pts.length)]);
    out.push({ tree: buildTree(sample, { maxDepth, randomFeature: randomFeatures ? rf.next : undefined }), sample });
  }
  return out;
}

/** Fraction of trees voting for class 1 at (x, y). */
export function predictForest(trees: TreeNode[], x: number, y: number): number {
  if (!trees.length) return 0.5;
  let votes = 0;
  for (const t of trees) votes += predictTree(t, x, y) >= 0.5 ? 1 : 0;
  return votes / trees.length;
}
