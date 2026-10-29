import { rng } from '../core/math';

/** A labelled 2D point. x, y live in [0, 1]. */
export interface Pt {
  x: number;
  y: number;
  label: 0 | 1;
}

export type DatasetName = 'blobs' | 'overlap' | 'moons' | 'xor' | 'pocket' | 'noisy';

const clamp01 = (v: number) => Math.min(0.98, Math.max(0.02, v));

/** Small seeded 2-class datasets on the unit square. */
export function makeDataset(name: DatasetName, n = 60, seed = 3): Pt[] {
  const r = rng(seed);
  const pts: Pt[] = [];
  for (let i = 0; i < n; i++) {
    let label = (i % 2) as 0 | 1;
    let x = 0;
    let y = 0;
    switch (name) {
      case 'blobs': {
        const c = label ? [0.68, 0.66] : [0.32, 0.34];
        x = c[0] + r.normal() * 0.1;
        y = c[1] + r.normal() * 0.1;
        break;
      }
      case 'overlap': {
        const c = label ? [0.6, 0.58] : [0.4, 0.42];
        x = c[0] + r.normal() * 0.14;
        y = c[1] + r.normal() * 0.14;
        break;
      }
      case 'moons': {
        const t = Math.PI * r.next();
        if (label) {
          x = 0.5 + 0.28 * Math.cos(t + Math.PI) + 0.14;
          y = 0.42 - 0.24 * Math.sin(t + Math.PI) - 0.06;
        } else {
          x = 0.5 + 0.28 * Math.cos(t) - 0.14;
          y = 0.5 + 0.24 * Math.sin(t) - 0.04;
        }
        x += r.normal() * 0.04;
        y += r.normal() * 0.04;
        break;
      }
      case 'xor': {
        const qx = r.next() < 0.5 ? 0 : 1;
        const qy = label ? 1 - qx : qx;
        x = 0.27 + qx * 0.46 + r.normal() * 0.08;
        y = 0.27 + qy * 0.46 + r.normal() * 0.08;
        break;
      }
      case 'noisy': {
        // a wavy diagonal truth, with about 12% of the labels flipped (noise)
        x = r.next();
        y = r.next();
        const truth: 0 | 1 = y > 0.25 + 0.5 * x + 0.12 * Math.sin(x * 7) ? 1 : 0;
        label = r.next() < 0.12 ? ((1 - truth) as 0 | 1) : truth;
        break;
      }
      case 'pocket': {
        // class 1 = a ring-ish region in the top-right plus a pocket bottom-left
        if (label) {
          const pick = r.next();
          if (pick < 0.65) {
            x = 0.72 + r.normal() * 0.09;
            y = 0.7 + r.normal() * 0.09;
          } else {
            x = 0.24 + r.normal() * 0.06;
            y = 0.26 + r.normal() * 0.06;
          }
        } else {
          x = 0.18 + r.next() * 0.7;
          y = 0.18 + r.next() * 0.7;
          if (x > 0.58 && y > 0.56) {
            x -= 0.32;
          }
        }
        break;
      }
    }
    pts.push({ x: clamp01(x), y: clamp01(y), label });
  }
  return pts;
}

/** Deterministic train / test split: every k-th *pair* goes to test, so both classes land there. */
export function splitTrainTest(pts: Pt[], every = 4) {
  return {
    train: pts.filter((_, i) => Math.floor(i / 2) % every !== 0),
    test: pts.filter((_, i) => Math.floor(i / 2) % every === 0)
  };
}
