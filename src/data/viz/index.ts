import type { VizMeta } from './types';
import { viz as part1 } from './part1';
import { viz as part2 } from './part2';
import { viz as part3 } from './part3';
import { viz as part4 } from './part4';
import { viz as part5 } from './part5';
import { viz as part6 } from './part6';
import { viz as part7 } from './part7';
import { viz as part8 } from './part8';
import { viz as part9 } from './part9';
import { viz as labs } from './labs';

export type { VizMeta };

export const allViz: VizMeta[] = [
  ...part1, ...part2, ...part3, ...part4, ...part5, ...part6, ...part7, ...part8, ...part9, ...labs
];

export const vizById: Record<string, VizMeta> = Object.fromEntries(allViz.map((v) => [v.id, v]));
