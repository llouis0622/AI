import type { GlossaryTerm } from './types';
import { terms as part1 } from './part1';
import { terms as part2 } from './part2';
import { terms as part3 } from './part3';
import { terms as part4 } from './part4';
import { terms as part5 } from './part5';
import { terms as part6 } from './part6';
import { terms as part7 } from './part7';
import { terms as part8 } from './part8';
import { terms as part9 } from './part9';

export type { GlossaryTerm };

/**
 * All glossary terms, deduplicated by id and by display name. When two parts
 * define the same term, the one introduced in the earlier chapter wins.
 */
export const glossaryTerms: GlossaryTerm[] = (() => {
  const all = [part1, part2, part3, part4, part5, part6, part7, part8, part9]
    .flat()
    .sort((a, b) => a.chapter - b.chapter);
  const seenIds = new Set<string>();
  const seenNames = new Set<string>();
  const out: GlossaryTerm[] = [];
  for (const t of all) {
    const name = t.term.replace(/\s+/g, '');
    if (seenIds.has(t.id) || seenNames.has(name)) continue;
    seenIds.add(t.id);
    seenNames.add(name);
    out.push(t);
  }
  return out;
})();
