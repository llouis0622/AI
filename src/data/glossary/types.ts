export interface GlossaryTerm {
  /** URL-safe id, used as the glossary page anchor. */
  id: string;
  /** Display name as it appears in the text, e.g. "파라미터". */
  term: string;
  /** Other spellings that should also get the tooltip, e.g. ["매개변수"]. */
  aliases?: string[];
  /** One line a middle-schooler can understand. */
  short: string;
  /** Optional 2–3 sentence explanation for the glossary page. */
  long?: string;
  emoji?: string;
  /** Chapter that introduces the term (0 = 여는 글). */
  chapter: number;
}
