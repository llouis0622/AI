import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

/** One card of the summary strip shown at the top of a chapter. */
const summaryCard = z.object({
  title: z.string(),
  text: z.string(),
  /** Legacy fields from the emoji cards; ignored. */
  icon: z.string().optional(),
  visual: z.string().optional()
});

const book = defineCollection({
  loader: glob({ pattern: '**/*.mdx', base: './src/content/book' }),
  schema: z.object({
    title: z.string(),
    /** 0 = 여는 글 */
    chapter: z.number().int().min(0),
    part: z.number().int().min(0),
    /** "이 장의 질문" — may contain **bold** markers. */
    question: z.string().optional(),
    /** "전제" line — may contain markdown links. */
    prereq: z.string().optional(),
    /** Legacy mascot line; ignored. */
    mascot: z.string().optional(),
    summary: z.array(summaryCard).default([]),
    /** Related code labs by number. */
    labs: z.array(z.number().int()).default([])
  })
});

const practice = defineCollection({
  loader: glob({ pattern: '**/*.mdx', base: './src/content/practice' }),
  schema: z.object({
    title: z.string(),
    /** 0 = 코드랩 개요 */
    lab: z.number().int().min(0),
    track: z.string(),
    /** One-line description of what gets built. */
    builds: z.string().optional(),
    /** Chapters to read first, by number. */
    chapters: z.array(z.number().int()).default([]),
    summary: z.array(summaryCard).default([])
  })
});

const review = defineCollection({
  loader: glob({ pattern: '**/*.mdx', base: './src/content/review' }),
  schema: z.object({
    title: z.string(),
    /** 0 = 복습 허브 */
    part: z.number().int().min(0),
    total: z.number().int().default(0)
  })
});

export const collections = { book, practice, review };
