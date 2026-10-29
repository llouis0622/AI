import { getCollection, type CollectionEntry } from 'astro:content';
import { url } from '@/data/curriculum';

export type BookEntry = CollectionEntry<'book'>;
export type LabEntry = CollectionEntry<'practice'>;
export type QuizEntry = CollectionEntry<'review'>;

export const bookHref = (id: string) => url(id === '00-intro' ? '/book/' : `/book/${id}/`);
export const labHref = (id: string) => url(id === '00-overview' ? '/practice/' : `/practice/${id}/`);
export const quizHref = (id: string) => url(id === 'index' ? '/review/' : `/review/${id}/`);

export async function getBook() {
  return (await getCollection('book')).sort((a, b) => a.data.chapter - b.data.chapter);
}

export async function getLabs() {
  return (await getCollection('practice')).sort((a, b) => a.data.lab - b.data.lab);
}

export async function getQuizzes() {
  return (await getCollection('review')).sort((a, b) => a.data.part - b.data.part);
}

/** Rough reading time for Korean prose: ~480 characters per minute, code excluded. */
export function readingMinutes(body = '') {
  const prose = body
    .replace(/```[\s\S]*?```/g, '')
    .replace(/^import .*$/gm, '')
    .replace(/<[^>]+>/g, '')
    .replace(/\$\$[\s\S]*?\$\$/g, '');
  return Math.max(3, Math.round(prose.replace(/\s+/g, '').length / 480));
}

export function vizIdsIn(body = '') {
  return [...body.matchAll(/<Viz\s+id="([^"]+)"/g)].map((m) => m[1]);
}

/** Strip markdown emphasis for meta descriptions. */
export const plain = (s = '') => s.replace(/\*\*/g, '').replace(/\[([^\]]+)\]\([^)]*\)/g, '$1');
