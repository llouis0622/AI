/**
 * A hand-made 3D "embedding space" for chapter 16. Real embeddings have hundreds
 * of dimensions; this toy keeps the two properties the chapter describes:
 * nearby = similar (clusters), and consistent directions = relations
 * (male→female, country→capital).
 */
export type Vec3 = [number, number, number];

export type CategoryKey = 'animal' | 'food' | 'vehicle' | 'emotion' | 'study' | 'people' | 'country' | 'capital' | 'ambiguous';

export const CATEGORIES: Record<CategoryKey, { label: string; emoji: string; hue: number }> = {
  animal: { label: '동물', emoji: '🐾', hue: 24 },
  food: { label: '음식', emoji: '🍎', hue: 350 },
  vehicle: { label: '탈것', emoji: '🚗', hue: 205 },
  emotion: { label: '감정', emoji: '💗', hue: 305 },
  study: { label: '학문', emoji: '📐', hue: 262 },
  people: { label: '사람·왕족', emoji: '👑', hue: 44 },
  country: { label: '나라', emoji: '🗺️', hue: 140 },
  capital: { label: '수도', emoji: '🏙️', hue: 172 },
  ambiguous: { label: '뜻이 여럿', emoji: '❓', hue: 220 }
};

export interface Word {
  w: string;
  cat: CategoryKey;
  p: Vec3;
}

const norm = (v: Vec3): Vec3 => {
  const l = Math.hypot(...v);
  return [v[0] / l, v[1] / l, v[2] / l];
};
export const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scale = (a: Vec3, k: number): Vec3 => [a[0] * k, a[1] * k, a[2] * k];
const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const cosine = (a: Vec3, b: Vec3) => (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (Math.hypot(...a) * Math.hypot(...b) || 1);

const R = 3.7;
const DIRS: Record<string, Vec3> = {
  food: norm([1, 1, 1]),
  vehicle: norm([1, -1, 1]),
  animal: norm([1, 1, -1]),
  country: norm([1, -1, -1]),
  people: norm([-1, 1, 1]),
  emotion: norm([-1, 1, -1]),
  study: norm([-1, -1, 1])
};

function build(): Word[] {
  let seed = 7;
  const rnd = () => {
    seed = (seed * 1103515245 + 12345) % 2147483648;
    return seed / 2147483648;
  };
  const jitter = (k: number): Vec3 => [(rnd() * 2 - 1) * k, (rnd() * 2 - 1) * k, (rnd() * 2 - 1) * k];
  const words: Word[] = [];
  const TAN: [number, number][] = [[0, 0], [0.6, 0.1], [-0.45, 0.45], [0.15, -0.6], [-0.55, -0.35], [0.5, 0.55], [-0.15, 0.75], [0.65, -0.45]];
  const cluster = (cat: CategoryKey, list: string[], offs = TAN) => {
    const d = DIRS[cat];
    const u = norm(cross([0, 1, 0], d));
    const v = cross(d, u);
    const center = scale(d, R);
    list.forEach((w, i) => {
      const [a, b] = offs[i % offs.length];
      words.push({ w, cat, p: add(add(add(center, scale(u, a * 1.15)), scale(v, b * 1.15)), jitter(0.06)) });
    });
  };
  const ANIMAL_TAN: [number, number][] = [[-0.55, 0.15], [-0.32, 0.36], [0.45, 0.2], [0.68, 0.48], [-0.78, -0.25], [0.35, -0.32], [-0.88, 0.48], [0.88, -0.12]];
  cluster('animal', ['고양이', '강아지', '호랑이', '사자', '토끼', '늑대', '햄스터', '코끼리'], ANIMAL_TAN);
  cluster('food', ['사과', '바나나', '딸기', '피자', '라면', '김치', '빵', '떡볶이']);
  cluster('vehicle', ['자동차', '버스', '기차', '비행기', '자전거', '지하철']);
  cluster('emotion', ['기쁨', '슬픔', '사랑', '분노', '행복', '두려움']);
  cluster('study', ['수학', '물리', '화학', '생물', '미분방정식']);
  {
    // people: a little cube of consistent directions
    const base = add(scale(DIRS.people, R), [-0.7, -0.9, 0.8]);
    const g: Vec3 = [0, 0, -1.6]; // → female
    const r: Vec3 = [0, 1.8, 0]; // → royal
    const y: Vec3 = [1.4, 0, 0]; // → young
    const items: [string, Vec3[]][] = [
      ['남자', []],
      ['여자', [g]],
      ['왕', [r]],
      ['여왕', [r, g]],
      ['소년', [y]],
      ['소녀', [y, g]],
      ['왕자', [y, r]],
      ['공주', [y, r, g]]
    ];
    for (const [w, vs] of items) words.push({ w, cat: 'people', p: add(vs.reduce((acc, v) => add(acc, v), base), jitter(0.08)) });
  }
  {
    // countries and their capitals: same offset for every pair
    const C = scale(DIRS.country, R);
    const cap: Vec3 = [0, 1.6, 0.6];
    const xs = [-1.5, -0.9, -0.3, 0.3, 0.9, 1.5];
    const zs = [0.2, -0.3, 0.3, -0.2, 0.25, -0.25];
    const pairs: [string, string][] = [
      ['한국', '서울'],
      ['일본', '도쿄'],
      ['프랑스', '파리'],
      ['영국', '런던'],
      ['미국', '워싱턴'],
      ['중국', '베이징']
    ];
    pairs.forEach(([country, capital], i) => {
      const pc = add(add(C, [xs[i], -0.5, zs[i]]), jitter(0.06));
      words.push({ w: country, cat: 'country', p: pc });
      words.push({ w: capital, cat: 'capital', p: add(add(pc, cap), jitter(0.07)) });
    });
  }
  // "배" (ship / pear / belly): one vector has to sit between meanings.
  words.push({ w: '배', cat: 'ambiguous', p: scale(norm([1, 0.05, 1]), R * 0.82) });
  return words;
}

export const WORDS: Word[] = build();
export const indexOf = (w: string) => WORDS.findIndex((x) => x.w === w);

export function nearest(v: Vec3, exclude: number[], k = 5): { i: number; sim: number }[] {
  return WORDS.map((x, i) => ({ i, sim: cosine(v, x.p) }))
    .filter((x) => !exclude.includes(x.i))
    .sort((a, b) => b.sim - a.sim)
    .slice(0, k);
}

export const ANALOGIES: { a: string; b: string; c: string; relation: string }[] = [
  { a: '왕', b: '남자', c: '여자', relation: '"왕이 되는" 방향 👑' },
  { a: '파리', b: '프랑스', c: '일본', relation: '"나라 → 수도" 방향 🏙️' },
  { a: '공주', b: '여자', c: '남자', relation: '"어린 왕족이 되는" 방향 👑' },
  { a: '서울', b: '한국', c: '영국', relation: '"나라 → 수도" 방향 🏙️' }
];
