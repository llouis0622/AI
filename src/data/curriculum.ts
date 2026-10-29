export interface PartInfo {
  part: number;
  title: string;
  emoji: string;
  /** One friendly sentence about what the part is about (middle-school level). */
  blurb: string;
  chapters: number[];
}

export const PARTS: PartInfo[] = [
  {
    part: 1,
    title: '기계가 배운다는 것',
    emoji: '🌱',
    blurb: '"배운다"가 정확히 무슨 뜻인지, 가장 단순한 학습 기계를 뜯어보며 알아봐요.',
    chapters: [1, 2, 3, 4]
  },
  {
    part: 2,
    title: '고전 머신러닝',
    emoji: '🌳',
    blurb: '신경망 이전부터 쓰인 든든한 도구들 — 분류, 나무, 무리 찾기, 그리고 성적표 읽기.',
    chapters: [5, 6, 7, 8]
  },
  {
    part: 3,
    title: '신경망',
    emoji: '🧠',
    blurb: '뉴런을 쌓아 만든 기계가 어떻게 배우고, 왜 가끔 아픈지까지.',
    chapters: [9, 10, 11, 12]
  },
  {
    part: 4,
    title: '지각',
    emoji: '👀',
    blurb: '기계에게 눈과 귀를 달아 줘요 — 사진을 보고, 소리를 듣는 원리.',
    chapters: [13, 14, 15]
  },
  {
    part: 5,
    title: '언어',
    emoji: '💬',
    blurb: '말을 숫자로 바꾸고, 문맥을 읽는 어텐션, 그리고 Transformer.',
    chapters: [16, 17, 18]
  },
  {
    part: 6,
    title: '생성',
    emoji: '🎨',
    blurb: '구분하는 기계에서 만들어내는 기계로 — VAE, GAN, 확산 모델.',
    chapters: [19, 20]
  },
  {
    part: 7,
    title: '거대 언어 모델',
    emoji: '🤖',
    blurb: '다음 단어 맞히기가 어떻게 ChatGPT 같은 조수가 되는지.',
    chapters: [21, 22, 23, 24]
  },
  {
    part: 8,
    title: '행동',
    emoji: '🎮',
    blurb: '정답 없이 보상만으로 배우는 강화학습 — 게임을 정복한 그 기술.',
    chapters: [25, 26]
  },
  {
    part: 9,
    title: '현장',
    emoji: '🚀',
    blurb: '만든 모델을 세상에 내보내는 법, 그리고 계속 공부하는 법.',
    chapters: [27, 28]
  }
];

export const partOf = (chapter: number) => PARTS.find((p) => p.chapters.includes(chapter));

export interface TrackInfo {
  id: string;
  title: string;
  emoji: string;
  labs: number[];
}

export const TRACKS: TrackInfo[] = [
  { id: 'A', title: 'A. 기초', emoji: '🧱', labs: [1, 2, 3] },
  { id: 'B', title: 'B. 컴퓨터 비전', emoji: '🖼️', labs: [4, 5, 6] },
  { id: 'C', title: 'C. NLP와 LLM', emoji: '📝', labs: [7, 8, 9, 10, 11, 12, 13] },
  { id: 'D', title: 'D. 생성 모델', emoji: '🎨', labs: [14, 15] },
  { id: 'E', title: 'E. 강화학습', emoji: '🕹️', labs: [16, 17] },
  { id: 'F', title: 'F. 멀티모달과 운영', emoji: '🛰️', labs: [18, 19, 20] }
];

/** Prefix a site path with the deploy base (BASE_URL ends with '/'). */
export function url(path = '/') {
  const base = import.meta.env.BASE_URL.replace(/\/$/, '');
  const clean = path.startsWith('/') ? path : `/${path}`;
  return `${base}${clean}`;
}

export const chapterLabel = (chapter: number) => (chapter === 0 ? '여는 글' : `${chapter}장`);
