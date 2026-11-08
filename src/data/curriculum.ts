export interface PartInfo {
  part: number;
  title: string;
  /** One plain sentence about what the part covers. */
  blurb: string;
  chapters: number[];
}

export const PARTS: PartInfo[] = [
  {
    part: 1,
    title: '기계가 배운다는 것',
    blurb: "가장 단순한 학습 기계를 하나씩 뜯어보며 '배운다'가 무슨 뜻인지 알아봐요.",
    chapters: [1, 2, 3, 4]
  },
  {
    part: 2,
    title: '고전 머신러닝',
    blurb: '신경망 이전부터 쓰여 온 분류, 결정 트리, 군집과 모델 평가를 다뤄요.',
    chapters: [5, 6, 7, 8]
  },
  {
    part: 3,
    title: '신경망',
    blurb: '뉴런을 쌓은 모델이 어떻게 배우는지, 학습이 왜 가끔 실패하는지 살펴봐요.',
    chapters: [9, 10, 11, 12]
  },
  {
    part: 4,
    title: '지각',
    blurb: '사진을 보고 소리를 듣는 모델이 어떻게 동작하는지 배워요.',
    chapters: [13, 14, 15]
  },
  {
    part: 5,
    title: '언어',
    blurb: '말을 숫자로 바꾸는 방법부터 어텐션과 Transformer까지 이어서 봐요.',
    chapters: [16, 17, 18]
  },
  {
    part: 6,
    title: '생성',
    blurb: 'VAE, GAN, 확산 모델처럼 새 데이터를 만들어 내는 모델을 다뤄요.',
    chapters: [19, 20]
  },
  {
    part: 7,
    title: '거대 언어 모델',
    blurb: '다음 단어 맞히기가 어떻게 ChatGPT 같은 조수로 이어지는지 따라가요.',
    chapters: [21, 22, 23, 24]
  },
  {
    part: 8,
    title: '행동',
    blurb: '정답 없이 보상만으로 배우는 강화학습을 다뤄요.',
    chapters: [25, 26]
  },
  {
    part: 9,
    title: '현장',
    blurb: '모델을 서비스로 내보내는 방법과, 그다음에 계속 공부하는 방법을 정리해요.',
    chapters: [27, 28]
  }
];

export const partOf = (chapter: number) => PARTS.find((p) => p.chapters.includes(chapter));

export interface TrackInfo {
  id: string;
  title: string;
  labs: number[];
}

export const TRACKS: TrackInfo[] = [
  { id: 'A', title: 'A. 기초', labs: [1, 2, 3] },
  { id: 'B', title: 'B. 컴퓨터 비전', labs: [4, 5, 6] },
  { id: 'C', title: 'C. NLP와 LLM', labs: [7, 8, 9, 10, 11, 12, 13] },
  { id: 'D', title: 'D. 생성 모델', labs: [14, 15] },
  { id: 'E', title: 'E. 강화학습', labs: [16, 17] },
  { id: 'F', title: 'F. 멀티모달과 운영', labs: [18, 19, 20] }
];

/** Prefix a site path with the deploy base (BASE_URL ends with '/'). */
export function url(path = '/') {
  const base = import.meta.env.BASE_URL.replace(/\/$/, '');
  const clean = path.startsWith('/') ? path : `/${path}`;
  return `${base}${clean}`;
}

export const chapterLabel = (chapter: number) => (chapter === 0 ? '여는 글' : `${chapter}장`);
