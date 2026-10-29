import type { VizMeta } from './types';

export const viz: VizMeta[] = [
  {
    id: 'line-fit',
    title: '직선 맞추기 놀이터',
    tagline: '손잡이를 끌어 집값 직선을 맞추면, 오차가 네모로 보여요.',
    icon: '📏',
    component: 'part1/LineFitPlayground',
    chapter: 2,
    howTo: '빨간 네모의 넓이가 곧 "오차의 제곱"이에요. 네모를 다 합친 평균이 손실이랍니다.',
    aspect: 1.25,
    featured: true
  },
  {
    id: 'gradient-descent-3d',
    title: '3D 경사하강법 실험실',
    tagline: '오차 지형 위에서 공이 골짜기 바닥을 찾아 굴러가는 모습을 직접 봐요.',
    icon: '⛰️',
    component: 'part1/GradientDescent3D',
    chapter: 2,
    is3D: true,
    howTo: '학습률을 "너무 크게"로 바꿔 보세요. 공이 골짜기를 넘어 튀어 오르는 "발산"을 볼 수 있어요.',
    aspect: 1.3,
    featured: true
  }
];
