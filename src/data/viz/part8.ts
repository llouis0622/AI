import type { VizMeta } from './types';

export const viz: VizMeta[] = [
  {
    id: 'discount-factor',
    title: '할인율 γ = 참을성 다이얼',
    tagline: '지금 사탕 1개 vs 나중에 사탕 5개, γ에 따라 로봇의 선택이 바뀌어요.',
    icon: '🍬',
    component: 'part8/DiscountFactor',
    chapter: 25,
    howTo: 'γ를 천천히 올려 보세요. 어느 순간 로봇이 "기다리기"로 마음을 바꿔요.',
    aspect: 1.25
  },
  {
    id: 'gridworld-q',
    title: 'Q-학습 보물찾기',
    tagline: '로봇이 시행착오만으로 보석까지 가는 길을 배우는 모습을 지켜봐요.',
    icon: '🤖',
    component: 'part8/GridworldQ',
    chapter: 25,
    howTo: '"⚡ 50판 순식간에"를 눌러 보세요. 초록색 가치가 보석에서 거꾸로 퍼져요. "지도 고치기"로 벽과 함정을 옮겨 볼 수도 있어요!',
    aspect: 1.05,
    featured: true
  },
  {
    id: 'value-terrain-3d',
    title: '가치의 지형 3D',
    tagline: '칸마다 가치를 높이로 쌓으면, 보석에서 시작한 산이 거꾸로 자라나 출발점까지 이어져요.',
    icon: '⛰️',
    component: 'part8/ValueTerrain3D',
    chapter: 25,
    is3D: true,
    howTo: '"⚡ 50판 순식간에"를 몇 번 눌러 산을 키운 뒤, γ를 0.5로 낮추고 다시 눌러 보세요. 산이 뾰족해져요!',
    aspect: 1.05
  },
  {
    id: 'policy-gradient-bandit',
    title: '정책 경사 슬롯머신',
    tagline: '결과가 좋았던 기계의 확률을 올려 가며, 가장 좋은 기계를 스스로 찾아요.',
    icon: '🎰',
    component: 'part8/PolicyGradientBandit',
    chapter: 26,
    howTo: '기준선을 끄고 켜 보세요. "기준선 비교 실험"에서는 16번 반복한 결과로 흔들림(분산)을 비교해요.',
    aspect: 1.15
  },
  {
    id: 'ppo-clip',
    title: 'PPO 클리핑',
    tagline: '"좋은 방향이라도 한 번에 조금만" — 잘라 낸 목적 함수를 직접 만져 봐요.',
    icon: '✂️',
    component: 'part8/PPOClip',
    chapter: 26,
    howTo: 'ρ를 안전 범위 밖으로 밀면 그래프가 평평해져요. "10번 배우기"에서 클리핑이 붕괴를 막는 걸 확인해요.',
    aspect: 1.2
  }
];
