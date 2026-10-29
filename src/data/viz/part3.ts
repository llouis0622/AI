import type { VizMeta } from './types';

export const viz: VizMeta[] = [
  {
    id: 'xor-transform',
    title: 'XOR의 벽 넘기',
    tagline: '직선 하나로는 못 가르는 점 네 개를, 은닉층이 공간을 접어 해결해요.',
    icon: '🧩',
    component: 'part3/XorTransform',
    chapter: 9,
    howTo: '① 단계에서 직선을 아무리 돌려도 3개가 최대예요. ② 단계에서 공간을 접어 보세요!',
    aspect: 1.15
  },
  {
    id: 'relu-approx',
    title: 'ReLU 조각 잇기',
    tagline: '꺾인 막대기(ReLU)를 이어 붙이면 어떤 곡선이든 흉내 낼 수 있어요.',
    icon: '📐',
    component: 'part3/ReluApprox',
    chapter: 9,
    howTo: '"ReLU 빼기"를 켜 보세요. 뉴런을 아무리 늘려도 직선 하나가 돼요.',
    aspect: 1.25
  },
  {
    id: 'nn-playground',
    title: '신경망 놀이터',
    tagline: '층과 뉴런을 골라 진짜 신경망을 브라우저에서 학습시켜 봐요.',
    icon: '🧠',
    component: 'part3/NNPlayground',
    chapter: 9,
    howTo: '"나선" 데이터는 뉴런 하나로는 못 풀어요. 층과 뉴런을 늘리고 뉴런 그림을 눌러 보세요.',
    aspect: 0.95,
    featured: true
  },
  {
    id: 'backprop-graph',
    title: '역전파 한 칸씩',
    tagline: '계산 그래프에서 값은 앞으로, 민감도는 뒤로 흐르는 걸 한 칸씩 따라가요.',
    icon: '🔁',
    component: 'part3/BackpropGraph',
    chapter: 10,
    howTo: '끝까지 간 뒤 "검산하기"를 눌러 보세요. 입력 x를 음수로 바꾸면 ReLU가 민감도를 막아요.',
    aspect: 1.15
  },
  {
    id: 'vanishing-product',
    title: '곱셈의 재앙',
    tagline: '층마다 곱해지는 작은 수가 기울기를 어떻게 지워 버리는지 봐요.',
    icon: '📉',
    component: 'part3/VanishingProduct',
    chapter: 10,
    howTo: '"시그모이드 0.25"와 "ReLU 1.0"을 번갈아 눌러 보세요. 층 수도 바꿔 보고요.',
    aspect: 1.3
  },
  {
    id: 'normalization-histograms',
    title: '층별 신호 건강검진',
    tagline: '초기화와 정규화 층이 층마다 신호 크기를 어떻게 지키는지 봐요.',
    icon: '📊',
    component: 'part3/NormalizationHistograms',
    chapter: 11,
    howTo: '"너무 작게" → "He 초기화" → "정규화 층" 순서로 눌러 비교해 보세요.',
    aspect: 1
  },
  {
    id: 'residual-highway',
    title: '잔차 고속도로',
    tagline: '1을 더하는 지름길 하나가 깊은 망의 기울기를 살려 내요.',
    icon: '🛣️',
    component: 'part3/ResidualHighway',
    chapter: 11,
    howTo: '층 수를 100까지 늘려 보세요. 일반 망은 사라져도 잔차 망은 살아 있어요.',
    aspect: 1.45
  },
  {
    id: 'optimizer-race-3d',
    title: '옵티마이저 경주',
    tagline: 'SGD·모멘텀·RMSProp·Adam이 같은 오차 지형에서 바닥까지 경주해요.',
    icon: '🏎️',
    component: 'part3/OptimizerRace3D',
    chapter: 11,
    is3D: true,
    howTo: '지형을 바꿔 가며 경주해 보세요. "안장점"에서는 누가 먼저 빠져나올까요?',
    aspect: 1.2
  },
  {
    id: 'loss-doctor',
    title: '학습 곡선 진료실',
    tagline: '증상을 보고 처방을 눌러, 병든 학습을 직접 고쳐 봐요.',
    icon: '🩺',
    component: 'part3/LossDoctor',
    chapter: 12,
    howTo: '처방은 위에서부터 순서대로! "진단 퀴즈"로 내 진단 실력도 확인해 보세요.',
    aspect: 1.35
  }
];
