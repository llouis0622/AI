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
  },
  {
    id: 'learning-rate-1d',
    title: '학습률 골짜기 점프',
    tagline: '보폭(학습률)에 따라 공이 느리게, 지그재그로, 혹은 폭발하듯 움직여요.',
    icon: '🦘',
    component: 'part1/LearningRate1D',
    chapter: 2,
    howTo: '네 가지 학습률을 차례로 골라 ▶ 시작을 눌러 보세요. 몇 걸음 만에 바닥에 닿는지 비교해요.',
    aspect: 0.95
  },
  {
    id: 'rules-vs-learning',
    title: '규칙 vs 학습 스팸 대결',
    tagline: '규칙을 직접 짜 볼까, 예시로 가르쳐 볼까? 스팸 메일로 겨뤄 봐요.',
    icon: '📧',
    component: 'part1/RulesVsLearning',
    chapter: 1,
    howTo: '"😈 스패머가 수법을 바꿨다!"를 누르고, 두 방법이 각각 어떻게 대처하는지 비교해 보세요.',
    aspect: 0.8
  },
  {
    id: 'ai-ml-dl-nest',
    title: '인공지능 동그라미 지도',
    tagline: '인공지능 ⊃ 머신러닝 ⊃ 딥러닝 ⊃ GPT — 누가 누구 안에 들어 있을까요?',
    icon: '🎯',
    component: 'part1/AiMlDlNest',
    chapter: 1,
    howTo: '아래 예시들을 하나씩 눌러 어느 동그라미에 들어가는지 확인해 보세요.',
    aspect: 1
  },
  {
    id: 'learning-types',
    title: '네 가지 배움의 방식',
    tagline: '정답·데이터만·보상·가린 부분 — 무엇을 주느냐에 따라 배움이 달라져요.',
    icon: '🎓',
    component: 'part1/LearningTypes',
    chapter: 1,
    howTo: '탭을 바꿔 가며 장면을 보고, 아래 "맞혀 보기" 네 문제를 풀어 보세요.',
    aspect: 0.85
  },
  {
    id: 'learning-pipeline',
    title: '학습 루프 지도',
    tagline: '데이터 → 모델 → 손실 → 기울기 → 갱신, 세 가지 수학이 어디에 꽂히는지 봐요.',
    icon: '🔁',
    component: 'part1/LearningPipeline',
    chapter: 3,
    howTo: '상자를 하나씩 눌러 그 단계에서 쓰는 수학을 확인해 보세요.',
    aspect: 1.3
  },
  {
    id: 'dot-product',
    title: '내적 닮음 측정기',
    tagline: '두 화살표를 끌어 보세요. 같은 쪽을 볼수록 내적이 커져요.',
    icon: '🧭',
    component: 'part1/DotProduct',
    chapter: 3,
    howTo: '예시 버튼(같은 방향·직각·반대)으로 "스스로 점검" 1번의 답을 확인해 보세요.',
    aspect: 0.9
  },
  {
    id: 'chain-rule-gears',
    title: '연쇄법칙 톱니바퀴',
    tagline: '페달 → 체인 → 바퀴, 민감도는 단계마다 곱해져요.',
    icon: '⚙️',
    component: 'part1/ChainRuleGears',
    chapter: 3,
    howTo: '두 번째 탭에서 0.5를 30번 곱하면 얼마나 작아지는지 확인해 보세요.',
    aspect: 1.2
  },
  {
    id: 'likelihood-coin',
    title: '동전으로 최대가능도',
    tagline: '동전 결과를 가장 그럴듯하게 만드는 확률 p를 찾아봐요.',
    icon: '🪙',
    component: 'part1/LikelihoodCoin',
    chapter: 3,
    howTo: '"손실(-log)로 뒤집어 보기"를 켜면 꼭대기가 골짜기로 바뀌어요. 손실 함수가 태어나는 순간!',
    aspect: 1
  },
  {
    id: 'overfitting-poly',
    title: '과적합 곡선 실험',
    tagline: '곡선의 차수를 올리면 공부한 점은 다 맞히는데, 새 점은 왜 틀릴까요?',
    icon: '🎢',
    component: 'part1/OverfittingPoly',
    chapter: 4,
    howTo: '차수를 12까지 올려 본 뒤, 데이터를 40개로 늘리거나 벌점을 켜서 곡선을 진정시켜 보세요.',
    aspect: 0.95
  },
  {
    id: 'data-split',
    title: '데이터 셋으로 나누기',
    tagline: '훈련·검증·테스트로 나누고, 교차검증이 어떻게 도는지 봐요.',
    icon: '🗂️',
    component: 'part1/DataSplit',
    chapter: 4,
    howTo: '"🙈 테스트 점수 보고 모델 고치기"를 눌러 무엇이 잘못인지 확인해 보세요.',
    aspect: 1.1
  }
];
