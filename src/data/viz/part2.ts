import type { VizMeta } from './types';

export const viz: VizMeta[] = [
  {
    id: 'sigmoid-squash',
    title: '시그모이드 구겨 넣기',
    tagline: '아무 숫자나 넣어도 0과 1 사이 "확률"로 쏙 — S자 기계를 직접 돌려 봐요.',
    icon: '🎚️',
    component: 'part2/SigmoidSquash',
    chapter: 5,
    howTo: '"42 넣어 보기"를 눌러 보세요. 엄청 큰 점수도 확률 1을 넘지 못해요. 두 번째 탭에서는 종양 데이터에 S자를 맞춰 봐요!',
    aspect: 1.05
  },
  {
    id: 'decision-boundary',
    title: '결정 경계 학습 실험',
    tagline: '로지스틱 회귀가 교차 엔트로피를 줄이며 직선 경계를 스스로 찾아가요.',
    icon: '🧭',
    component: 'part2/DecisionBoundary',
    chapter: 5,
    howTo: '학습이 끝나면 빈 곳을 눌러 "반대편"에 점을 심어 보세요. 그리고 XOR 함정 데이터도 꼭 돌려 보세요!',
    aspect: 0.95
  },
  {
    id: 'softmax-temperature',
    title: '소프트맥스 온도 다이얼',
    tagline: '점수 세 개가 확률 분포로 바뀌는 과정과, 온도에 따라 뾰족해지고 평평해지는 모습을 봐요.',
    icon: '🌡️',
    component: 'part2/SoftmaxTemperature',
    chapter: 5,
    howTo: '온도를 🧊 0.3과 🔥 10으로 바꿔 가며 "20번 뽑기"를 해 보세요. 결과가 얼마나 다양해지나요?',
    aspect: 0.95
  },
  {
    id: 'decision-tree',
    title: '스무고개 결정 트리',
    tagline: '질문 하나마다 지도가 칼질되고, 나무가 한 층씩 자라요. 너무 깊으면 무슨 일이?',
    icon: '🌳',
    component: 'part2/DecisionTree',
    chapter: 6,
    howTo: '깊이를 1부터 8까지 올리며 시험 정확도를 지켜보세요. 지도의 칸이나 나무의 질문을 누르면 그 규칙이 문장으로 보여요.',
    aspect: 0.9
  },
  {
    id: 'random-forest',
    title: '랜덤 포레스트 투표',
    tagline: '들쭉날쭉한 나무들도 여럿이 투표하면 부드럽고 똑똑한 경계가 돼요.',
    icon: '🌲',
    component: 'part2/RandomForest',
    chapter: 6,
    howTo: '나무 수를 1에서 60까지 천천히 늘려 보세요. 지도를 누르면 그 자리에서 몇 대 몇으로 표가 갈렸는지 보여요.',
    aspect: 0.85
  },
  {
    id: 'kmeans',
    title: 'k-평균 무리 찾기',
    tagline: '① 가까운 중심 편 들기 ② 중심을 평균으로 옮기기 — 두 박자를 반복하면 무리가 보여요.',
    icon: '🎯',
    component: 'part2/KMeans',
    chapter: 7,
    howTo: '"한 걸음"을 눌러 배정과 갱신을 하나씩 보세요. "책 속 1차원 예"를 고르면 본문의 계산이 그대로 재현돼요!',
    aspect: 0.95
  },
  {
    id: 'pca-3d',
    title: '3D 주성분 분석',
    tagline: '3차원 점 구름을 가장 넓게 퍼진 평면에 납작하게 눌러, 정보를 거의 잃지 않고 2차원으로 줄여요.',
    icon: '🫓',
    component: 'part2/PCA3D',
    chapter: 7,
    is3D: true,
    howTo: '"납작하게 누르기"를 누른 뒤 "최선"과 "나쁨" 평면을 번갈아 골라 보세요. 오른쪽 그림자에서 세 무리가 어떻게 달라지나요?',
    aspect: 1.1,
    featured: true
  },
  {
    id: 'confusion-threshold',
    title: '임계값과 혼동 행렬',
    tagline: '희귀병 검진 모델의 판정 기준을 움직이며, 정확도가 왜 거짓말을 하는지 확인해요.',
    icon: '🩺',
    component: 'part2/ConfusionThreshold',
    chapter: 8,
    howTo: '"0.2% (책 예시)"를 고르고 🤡 무조건 정상을 눌러 보세요. 정확도 99.8%인데 병자는 한 명도 못 잡아요!',
    aspect: 0.85
  }
];
