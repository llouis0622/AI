import type { GlossaryTerm } from './types';

/** Terms first introduced in part 2. */
export const terms: GlossaryTerm[] = [
  {
    id: 'sigmoid',
    term: '시그모이드',
    short: '어떤 숫자든 0과 1 사이로 부드럽게 구겨 넣는 S자 함수. 점수를 확률로 바꿔 줘요.',
    long: '큰 음수는 0에, 0은 정확히 0.5에, 큰 양수는 1에 가까워져요. 식은 σ(z) = 1 / (1 + e^(-z))예요.',
    emoji: '🎚️',
    chapter: 5
  },
  {
    id: 'logistic-regression',
    term: '로지스틱 회귀',
    short: '직선 점수(wx + b)를 시그모이드에 넣어 "예일 확률"을 내놓는 분류 모델.',
    long: '이름에 "회귀"가 붙었지만 분류 모델이에요. 가중치를 보면 어떤 특징이 판정을 밀어 올리는지 읽을 수 있어요.',
    emoji: '📈',
    chapter: 5
  },
  {
    id: 'decision-boundary',
    term: '결정 경계',
    short: '모델이 "이쪽은 A, 저쪽은 B"라고 판정을 바꾸는 경계선. 확률 0.5인 곳이에요.',
    long: '로지스틱 회귀의 결정 경계는 언제나 직선(고차원에서는 평면)이에요. 그래서 XOR처럼 꼬인 데이터는 못 갈라요.',
    emoji: '🧭',
    chapter: 5
  },
  {
    id: 'cross-entropy',
    term: '교차 엔트로피',
    short: '정답에 준 확률의 -log. 정답을 낮은 확률로 찍을수록 벌점이 커지는 분류용 손실 함수.',
    long: '정답에 0.9를 줬으면 벌점 0.1, 0.01을 줬으면 4.6이에요. "확신에 찬 오답"을 가장 무겁게 벌해요. 최대가능도에서 바로 나와요.',
    emoji: '😱',
    chapter: 5
  },
  {
    id: 'softmax',
    term: '소프트맥스',
    short: '점수 여러 개를 합이 1인 확률들로 바꾸는 함수. 점수가 클수록 확률을 많이 가져가요.',
    long: '각 점수의 e^z를 전체 합으로 나눠요. 점수를 온도 T로 나누면 분포가 뾰족해지거나 평평해져요. GPT의 다음 단어 고르기도 이걸 써요.',
    emoji: '🍰',
    chapter: 5
  },
  {
    id: 'knn',
    term: 'k-최근접 이웃',
    aliases: ['kNN'],
    short: '새 데이터와 가장 가까운 이웃 k개를 찾아 다수결로 답하는 방법. 따로 학습하지 않아요.',
    long: '"가깝다"를 어떻게 재느냐가 성능을 좌우해요. 차원이 아주 높으면 모든 점이 비슷하게 멀어지는 "차원의 저주"에 빠져요.',
    emoji: '🏘️',
    chapter: 5
  },
  {
    id: 'decision-tree',
    term: '결정 트리',
    short: '"크기가 5보다 큰가?" 같은 예/아니오 질문을 이어 가며 답을 찾는 스무고개 모델.',
    long: '갈림길마다 데이터를 가장 깔끔하게 나누는 질문을 골라요. 사람이 규칙을 읽을 수 있지만, 혼자 깊게 자라면 과적합해요.',
    emoji: '🌳',
    chapter: 6
  },
  {
    id: 'impurity',
    term: '불순도',
    aliases: ['지니 불순도'],
    short: '한 무리 안에 여러 종류가 얼마나 뒤섞였는지 재는 점수. 한 종류뿐이면 0이에요.',
    emoji: '🧪',
    chapter: 6
  },
  {
    id: 'random-forest',
    term: '랜덤 포레스트',
    short: '조금씩 다른 데이터로 키운 결정 트리 여러 그루가 투표해서 답하는 앙상블 모델.',
    long: '나무마다 무작위로 뽑은 표본과 특징으로 자라서 서로 다르게 틀려요. 투표하면 실수가 서로 지워져 훨씬 안정적이에요.',
    emoji: '🌲',
    chapter: 6
  },
  {
    id: 'boosting',
    term: '부스팅',
    aliases: ['그래디언트 부스팅'],
    short: '얕은 나무를 하나씩 더하면서, 앞 나무들이 틀린 만큼을 다음 나무가 메워 가는 앙상블.',
    long: 'XGBoost, LightGBM 같은 그래디언트 부스팅이 표(테이블) 데이터에서 가장 강한 방법으로 꼽혀요.',
    emoji: '🚀',
    chapter: 6
  },
  {
    id: 'k-means',
    term: 'k-평균',
    aliases: ['k-means'],
    short: '중심 k개를 두고 ① 가까운 중심에 배정 ② 중심을 평균으로 이동을 반복해 무리를 찾는 방법.',
    long: '무리 수 k는 사람이 정해요. 둥근 무리를 잘 찾지만 초승달 같은 모양은 잘 못 자르고, 거리를 재기 전에 표준화가 필요해요.',
    emoji: '🎯',
    chapter: 7
  },
  {
    id: 'pca',
    term: '주성분 분석',
    aliases: ['PCA'],
    short: '데이터가 가장 넓게 퍼진 방향들만 남겨, 정보를 적게 잃으면서 차원을 줄이는 방법.',
    long: '퍼짐(분산)이 곧 점들을 구별해 주는 정보예요. "축 2개로 분산의 90% 보존"처럼 얼마나 남겼는지 성적표가 함께 나와요.',
    emoji: '🫓',
    chapter: 7
  },
  {
    id: 'confusion-matrix',
    term: '혼동 행렬',
    short: '예측과 실제를 맞힘·놓침·괜한 걱정·정상 확인의 네 칸으로 쪼갠 성적표.',
    long: '진양성(TP), 위음성(FN), 위양성(FP), 진음성(TN) 네 칸이에요. 정확도 하나로는 안 보이는 실수의 종류가 드러나요.',
    emoji: '🔲',
    chapter: 8
  },
  {
    id: 'precision',
    term: '정밀도(precision)',
    short: '"그렇다"고 판정한 것 중 진짜로 맞은 비율. 판정을 얼마나 믿을 수 있나를 알려 줘요.',
    long: 'TP ÷ (TP + FP). 정상 메일을 막으면 안 되는 스팸 필터처럼, 괜한 오탐이 비싼 일에서 중요해요.',
    emoji: '🎯',
    chapter: 8
  },
  {
    id: 'recall',
    term: '재현율',
    short: '진짜 정답(예: 실제 환자) 중에서 놓치지 않고 잡아낸 비율.',
    long: 'TP ÷ (TP + FN). 놓치면 큰일 나는 암 검진에서 중요해요. 정밀도와는 서로 당기는 줄다리기 관계예요.',
    emoji: '🥅',
    chapter: 8
  }
];
