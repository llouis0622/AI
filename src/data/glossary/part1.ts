import type { GlossaryTerm } from './types';

/** Terms first introduced in part 1. */
export const terms: GlossaryTerm[] = [
  {
    id: 'model',
    term: '모델',
    short: '데이터에서 찾아낸 규칙 뭉치. 입력을 넣으면 답(예측)을 내놓는 계산식이에요.',
    long: '학습이 끝나면 남는 결과물이에요. 집값 예제에서는 "가격 = w × 크기 + b"라는 직선 하나가 모델이고, GPT도 거대한 계산식 하나라는 점에서 똑같아요.',
    emoji: '🧩',
    chapter: 1
  },
  {
    id: 'parameter',
    term: '파라미터',
    aliases: ['매개변수'],
    short: '모델 안에서 학습이 골라야 하는 숫자들. 직선의 기울기 w와 절편 b 같은 것이에요.',
    long: '파라미터가 바뀌면 모델의 답도 바뀌어요. 학습이란 결국 "좋은 파라미터 값 찾기"예요. 직선은 2개, GPT는 수천억 개를 가지고 있어요.',
    emoji: '🎛️',
    chapter: 2
  },
  {
    id: 'loss-function',
    term: '손실 함수',
    aliases: ['손실'],
    short: '모델이 얼마나 틀렸는지를 하나의 숫자로 채점하는 함수. 작을수록 좋아요.',
    long: '"잘 맞는다"는 애매한 말을 숫자로 바꿔 주는 채점표예요. 이 숫자를 줄이는 방향으로 파라미터를 고치는 것이 학습이에요.',
    emoji: '📉',
    chapter: 2
  },
  {
    id: 'mse',
    term: '평균제곱오차',
    aliases: ['MSE'],
    short: '예측과 정답의 차이를 제곱해서 평균 낸 값. 가장 기본적인 손실 함수예요.',
    long: '제곱을 하기 때문에 +오차와 -오차가 서로 지워지지 않고, 크게 틀린 것에 훨씬 큰 벌점을 줘요.',
    emoji: '🟥',
    chapter: 2
  },
  {
    id: 'gradient-descent',
    term: '경사하강법',
    short: '오차 지형에서 발밑의 가장 가파른 내리막으로 한 걸음씩 내려가며 답을 찾는 방법.',
    long: '기울기(경사)의 반대 방향으로 학습률만큼 파라미터를 고치는 일을 반복해요. GPT를 훈련시킨 방법도 본질적으로 이것이에요.',
    emoji: '⛰️',
    chapter: 2
  },
  {
    id: 'learning-rate',
    term: '학습률',
    short: '경사하강법에서 한 번에 내딛는 보폭의 크기. 너무 크면 튀어 오르고, 너무 작으면 느려요.',
    long: '사람이 정해 주는 다이얼(하이퍼파라미터)이에요. 학습이 폭발(발산)하면 가장 먼저 의심해야 할 범인이랍니다.',
    emoji: '👣',
    chapter: 2
  },
  {
    id: 'hyperparameter',
    term: '하이퍼파라미터',
    short: '학습이 스스로 정하지 않고 사람이 미리 정해 주는 설정값. 학습률이 대표적이에요.',
    emoji: '🔧',
    chapter: 2
  },
  {
    id: 'derivative',
    term: '미분',
    short: '"이걸 아주 조금 바꾸면 결과가 얼마나 변하나?"를 재는 도구. 민감도라고 생각하면 돼요.',
    long: '값이 양수면 키울수록 결과가 커지고, 음수면 작아져요. 학습에서는 손실을 줄이려면 파라미터를 어느 쪽으로 움직여야 하는지 알려줘요.',
    emoji: '🔍',
    chapter: 2
  },
  {
    id: 'divergence',
    term: '발산',
    short: '학습률이 너무 커서 손실이 줄지 않고 점점 커지다 폭발하는 상황.',
    emoji: '🚀',
    chapter: 2
  },
  {
    id: 'convex',
    term: '볼록 함수',
    short: '밥그릇처럼 바닥이 하나뿐인 모양의 함수. 어디서 출발해도 같은 바닥에 도착해요.',
    emoji: '🥣',
    chapter: 2
  },
  {
    id: 'machine-learning',
    term: '머신러닝',
    short: '규칙을 사람이 짜 주는 대신, 예시와 정답을 보고 기계가 규칙을 스스로 찾게 하는 방법.',
    long: '"데이터 + 답 → 규칙"으로 발상을 뒤집은 것이에요. 스팸 필터, 추천, 번역처럼 규칙을 다 적기 어려운 일에서 빛나요.',
    emoji: '🤖',
    chapter: 1
  },
  {
    id: 'deep-learning',
    term: '딥러닝',
    short: '신경망을 여러 층 깊게 쌓아서 배우는 머신러닝. GPT도 딥러닝이에요.',
    long: 'AI ⊃ 머신러닝 ⊃ 딥러닝 순서로 안에 들어 있어요. 사진·소리·글처럼 복잡한 데이터에서 특히 강해요.',
    emoji: '🧠',
    chapter: 1
  },
  {
    id: 'supervised-learning',
    term: '지도학습',
    short: '문제와 정답이 짝지어진 데이터로 배우는 방식. "이 메일은 스팸"처럼 답을 알려 줘요.',
    emoji: '👩‍🏫',
    chapter: 1
  },
  {
    id: 'unsupervised-learning',
    term: '비지도학습',
    short: '정답 없이 데이터만 보고 비슷한 것끼리 묶거나 숨은 구조를 찾는 방식.',
    long: '고객을 몇 부류로 나누는 군집화, 데이터를 압축하는 차원 축소가 대표적이에요(7장).',
    emoji: '🧩',
    chapter: 1
  },
  {
    id: 'reinforcement-learning',
    term: '강화학습',
    short: '정답 대신 "잘했어/못했어" 보상 점수만 받으며 시행착오로 배우는 방식.',
    long: '게임 AI와 로봇이 이렇게 배워요. ChatGPT를 사람 취향에 맞게 다듬을 때도 쓰여요(22장, 25장).',
    emoji: '🎮',
    chapter: 1
  },
  {
    id: 'self-supervised-learning',
    term: '자기지도학습',
    short: '데이터 일부를 가리고 그걸 맞히게 해서, 정답표 없이도 스스로 문제를 만들어 배우는 방식.',
    long: '"다음 단어 맞히기"가 대표적인 예이고, GPT가 바로 이렇게 배워요. 사람이 정답을 달 필요가 없어 인터넷 전체가 교재가 돼요.',
    emoji: '🙈',
    chapter: 1
  },
  {
    id: 'vector',
    term: '벡터',
    short: '숫자 여러 개를 한 줄로 묶은 것. 화살표(방향과 길이)로 그려 볼 수 있어요.',
    long: '집 하나를 [크기, 방 수, 역까지 거리]처럼 숫자 묶음으로 나타내면 벡터가 돼요. AI는 글자·그림·소리도 전부 벡터로 바꿔서 다뤄요.',
    emoji: '➡️',
    chapter: 3
  },
  {
    id: 'dot-product',
    term: '내적',
    short: '두 벡터의 같은 자리 숫자끼리 곱해서 모두 더한 값. 두 화살표가 얼마나 같은 쪽을 보는지 알려 줘요.',
    long: '같은 방향이면 크고, 직각이면 0, 반대 방향이면 음수예요. 길이로 나눠 주면 "코사인 유사도"라는 닮음 점수가 돼요.',
    emoji: '🤝',
    chapter: 3
  },
  {
    id: 'chain-rule',
    term: '연쇄법칙',
    aliases: ['연쇄 법칙'],
    short: '단계가 이어진 계산에서, 각 단계의 변화율을 곱하면 전체 변화율이 된다는 미분 규칙.',
    long: '톱니바퀴가 맞물려 돌 때 회전 비율이 곱해지는 것과 같아요. 신경망을 학습시키는 역전파가 바로 이 규칙이에요(10장).',
    emoji: '⚙️',
    chapter: 3
  },
  {
    id: 'mle',
    term: '최대가능도',
    aliases: ['최대 가능도', 'MLE'],
    short: '관찰한 데이터가 나올 확률을 가장 크게 만드는 값을 고르는 원리.',
    long: '동전이 10번 중 7번 앞면이었다면 "앞면 확률 0.7"이 가장 그럴듯해요. 손실 함수는 대부분 이 원리에서 나와요.',
    emoji: '🪙',
    chapter: 3
  },
  {
    id: 'generalization',
    term: '일반화',
    short: '배운 적 없는 새 데이터에서도 잘 맞히는 능력. 머신러닝의 진짜 목표예요.',
    emoji: '🌍',
    chapter: 4
  },
  {
    id: 'overfitting',
    term: '과적합',
    short: '훈련 데이터의 잡음까지 외워 버려서, 새 데이터에서는 오히려 틀리는 상태.',
    long: '족보만 외운 학생과 같아요. 반대로 너무 단순해서 패턴조차 못 잡으면 "과소적합"이라고 해요.',
    emoji: '📝',
    chapter: 4
  },
  {
    id: 'cross-validation',
    term: '교차검증',
    short: '데이터를 여러 조각으로 나눠, 조각마다 한 번씩 시험지로 쓰고 평균 내는 평가 방법.',
    emoji: '🔁',
    chapter: 4
  },
  {
    id: 'data-leakage',
    term: '데이터 누수',
    short: '시험 문제의 정보가 공부 자료에 몰래 섞여 들어가, 성적이 가짜로 좋아지는 사고.',
    long: '전체 데이터로 전처리를 먼저 하거나, 미래 데이터로 과거를 맞히게 나누면 생겨요. 배포하고 나서야 들통나는 비싼 실수예요.',
    emoji: '💧',
    chapter: 4
  },
  {
    id: 'regularization',
    term: '정규화',
    short: '두 가지 뜻으로 써요. ① 모델이 너무 복잡해지지 않게 벌점을 주는 기법 ② 숫자 크기를 고르게 맞추는 일.',
    long: '①은 과적합을 막는 무기(regularization, 예: L2 벌점)이고, ②는 전처리나 신경망 층에서 값의 범위를 맞추는 일(normalization)이에요. 문맥으로 구별해요.',
    emoji: '⚖️',
    chapter: 4
  }
];
