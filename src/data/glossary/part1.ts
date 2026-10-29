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
    long: '제곱을 하기 때문에 +오차와 −오차가 서로 지워지지 않고, 크게 틀린 것에 훨씬 큰 벌점을 줘요.',
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
  }
];
