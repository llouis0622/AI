import type { GlossaryTerm } from './types';

/** Terms first introduced in part 8. */
export const terms: GlossaryTerm[] = [
  {
    id: 'mdp',
    term: 'MDP',
    aliases: ['마르코프 결정 과정'],
    short: '상태·행동·전이·보상·할인율로 강화학습 문제를 정확히 적는 틀이에요.',
    emoji: '🗺️',
    chapter: 25
  },
  {
    id: 'discount-factor',
    term: '할인율',
    short: '미래 보상을 얼마나 작게 쳐줄지 정하는 숫자 γ(0~1). 클수록 먼 미래까지 내다봐요.',
    long: 'γ = 0.9라면 한 걸음 뒤의 보상은 0.9배, 두 걸음 뒤는 0.81배로 쳐요. "같은 보상이면 빨리 받는 게 낫다"는 상식을 숫자로 적은 거예요.',
    emoji: '⏳',
    chapter: 25
  },
  {
    id: 'return',
    term: '리턴',
    short: '앞으로 받을 보상을 할인해서 모두 더한 값. 강화학습이 키우려는 목표예요.',
    emoji: '💰',
    chapter: 25
  },
  {
    id: 'q-learning',
    term: 'Q-학습',
    aliases: ['Q-러닝'],
    short: '"이 상태에서 이 행동은 얼마나 좋은가(Q)"를 경험으로 조금씩 고쳐 가는 방법이에요.',
    long: '보상을 받은 칸에서 시작해 좋은 소식이 한 칸씩 거꾸로 퍼져요. 다 배우면 각 칸에서 Q가 가장 큰 행동만 고르면 돼요.',
    emoji: '🤖',
    chapter: 25
  },
  {
    id: 'action-value',
    term: '행동 가치 함수',
    aliases: ['Q 함수'],
    short: '어떤 상태에서 어떤 행동을 했을 때 앞으로 기대되는 리턴. Q로 적어요.',
    emoji: '💎',
    chapter: 25
  },
  {
    id: 'explore-exploit',
    term: '탐색과 활용',
    short: '새 행동을 시험해 볼지(탐색), 아는 최선을 쓸지(활용) 사이의 균형이에요.',
    long: '아는 길만 가면 더 좋은 길을 영영 못 찾고, 모험만 하면 아는 것을 써먹지 못해요. 처음엔 많이 모험하고 점점 줄이는 방법을 많이 써요.',
    emoji: '🎲',
    chapter: 25
  },
  {
    id: 'dqn',
    term: 'DQN',
    short: 'Q 값을 표 대신 신경망으로 배우는 방법. 화면만 보고 아타리 게임을 정복했어요.',
    emoji: '🕹️',
    chapter: 25
  },
  {
    id: 'replay-buffer',
    term: '재생 버퍼',
    aliases: ['경험 재생'],
    short: '지난 경험을 모아 두었다가 무작위로 꺼내 다시 배우는 저장소예요.',
    emoji: '📼',
    chapter: 25
  },
  {
    id: 'target-network',
    term: '목표 네트워크',
    short: '학습 목표가 흔들리지 않도록 가끔씩만 갱신하는 복사본 신경망이에요.',
    emoji: '🎯',
    chapter: 25
  },
  {
    id: 'policy-gradient',
    term: '정책 경사',
    short: '행동할 확률(정책)을 보상이 커지는 방향으로 직접 밀어 주는 학습 방법이에요.',
    long: '결과가 좋았던 행동은 확률을 올리고, 나빴던 행동은 내려요. 평균과 비교한 점수를 쓰면 운에 덜 흔들려요.',
    emoji: '🧭',
    chapter: 26
  },
  {
    id: 'ppo',
    term: 'PPO',
    short: '확률을 한 번에 너무 많이 바꾸지 못하게 잘라서 안정적으로 배우는 정책 경사 방법.',
    long: '새 정책과 옛 정책의 확률 비율이 안전 범위(예: 0.8~1.2)를 벗어나면 더 밀지 않아요. 덕분에 같은 데이터로 여러 번 배워도 무너지지 않아요. ChatGPT의 RLHF에도 쓰였어요.',
    emoji: '🛡️',
    chapter: 26
  },
  {
    id: 'actor-critic',
    term: '액터-크리틱',
    short: '행동을 고르는 배우(정책)와 점수를 매기는 비평가(가치)를 함께 배우는 구조예요.',
    emoji: '🎭',
    chapter: 26
  },
  {
    id: 'state-value',
    term: '상태 가치',
    short: '그 상태에서 시작하면 앞으로 기대되는 리턴. 기준으로 삼아 흔들림을 줄여요.',
    emoji: '📍',
    chapter: 26
  }
];
