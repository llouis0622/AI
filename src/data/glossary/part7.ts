import type { GlossaryTerm } from './types';

/** Terms first introduced in part 7. */
export const terms: GlossaryTerm[] = [
  {
    id: 'pretraining',
    term: '사전학습',
    short: '본 과제 전에 엄청난 양의 데이터로 먼저 넓게 배워 두는 단계. LLM은 "다음 단어 맞히기"로 해요.',
    long: '정답표를 따로 만들 필요 없이, 글 자체가 문제와 정답이 돼요. 이렇게 쌓은 능력을 바탕으로 나중에 원하는 일에 맞게 조금만 더 가르쳐요.',
    emoji: '📚',
    chapter: 21
  },
  {
    id: 'scaling-law',
    term: '스케일링 법칙',
    short: '모델·데이터·계산량을 키울수록 손실이 일정한 비율로 줄어드는 경험 법칙이에요.',
    long: '로그 눈금 그래프에서 거의 곧은 직선으로 보여요. 그래서 큰 모델을 만들기 전에 작은 실험으로 성능을 미리 내다볼 수 있어요.',
    emoji: '📈',
    chapter: 21
  },
  {
    id: 'in-context-learning',
    term: '문맥 내 학습',
    short: '가중치는 그대로 두고, 프롬프트 속 예시만 보고 새 과제를 해내는 능력이에요.',
    long: '"사과→apple, 고양이→cat, 강아지→?"처럼 예시 몇 개만 보여 줘도 규칙을 알아채요. 모델을 다시 학습시키지 않아도 된다는 게 핵심이에요.',
    emoji: '🪄',
    chapter: 21
  },
  {
    id: 'hallucination',
    term: '환각',
    short: '모델이 그럴듯하지만 사실이 아닌 내용을 자신 있게 지어내는 현상이에요.',
    long: '모델은 "사실"이 아니라 "그럴듯한 다음 단어"를 배웠기 때문에 생겨요. 근거 문서를 붙여 주는 RAG, 출처 확인 같은 방법으로 줄여요.',
    emoji: '🌀',
    chapter: 21
  },
  {
    id: 'knowledge-cutoff',
    term: '지식 마감일',
    aliases: ['지식의 마감일'],
    short: '학습 데이터를 모은 마지막 날짜. 그 뒤에 생긴 일은 모델이 몰라요.',
    emoji: '📅',
    chapter: 21
  },
  {
    id: 'prompt',
    term: '프롬프트',
    short: 'AI에게 주는 입력 글(지시·질문·예시). 어떻게 쓰느냐에 따라 답이 크게 달라져요.',
    emoji: '💬',
    chapter: 21
  },
  {
    id: 'rlhf',
    term: 'RLHF',
    aliases: ['인간 피드백 강화학습'],
    short: '사람이 고른 "더 나은 답"으로 보상 모델을 만들고, 강화학습으로 모델을 길들이는 방법.',
    long: '모범 답안 따라 하기(SFT) → 사람의 선호로 채점기 만들기(보상 모델) → 채점기 점수가 오르도록 모델 다듬기(PPO)의 세 단계예요.',
    emoji: '🐎',
    chapter: 22
  },
  {
    id: 'sft',
    term: '지도 미세조정',
    aliases: ['SFT'],
    short: '사람이 쓴 모범 답안을 보여 주며 "이렇게 답해"라고 가르치는 단계예요.',
    emoji: '🧑‍🏫',
    chapter: 22
  },
  {
    id: 'reward-model',
    term: '보상 모델',
    short: '두 답 중 사람이 더 좋아할 쪽에 높은 점수를 주도록 학습한 채점 모델이에요.',
    long: '사람이 모든 답을 일일이 채점할 수는 없으니, 사람의 취향을 흉내 내는 채점기를 따로 만들어요. 채점기에 허점이 있으면 모델이 그 허점을 파고들어요(보상 해킹).',
    emoji: '🏅',
    chapter: 22
  },
  {
    id: 'dpo',
    term: 'DPO',
    short: '보상 모델과 강화학습 없이, "이 답이 더 좋다"는 쌍 데이터로 모델을 바로 길들이는 방법.',
    emoji: '✂️',
    chapter: 22
  },
  {
    id: 'kl-penalty',
    term: 'KL 벌점',
    short: '원래 모델에서 너무 멀어지지 않도록 붙이는 벌점. 보상 해킹을 막아 줘요.',
    emoji: '🪢',
    chapter: 22
  },
  {
    id: 'rag',
    term: 'RAG',
    aliases: ['검색 증강 생성', '검색 증강'],
    short: '질문과 관련된 문서를 먼저 찾아 프롬프트에 붙여 주고, 그걸 근거로 답하게 하는 방법.',
    long: '모델이 모르는 최신 정보나 회사 문서도 답할 수 있게 되고, 어디서 나온 내용인지 출처를 댈 수 있어요.',
    emoji: '🔎',
    chapter: 23
  },
  {
    id: 'tool-use',
    term: '도구 사용',
    short: '모델이 계산기·검색 같은 도구를 불러 쓰는 것. 실제 실행은 우리 코드가 대신해요.',
    emoji: '🧰',
    chapter: 24
  },
  {
    id: 'agent',
    term: '에이전트',
    short: '스스로 판단하며 도구를 쓰고 행동해서 목표를 이루는 AI. 강화학습에선 행동하는 주체예요.',
    long: 'LLM 에이전트는 "생각 → 도구 사용 → 결과 확인"을 반복하는 반복문 하나예요. 실제로 행동하기 때문에 최소 권한, 샌드박스, 사람의 승인 같은 안전장치가 꼭 필요해요.',
    emoji: '🤖',
    chapter: 24
  },
  {
    id: 'sandbox',
    term: '샌드박스',
    short: '밖에 피해를 주지 못하도록 격리된 실행 공간. 에이전트가 실수해도 안전해요.',
    emoji: '🏖️',
    chapter: 24
  }
];
