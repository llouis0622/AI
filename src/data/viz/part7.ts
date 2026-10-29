import type { VizMeta } from './types';

export const viz: VizMeta[] = [
  {
    id: 'next-token',
    title: '다음 단어 맞히기 기계',
    tagline: '모델이 매긴 "다음 단어의 확률"을 보고, 한 단어씩 문장을 지어 봐요.',
    icon: '🔮',
    component: 'part7/NextToken',
    chapter: 21,
    howTo: '온도를 0.2와 2.5로 바꿔 가며 문장을 써 보세요. (문장 40개로 배운 간이 모델이라 진짜 GPT보다 훨씬 작아요)',
    aspect: 1.05
  },
  {
    id: 'scaling-law',
    title: '스케일링 법칙',
    tagline: '계산을 10배씩 키우면 손실이 어떻게 줄어드는지, 예산은 어떻게 나눠야 하는지 봐요.',
    icon: '📈',
    component: 'part7/ScalingLaw',
    chapter: 21,
    howTo: '"로그 눈금"을 꺼 보면 왜 로그 눈금이 필요한지 보여요. (실제 연구 공식을 단순하게 만든 장난감 법칙이에요)',
    aspect: 1.15
  },
  {
    id: 'rlhf-pipeline',
    title: 'RLHF 길들이기 공장',
    tagline: '사전학습 → 지시 튜닝 → 보상 모델 → 강화학습, 야생마가 조수가 되는 과정을 직접 해 봐요.',
    icon: '🏇',
    component: 'part7/RLHFPipeline',
    chapter: 22,
    howTo: '2단계에서 당신이 사람 평가자가 돼요. 3단계에서 KL 벌점을 낮추면 "채점기 해킹"이 일어나요!',
    aspect: 1.0
  },
  {
    id: 'rag-pipeline',
    title: 'RAG 파이프라인',
    tagline: '질문과 가까운 문서 조각을 찾아 프롬프트에 붙이는 과정을 한 단계씩 따라가요.',
    icon: '🔎',
    component: 'part7/RAGPipeline',
    chapter: 23,
    howTo: '"학교 축제는 언제예요?"를 골라 보세요. 근거가 없으면 모른다고 해요. "RAG 끄고 물어보기"와 비교해 보세요!',
    aspect: 1.0
  },
  {
    id: 'three-dials',
    title: '세 개의 다이얼 퀴즈',
    tagline: '상황마다 프롬프트·RAG·파인튜닝 중 어떤 다이얼을 돌려야 할지 맞혀 봐요.',
    icon: '🎛️',
    component: 'part7/ThreeDials',
    chapter: 23,
    howTo: '정답보다 이유가 중요해요. "지식은 RAG로, 행동과 형식은 파인튜닝으로"를 기억하세요.',
    aspect: 1.15
  },
  {
    id: 'agent-loop',
    title: '에이전트 루프 관찰기',
    tagline: '생각 → 도구 호출 → 결과 → 다시 생각… 에이전트의 반복문을 한 걸음씩 지켜봐요.',
    icon: '🔁',
    component: 'part7/AgentLoop',
    chapter: 24,
    howTo: '"로그 정리"에서 승인을 켜고/끄고 비교해 보세요. (미리 짜 둔 시나리오로 루프의 흐름을 보여 줘요)',
    aspect: 1.05
  }
];
