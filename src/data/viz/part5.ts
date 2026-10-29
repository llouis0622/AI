import type { VizMeta } from './types';

export const viz: VizMeta[] = [
  {
    id: 'tokenizer-bpe',
    title: 'BPE 토큰 공장',
    tagline: '가장 자주 붙어 나오는 글자 쌍을 하나로 합치며, 기계가 낱말 조각을 배우는 과정을 봐요.',
    icon: '✂️',
    component: 'part5/TokenizerBPE',
    chapter: 16,
    howTo: '「한 번 병합」을 여러 번 눌러 「학교」가 한 덩어리가 되는 순간을 찾아보세요. 그다음 2번 탭에서 내 문장을 잘라 봐요!',
    aspect: 1.05
  },
  {
    id: 'embedding-space-3d',
    title: '3D 낱말 지도',
    tagline: '뜻이 비슷한 낱말은 가까이, 관계는 같은 방향의 화살표로 — 낱말의 기하학을 돌려 봐요.',
    icon: '🌌',
    component: 'part5/EmbeddingSpace3D',
    chapter: 16,
    is3D: true,
    howTo: '「왕-남자+여자」를 계산해 보세요. 실제 임베딩은 수백 차원이지만, 여기선 3차원으로 만든 예시 공간이에요.',
    aspect: 1.15,
    featured: true
  },
  {
    id: 'rnn-bottleneck',
    title: 'RNN의 기억 상자',
    tagline: '한 낱말씩 읽으며 작은 상자에 요약하다 보면, 먼 낱말은 어떻게 될까요?',
    icon: '📦',
    component: 'part5/RnnBottleneck',
    chapter: 17,
    howTo: '기억이 남는 비율을 낮게 두고 끝까지 읽어 보세요. 그다음 어텐션 탭과 비교해 봐요. (예측은 개념을 보여 주는 간이 모델이에요)',
    aspect: 1.2
  },
  {
    id: 'attention-heatmap',
    title: '어텐션 돋보기',
    tagline: '낱말을 누르면 누구를 얼마나 바라보는지, 그리고 새 벡터가 어떻게 섞이는지 보여요.',
    icon: '🔦',
    component: 'part5/AttentionHeatmap',
    chapter: 17,
    howTo: '「강가 은행」과 「돈 찾는 은행」에서 「은행에」를 눌러 비교해 보세요. 같은 낱말인데 뜻이 바뀌어요!',
    aspect: 0.95
  },
  {
    id: 'transformer-block-2d',
    title: 'Transformer 블록 해부도',
    tagline: '토큰 벡터 네 개가 블록 하나를 통과하는 길을 한 단계씩 따라가요.',
    icon: '🏗️',
    component: 'part5/TransformerBlock2D',
    chapter: 18,
    howTo: '「다음 단계」로 따라가다가 그림 속 부품을 눌러 보세요. 어텐션에선 색이 섞이고, 피드포워드에선 섞이지 않아요.',
    aspect: 1.0
  },
  {
    id: 'transformer-3d',
    title: 'Transformer 3D 해부도',
    tagline: '단어 상자 4개가 블록을 층층이 올라가며 서로 대화하고(어텐션) 각자 생각을 정리해요(피드포워드).',
    icon: '🏗️',
    component: 'part5/Transformer3D',
    chapter: 18,
    is3D: true,
    howTo: '▶ 시작으로 한 층씩 올라가 보세요. 판을 누르면 그 단계로 바로 가요. GPT와 BERT를 바꾸면 어텐션 선의 방향이 달라져요.',
    aspect: 0.9
  },
  {
    id: 'mask-variants',
    title: '가림막 세 가지',
    tagline: '같은 Transformer라도 누가 누구를 볼 수 있게 하느냐에 따라 BERT, GPT, 번역기가 돼요.',
    icon: '🎭',
    component: 'part5/MaskVariants',
    chapter: 18,
    howTo: 'GPT에서 「생성: 한 토큰씩」을 골라 문장을 한 토큰씩 만들어 보세요. 🔒은 볼 수 없는 미래예요.',
    aspect: 1.15
  }
];
