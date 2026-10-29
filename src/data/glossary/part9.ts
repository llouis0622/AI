import type { GlossaryTerm } from './types';

/** Terms first introduced in part 9. */
export const terms: GlossaryTerm[] = [
  {
    id: 'serving',
    term: '서빙',
    short: '학습한 모델을 요청을 받아 예측을 돌려주는 서비스로 운영하는 일이에요.',
    emoji: '🍽️',
    chapter: 27
  },
  {
    id: 'training-serving-skew',
    term: '학습-서빙 왜곡',
    short: '학습 때와 서비스 때 전처리가 달라 성능이 소리 없이 깎이는 버그예요.',
    long: '예를 들어 학습 때는 0~1로 정규화했는데 서비스에서는 0~255 그대로 넣으면, 오류 없이 엉뚱한 답만 나와요. 전처리 코드를 한 함수로 공유하면 막을 수 있어요.',
    emoji: '🪞',
    chapter: 27
  },
  {
    id: 'dynamic-batching',
    term: '동적 배칭',
    aliases: ['배칭'],
    short: '짧은 시간 동안 들어온 요청을 모아 한 번에 처리하는 기법. 처리량이 크게 늘어요.',
    long: 'GPU는 하나씩보다 한꺼번에 많이 계산할 때 효율이 좋아요. 대신 묶음이 찰 때까지 몇 밀리초를 기다리니 지연은 조금 늘어요.',
    emoji: '📦',
    chapter: 27
  },
  {
    id: 'kv-cache',
    term: 'KV 캐시',
    short: '앞 토큰들의 키·밸류를 저장해 두고 다시 쓰는 메모리. 글 생성이 훨씬 빨라져요.',
    emoji: '🗃️',
    chapter: 27
  },
  {
    id: 'quantization',
    term: '양자화',
    short: '가중치를 더 적은 비트(16비트 → 4비트 등)로 저장해 메모리와 속도를 버는 기법이에요.',
    emoji: '🧮',
    chapter: 27
  },
  {
    id: 'speculative-decoding',
    term: '추측 디코딩',
    short: '작은 모델이 초안을 빨리 쓰고, 큰 모델이 한꺼번에 검사해 속도를 높이는 기법이에요.',
    emoji: '🏃',
    chapter: 27
  },
  {
    id: 'throughput',
    term: '처리량',
    short: '정해진 시간에 처리하는 양(예: 초당 요청 수). 많을수록 같은 서버로 더 많이 처리해요.',
    emoji: '🚚',
    chapter: 27
  },
  {
    id: 'p95',
    term: 'p95',
    aliases: ['p99'],
    short: '요청 100개 중 95개(p99는 99개)가 이 시간 안에 끝난다는 뜻. 느린 쪽을 보는 지표예요.',
    long: '평균은 몇몇 아주 느린 요청을 가려 버려요. 평균이 50ms여도 p99가 2초라면, 100명 중 1명은 2초씩 기다리고 있는 거예요.',
    emoji: '⏱️',
    chapter: 27
  },
  {
    id: 'drift',
    term: '드리프트',
    short: '학습 때의 데이터와 실제로 들어오는 데이터가 점점 달라지는 현상. 모델이 늙어요.',
    emoji: '🌊',
    chapter: 27
  },
  {
    id: 'data-drift',
    term: '데이터 드리프트',
    short: '입력의 분포가 바뀌는 것. 정답 없이 입력 통계만 봐도 알아챌 수 있어요.',
    emoji: '👥',
    chapter: 27
  },
  {
    id: 'concept-drift',
    term: '컨셉 드리프트',
    short: '입력과 정답의 관계가 바뀌는 것. 늦게 오는 정답으로 성능을 재야 알 수 있어요.',
    emoji: '🔀',
    chapter: 27
  },
  {
    id: 'delayed-label',
    term: '지연 레이블',
    short: '예측하고 한참 뒤에야 도착하는 정답. 예: 대출을 갚았는지는 몇 달 뒤에 알 수 있어요.',
    emoji: '📬',
    chapter: 27
  },
  {
    id: 'shadow-deployment',
    term: '섀도 배포',
    short: '실제 트래픽을 새 모델에도 복제해 보내되, 응답은 쓰지 않고 관찰만 하는 배포예요.',
    emoji: '👤',
    chapter: 27
  },
  {
    id: 'canary-deployment',
    term: '카나리 배포',
    short: '새 모델에 트래픽을 1~5%만 보내 지켜보다가, 괜찮으면 조금씩 늘리는 배포 방법이에요.',
    long: '옛날 광부들이 유독가스를 먼저 알아채려고 카나리아 새를 데려간 데서 온 이름이에요. 문제가 생겨도 소수만 영향을 받아요.',
    emoji: '🐤',
    chapter: 27
  },
  {
    id: 'ab-test',
    term: 'A/B 테스트',
    short: '사용자를 무작위로 두 무리로 나눠, 어느 쪽이 실제로 더 나은지 통계로 비교하는 실험이에요.',
    emoji: '🆎',
    chapter: 27
  }
];
