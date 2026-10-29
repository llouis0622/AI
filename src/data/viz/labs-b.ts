import type { VizMeta } from './types';

/** Code lab visualizations (labs 11–20). */
export const viz: VizMeta[] = [
  {
    id: 'lab11-lora',
    title: 'LoRA 계산기',
    tagline: '거대한 W는 얼려 두고 가느다란 A·B만 학습할 때, 파라미터와 GPU 메모리가 얼마나 줄어드는지 직접 바꿔 봐요.',
    icon: '🧊',
    component: 'labs/Lab11Lora',
    lab: 11,
    howTo: '랭크 r을 4 → 64로 바꾸며 가운데 병목과 "학습하는 비율"을 비교하고, 마지막에 "병합"을 눌러 옆길이 사라지는 걸 보세요.',
    aspect: 1.2
  },
  {
    id: 'lab12-rag',
    title: 'RAG 파이프라인 한 바퀴',
    tagline: '조각내고, 좌표로 바꾸고, 찾아서, 근거와 함께 답하는 과정을 단계별로 봐요.',
    icon: '🔎',
    component: 'labs/Lab12Rag',
    lab: 12,
    howTo: '"다음 →"을 누를 때마다 오른쪽 코드의 어느 줄이 그림의 어느 부분인지 함께 확인해 보세요.',
    aspect: 1.1
  },
  {
    id: 'lab13-agent-trace',
    title: '에이전트 실행 기록',
    tagline: '도구를 고르고, 실수하고, 스스로 고쳐서 답하는 한 번의 실행을 turn 단위로 따라가요.',
    icon: '🧭',
    component: 'labs/Lab13AgentTrace',
    lab: 13,
    howTo: '왼쪽 고리의 불빛과 오른쪽 메시지 기록이 함께 움직여요. 4단계의 "실수 → 스스로 복구"를 꼭 보세요.',
    aspect: 1.1
  },
  {
    id: 'lab14-vae-gan',
    title: 'VAE와 GAN 코드 해부',
    tagline: '분포(구름)를 배우는 VAE와 위조꾼·감별사가 겨루는 GAN을 코드 줄 단위로 비교해요.',
    icon: '🎭',
    component: 'labs/Lab14VaeGan',
    lab: 14,
    howTo: '2단계의 재파라미터화와 6단계의 detach()가 이 코드의 두 급소예요. 그림에서 기울기가 어디로 흐르는지 보세요.',
    aspect: 1.1
  },
  {
    id: 'lab15-ddpm',
    title: 'DDPM 한 바퀴',
    tagline: '노이즈 스케줄부터 "노이즈 맞히기" 학습, 1000걸음 샘플링까지 실제 숫자로 따라가요.',
    icon: '🌫️',
    component: 'labs/Lab15Ddpm',
    lab: 15,
    howTo: '1단계 그래프에서 t=500일 때 원래 그림이 몇 % 남는지 확인하고, 2단계에서 직접 눈으로 비교해 보세요.',
    aspect: 1.1
  },
  {
    id: 'lab16-dqn-cartpole',
    title: 'DQN 막대 세우기 실험',
    tagline: '브라우저 안에서 진짜 DQN이 CartPole을 학습해요. 리플레이 버퍼와 타깃 네트워크가 움직이는 모습을 보면서요.',
    icon: '🤸',
    component: 'labs/Lab16DqnCartpole',
    lab: 16,
    howTo: '▶ 학습 시작을 누르고(×20 속도면 약 15초) 굵은 선이 475를 넘는지 보세요. 해결되면 시연이 자동으로 시작돼요. 그다음 타깃 동기화 주기를 1로 바꿔 다시 학습하면 두 곡선을 한 그래프에서 비교할 수 있어요.',
    aspect: 0.75,
    featured: true
  },
  {
    id: 'lab17-ppo',
    title: 'PPO 업데이트 해부',
    tagline: '롤아웃 → GAE → 정규화 → 클리핑 → 10 에포크 재사용까지, PPO 한 번의 업데이트를 단계별로 봐요.',
    icon: '🎯',
    component: 'labs/Lab17Ppo',
    lab: 17,
    howTo: '3단계의 GAE는 오른쪽에서 왼쪽으로 계산돼요. 5단계에서 점이 0.8~1.2 구간을 벗어나면 그래프가 평평해지는 걸 보세요.',
    aspect: 1.1
  },
  {
    id: 'lab18-clip',
    title: 'CLIP 한눈에 보기',
    tagline: '사진과 글이 한 공간에 놓이는 원리, 제로샷 분류, 이미지 검색, 대조 손실까지 그림으로 봐요.',
    icon: '🖼️',
    component: 'labs/Lab18Clip',
    lab: 18,
    howTo: '3단계에서 유사도 차이는 작은데 확률은 확 갈리는 이유(×100)를 찾아보세요.',
    aspect: 1.1
  },
  {
    id: 'lab19-whisper',
    title: '소리가 이미지가 되기까지',
    tagline: '파형을 잘라 FFT하고, 멜로 묶고, 로그를 씌워 "소리 이미지"를 만드는 과정을 실제 계산으로 봐요.',
    icon: '🎧',
    component: 'labs/Lab19Whisper',
    lab: 19,
    howTo: '3~5단계의 그림은 브라우저에서 직접 계산한 진짜 스펙트로그램이에요. "아 · 이 · 스"가 어떻게 다르게 보이는지 비교해 보세요.',
    aspect: 1.1
  },
  {
    id: 'lab20-serving',
    title: '모델 서빙의 모든 것',
    tagline: '한 번 로드, 요청의 일생, 헬스체크, 동적 배칭, Docker 캐시까지 서빙 코드를 그림으로 따라가요.',
    icon: '🚀',
    component: 'labs/Lab20Serving',
    lab: 20,
    howTo: '5단계에서 위(배칭)와 아래(한 건씩)의 GPU 막대 개수를 세어 보세요. 그 차이가 처리량의 차이예요.',
    aspect: 1.1
  }
];
