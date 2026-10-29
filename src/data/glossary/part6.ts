import type { GlossaryTerm } from './types';

/** Terms first introduced in part 6. */
export const terms: GlossaryTerm[] = [
  {
    id: 'generative-model',
    term: '생성 모델',
    short: '데이터가 어떻게 생겼는지(분포)를 배워서, 그럴듯한 새 데이터를 만들어 내는 모델.',
    long: '"이 사진은 고양이"처럼 구분하는 모델과 달리, 본 적 없는 새 얼굴이나 문장을 지어내요. VAE, GAN, 확산 모델, GPT가 모두 생성 모델이에요.',
    emoji: '🎨',
    chapter: 19
  },
  {
    id: 'latent-vector',
    term: '잠재 벡터',
    aliases: ['잠재 공간', '잠재 표현'],
    short: '데이터의 핵심만 담은 작은 숫자 묶음 z. 이걸 바꾸면 만들어지는 그림이 바뀌어요.',
    long: '보통 다루기 쉬운 정규분포에서 뽑아요. 잘 정돈된 잠재 공간에서는 가까운 점끼리 비슷한 그림이 나와요.',
    emoji: '🧭',
    chapter: 19
  },
  {
    id: 'autoencoder',
    term: '오토인코더',
    short: '입력을 작게 압축(인코더)했다가 다시 복원(디코더)하도록 배우는 신경망.',
    long: '복원을 잘하려면 압축본에 본질이 담겨야 해요. 하지만 학습 사진이 없던 빈 곳에서는 엉뚱한 그림이 나와서, 그대로는 생성기가 되기 어려워요.',
    emoji: '🗜️',
    chapter: 19
  },
  {
    id: 'vae',
    term: '변분 오토인코더',
    aliases: ['VAE'],
    short: '점 대신 작은 구름(분포)으로 압축하고, 구름들을 정규분포 근처로 모으는 오토인코더.',
    long: '손실 = 복원 오차 + 분포 벌점(KL)이에요. 잠재 공간이 빈틈없이 정돈되어 보간이 매끄럽지만, 결과가 조금 흐릿한 편이에요.',
    emoji: '☁️',
    chapter: 19
  },
  {
    id: 'gan',
    term: '적대적 생성망',
    aliases: ['GAN'],
    short: '가짜를 만드는 생성기와 가짜를 가려내는 판별기가 서로 겨루며 함께 실력을 키우는 방법.',
    long: '위조지폐범과 감정사의 대결이에요. 결과가 날카롭고 사실적이지만, 두 선수의 균형을 맞추기 어려워 훈련이 불안정해요.',
    emoji: '🥊',
    chapter: 19
  },
  {
    id: 'generator',
    term: '생성기',
    short: '무작위 숫자 z를 받아 그럴듯한 데이터(그림 등)로 바꿔 주는 신경망.',
    emoji: '🖌️',
    chapter: 19
  },
  {
    id: 'discriminator',
    term: '판별기',
    aliases: ['감별사'],
    short: 'GAN에서 진짜 데이터와 생성기가 만든 가짜를 구분하도록 배우는 이진 분류기.',
    long: '판별기가 너무 강해지면 생성기가 "다 틀렸어"라는 말만 듣게 되어 배울 신호를 잃어요.',
    emoji: '🔍',
    chapter: 19
  },
  {
    id: 'mode-collapse',
    term: '모드 붕괴',
    short: '생성기가 판별기를 속이기 쉬운 한 종류만 계속 만들어 다양성이 사라지는 GAN의 실패.',
    long: '숫자를 그리랬더니 "1"만 그리는 식이에요. 판별기는 속였지만, 데이터의 다른 모습(모드)들은 영영 만들지 못해요.',
    emoji: '🔁',
    chapter: 19
  },
  {
    id: 'interpolation',
    term: '보간',
    short: '두 점 사이를 이어 그 중간 값들을 만드는 것. 두 얼굴의 z를 이으면 얼굴이 서서히 변해요.',
    emoji: '🌈',
    chapter: 19
  },
  {
    id: 'autoregressive',
    term: '자기회귀',
    short: '앞에서 만든 것을 보고 다음 하나를 만드는 일을 반복하는 생성 방식. GPT가 이렇게 글을 써요.',
    long: '전체 확률을 "다음 토큰 확률"의 곱으로 쪼개서, 각 조각을 분류 문제로 배워요. 순서가 자연스러운 글에 잘 맞아요.',
    emoji: '⛓️',
    chapter: 19
  },
  {
    id: 'diffusion-model',
    term: '확산 모델',
    short: '그림을 노이즈로 망가뜨리는 과정을 정해 두고, 그걸 거꾸로 되돌리는 법을 배우는 생성 모델.',
    long: '학습은 "섞인 노이즈 맞히기"라는 단순한 회귀라 안정적이고, 생성은 노이즈에서 출발해 조금씩 청소하는 수백 걸음이에요. 대신 느려요.',
    emoji: '🌫️',
    chapter: 20
  },
  {
    id: 'reverse-process',
    term: '역재생',
    short: '확산 모델의 생성 과정. 순수한 노이즈에서 출발해 예측한 노이즈를 조금씩 걷어 내요.',
    long: '걸음마다 새 노이즈를 살짝 다시 섞어서, 같은 출발점이라도 매번 다른 그림에 도착할 수 있어요.',
    emoji: '⏪',
    chapter: 20
  },
  {
    id: 'latent-diffusion',
    term: '잠재 확산',
    aliases: ['Stable Diffusion'],
    short: '그림을 오토인코더로 작게 압축한 공간에서 확산을 돌려 비용을 크게 줄이는 방법.',
    long: '세부 묘사는 오토인코더가, 내용 만들기는 확산이 나눠 맡아요. Stable Diffusion이 이 설계로 유명해졌어요.',
    emoji: '🗺️',
    chapter: 20
  },
  {
    id: 'classifier-free-guidance',
    term: 'classifier-free guidance',
    aliases: ['CFG'],
    short: '"글을 준 예측"과 "안 준 예측"의 차이 방향을 부풀려 걸어, 그림이 글을 더 잘 따르게 하는 기법.',
    long: '세기를 올리면 프롬프트를 충실히 따르지만, 너무 올리면 다양성이 줄고 색이 과하게 진해져요.',
    emoji: '🎚️',
    chapter: 20
  }
];
