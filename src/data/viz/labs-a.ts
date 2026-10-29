import type { VizMeta } from './types';

/** Code lab visualizations (labs 1–10). */
export const viz: VizMeta[] = [
  {
    id: 'lab01-forward-backward',
    title: '숫자 784개의 여행',
    tagline: '그림이 행렬이 되고, 확률이 되고, 기울기가 되어 거꾸로 흐르는 과정을 단계별로 봐요.',
    icon: '🔢',
    component: 'labs/Lab01ForwardBackward',
    lab: 1,
    howTo: '"다음 →"을 누르며 오른쪽 코드의 노란 줄이 그림의 어느 부분인지 짝지어 보세요.',
    aspect: 1.5
  },
  {
    id: 'lab02-training-pipeline',
    title: '학습 파이프라인 한 바퀴',
    tagline: '데이터 증강부터 AMP 스텝 순서, 학습률 곡선, 체크포인트까지 한 장씩 넘겨 봐요.',
    icon: '🏭',
    component: 'labs/Lab02TrainingPipeline',
    lab: 2,
    howTo: '3단계의 번호 순서(scale → backward → unscale → clip → step)를 꼭 기억해 두세요. 순서가 틀리면 버그예요!',
    aspect: 1.5
  },
  {
    id: 'lab03-freeze-unfreeze',
    title: '얼리고 녹이는 전이학습',
    tagline: '사전학습 몸통을 얼마나 얼려 둘지에 따라 학습하는 파라미터와 정확도가 어떻게 바뀌는지 봐요.',
    icon: '🧊',
    component: 'labs/Lab03FreezeUnfreeze',
    lab: 3,
    howTo: '주황 화살표(기울기)가 어디서 멈추는지 보세요. 멈춘 곳 왼쪽은 학습되지 않아요.',
    aspect: 1.5
  },
  {
    id: 'lab04-unet',
    title: 'U자 모양 분할 네트워크',
    tagline: '내려가며 압축하고, 올라오며 스킵 연결로 위치를 되살리는 U-Net을 차근차근 조립해요.',
    icon: '🫧',
    component: 'labs/Lab04Unet',
    lab: 4,
    howTo: '3단계에서 주황색 스킵 화살표가 어느 층과 어느 층을 잇는지 따라가 보세요.',
    aspect: 1.5
  },
  {
    id: 'lab05-detection',
    title: '박스 정리 대작전 (IoU·NMS)',
    tagline: '겹친 박스의 IoU를 재고, NMS로 중복을 지워 사람 둘만 남기는 과정을 직접 봐요.',
    icon: '📦',
    component: 'labs/Lab05Detection',
    lab: 5,
    howTo: '3단계에서 어떤 박스가 몇 번째 라운드에 지워지는지, 그때의 IoU 숫자를 확인해 보세요.',
    aspect: 1.5
  },
  {
    id: 'lab06-vit',
    title: '사진을 문장처럼 읽는 ViT',
    tagline: '사진을 조각내 토큰 줄로 세우고, [CLS] 토큰 하나가 정답을 말하기까지를 따라가요.',
    icon: '🧩',
    component: 'labs/Lab06Vit',
    lab: 6,
    howTo: '1~3단계에서 텐서 모양이 (3, 32, 32) → (64, 192) → (65, 192)로 바뀌는 순간을 눈여겨보세요.',
    aspect: 1.5
  },
  {
    id: 'lab07-bpe-train',
    title: '바이트를 합쳐 어휘 만들기',
    tagline: '실제 BPE를 돌려서, 가장 흔한 쌍이 새 토큰이 되고 어휘가 자라는 모습을 봐요.',
    icon: '🧱',
    component: 'labs/Lab07BpeTrain',
    lab: 7,
    howTo: '5단계에서 "lowest"가 병합 순서대로 [low][est]가 되는 과정을 한 줄씩 확인해 보세요.',
    aspect: 1.5
  },
  {
    id: 'lab08-transformer-map',
    title: '인코더·디코더 지도',
    tagline: '문자열 뒤집기 Transformer에서 두 마스크와 cross-attention, 한 토큰씩 생성하는 과정을 봐요.',
    icon: '🗺️',
    component: 'labs/Lab08TransformerMap',
    lab: 8,
    howTo: '3~4단계의 격자를 비교해 보세요. 회색 칸이 "보지 않는" 자리예요. 패딩 마스크는 열, 인과 마스크는 삼각형이에요.',
    aspect: 1.5
  },
  {
    id: 'lab09-gpt',
    title: '다음 글자를 맞히는 GPT',
    tagline: '한 칸 민 정답, 인과 마스크, 가중치 공유, 샘플링, KV 캐시까지 미니 GPT를 한 바퀴 돌아요.',
    icon: '✍️',
    component: 'labs/Lab09Gpt',
    lab: 9,
    howTo: '4단계에서 온도 → top-k → 주사위 순서로 후보 막대가 어떻게 바뀌는지 보세요.',
    aspect: 1.5
  },
  {
    id: 'lab10-bert-finetune',
    title: 'BERT로 리뷰 감성 분류',
    tagline: '사전학습 BERT에 작은 분류 헤드를 붙이고, 작은 학습률로 살살 다듬는 파인튜닝 과정을 봐요.',
    icon: '🎬',
    component: 'labs/Lab10BertFinetune',
    lab: 10,
    howTo: '4단계에서 학습률을 너무 크게 잡으면 사전학습 지식이 무너지는 모습을 확인해 보세요.',
    aspect: 1.5
  }
];
