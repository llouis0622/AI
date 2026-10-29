import type { VizMeta } from './types';

export const viz: VizMeta[] = [
  {
    id: 'latent-explorer',
    title: '얼굴 잠재 공간 탐험',
    tagline: '지도 위 점 하나가 얼굴 하나 — 잠재 벡터 z를 움직이면 디코더가 그리는 얼굴이 바뀌어요.',
    icon: '🧭',
    component: 'part6/LatentExplorer',
    chapter: 19,
    howTo: '「그냥 오토인코더」로 바꾼 뒤 🎲를 여러 번 눌러 보세요. 빈 곳에 떨어지면 쓰레기 그림이 나와요! (얼굴은 개념을 보여 주려고 만든 장난감 디코더가 그려요)',
    aspect: 1.1
  },
  {
    id: 'gan-duel',
    title: '위조범 vs 감정사',
    tagline: '진짜 같은 가짜를 만들려는 생성기와, 가려내려는 판별기의 대결을 실제로 훈련시켜 봐요.',
    icon: '🥊',
    component: 'part6/GanDuel',
    chapter: 19,
    howTo: '세 가지 실험을 차례로 ▶ 시작해 보세요. 「모드 붕괴」에서는 가짜가 한쪽 봉우리에만 몰려요! (1차원 데이터로 만든 작은 진짜 GAN이에요)',
    aspect: 1.25
  },
  {
    id: 'diffusion-noise',
    title: '망가뜨리고 되살리기',
    tagline: '그림에 노이즈를 섞어 망가뜨리는 앞 방향과, 노이즈를 조금씩 걷어 내는 역재생을 둘 다 해 봐요.',
    icon: '🌫️',
    component: 'part6/DiffusionNoise',
    chapter: 20,
    howTo: 't를 끝까지 밀어 그림을 완전히 망가뜨린 뒤, 2번 탭에서 노이즈로부터 그림을 되살려 보세요. (그림 4장만 아는 완벽한 계산기가 노이즈를 짐작해요)',
    aspect: 0.95
  },
  {
    id: 'diffusion-2d',
    title: '점 구름 확산 실험',
    tagline: '웃는 얼굴 모양의 점 500개를 노이즈로 흩뜨렸다가, 화살표를 따라 다시 모아요.',
    icon: '☁️',
    component: 'part6/Diffusion2D',
    chapter: 20,
    howTo: '2번 탭에서 100걸음으로 되돌린 다음, 걸음 수를 1로 바꿔 다시 해 보세요. 점들이 어디로 갈까요? (화살표는 데이터 분포를 정확히 알 때 계산되는 "완벽한 모델"의 답이에요)',
    aspect: 1.25
  }
];
