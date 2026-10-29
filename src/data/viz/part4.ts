import type { VizMeta } from './types';

export const viz: VizMeta[] = [
  {
    id: 'convolution-scan',
    title: '합성곱 도장 찍기',
    tagline: '3×3 필터가 그림 위를 훑으며 곱하고 더해 특징 맵을 만드는 과정을 한 칸씩 봐요.',
    icon: '🔍',
    component: 'part4/ConvolutionScan',
    chapter: 13,
    howTo: '필터를 "세로 경계"와 "가로 경계"로 바꿔 보세요. "✏️ 지우고 그리기"로 직접 그린 그림에도 찍을 수 있어요.',
    aspect: 1.1
  },
  {
    id: 'cnn-hierarchy-3d',
    title: 'CNN 층 쌓기 3D',
    tagline: '사진 한 장이 층을 지나며 작아지고 두꺼워지는 모습을 3D로 둘러봐요.',
    icon: '🧱',
    component: 'part4/CnnHierarchy3D',
    chapter: 13,
    is3D: true,
    howTo: '"▶ 사진 한 장 통과시키기"를 누르고, 층을 하나씩 눌러 무엇을 보는지와 수용 영역을 확인해 보세요.',
    aspect: 1.25
  },
  {
    id: 'vision-task-ladder',
    title: '비전 과제의 사다리',
    tagline: '같은 사진에 "무엇?", "어디?", "어떤 픽셀?"을 차례로 물어보며 답이 어떻게 자세해지는지 봐요.',
    icon: '🪜',
    component: 'part4/VisionTaskLadder',
    chapter: 14,
    howTo: '1층→2층→3층을 차례로 눌러 보고, 사진 속 물체를 끌어서 다른 물체를 가려 보세요.',
    aspect: 1.1
  },
  {
    id: 'iou-nms',
    title: 'IoU 재기와 NMS 정리',
    tagline: '두 상자의 겹침을 재는 IoU, 그리고 겹친 후보 상자를 정리하는 NMS를 직접 해 봐요.',
    icon: '📦',
    component: 'part4/IouNms',
    chapter: 14,
    howTo: '"📘 책의 문제"로 IoU를 검산한 뒤, ② 탭에서 문턱을 0.3과 0.8로 바꿔 NMS를 돌려 보세요.',
    aspect: 1.2
  },
  {
    id: 'vit-patches',
    title: 'ViT: 사진을 문장으로',
    tagline: '사진을 조각내 한 줄로 세우고 토큰으로 바꿔 Transformer에 넣는 과정을 따라가요.',
    icon: '🧩',
    component: 'part4/VitPatches',
    chapter: 14,
    howTo: '"다음 ▶"으로 한 단계씩 넘기고, 마지막 단계에서 조각에 마우스를 올려 어텐션 선을 보세요.',
    aspect: 1.3
  },
  {
    id: 'wave-to-spectrogram',
    title: '소리를 그림으로: 스펙트로그램',
    tagline: '소리의 떨림을 프리즘처럼 주파수로 나누고, 시간 순서로 이어 붙여 "소리의 악보"를 만들어요.',
    icon: '🎵',
    component: 'part4/WaveToSpectrogram',
    chapter: 15,
    howTo: '"도·미·솔 화음"을 듣고 세 줄로 나뉘는 걸 확인한 뒤, "아·이·우"와 "박수"의 무늬를 비교해 보세요.',
    aspect: 0.95
  }
];
