import type { VizMeta } from './types';

export const viz: VizMeta[] = [
  {
    id: 'serving-sim',
    title: '모델 서빙 시뮬레이터',
    tagline: '요청이 줄을 서고, GPU가 한 묶음씩 처리해요. 빠르게, 많이, 싸게 — 셋 다 잡을 수 있을까요?',
    icon: '🍽️',
    component: 'part9/ServingSim',
    chapter: 27,
    howTo: '"🔥 트래픽 폭주"를 눌러 대기열이 터지는 걸 본 뒤, 배치 크기와 서버 수로 구해 보세요! (실제 서버를 단순화한 시뮬레이션이에요)',
    aspect: 1.05
  },
  {
    id: 'drift-monitor',
    title: '늙는 모델 모니터',
    tagline: '세상이 바뀌면 모델은 늙어요. 경보를 보고 언제 다시 가르칠지 정해 봐요.',
    icon: '📉',
    component: 'part9/DriftMonitor',
    chapter: 27,
    howTo: '▶ 1년 돌려 보기를 누르고, 경보가 울리면 🔁 재학습을 눌러 보세요. 정답이 늦게 오면 재학습도 늦어져요! (개념을 보여 주기 위한 간이 모델이에요)',
    aspect: 0.95
  },
  {
    id: 'skill-tree',
    title: '실력 나무',
    tagline: '코드랩을 끝내고 할 일을 체크할 때마다 나무에 잎이 돋아요. 다음엔 무엇을 배울까요?',
    icon: '🌳',
    component: 'part9/SkillTree',
    chapter: 28,
    howTo: '가지를 눌러 단계별 할 일을 확인하고, 다른 탭에서 "변하는 것 vs 안 변하는 것" 카드 게임도 해 보세요!',
    aspect: 0.9
  }
];
