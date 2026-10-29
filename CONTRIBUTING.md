# AIology 개발 가이드

AIology는 **Astro 7 + React 19 + MDX**로 만든 정적 사이트다. 원고는 MDX, 시각화는 React 컴포넌트(2D는 SVG/Canvas, 3D는 three.js + React Three Fiber)이며 GitHub Pages로 배포된다.

## 준비

```bash
# Node 24 이상
npm ci
npm run dev        # http://localhost:4321/AI/
npm run build      # astro build + Pagefind 검색 색인 (dist/)
npm run preview    # 빌드 결과 미리보기
npm run typecheck  # astro sync 후 실행
npm run check:viz  # <Viz> 참조·등록부·컴포넌트 파일 정합성 검사
```

## 폴더 지도

| 경로 | 내용 |
| --- | --- |
| `src/content/book/*.mdx` | 28장 원고 + 여는 글. frontmatter에 `summary`(그림 요약 카드), `mascot`, `labs` |
| `src/content/practice/*.mdx` | 코드랩 20편 + 개요 |
| `src/content/review/*.mdx` | 복습 퀴즈 9세트 (`<Quiz>`, `<QuizProgress>`) |
| `src/components/viz/` | 시각화. `core/`는 공통 키트, `part1/`~`part9/`는 장별, `labs/`는 코드랩 |
| `src/components/viz/Viz.tsx` | 모든 시각화의 호스트(틀·지연 로딩·3D 실행 버튼·크게 보기·처음으로) |
| `src/data/viz/*.ts` | 시각화 등록부 (제목, 한 줄 설명, 아이콘, 위치, 3D 여부, 안내 문구) |
| `src/data/glossary/*.ts` | 용어 사전. 본문에서 처음 나오는 용어에 자동으로 툴팁이 붙는다 |
| `src/lib/rehype-*.mjs` | 수식·코드 접기, 용어 툴팁, 링크 base 보정 플러그인 |
| `src/styles/` | 디자인 토큰(파스텔 라이트 / 네온 다크), 본문·시각화·지도 스타일 |

## 시각화 하나 추가하기

1. **컴포넌트** — `src/components/viz/<그룹>/<이름>.tsx`

   ```tsx
   import type { VizProps } from '../Viz';
   import { Controls, Slider, Note } from '../core/ui';
   import { Plot } from '../core/plot';
   import { useVizColors } from '../core/colors';

   export default function MyViz({ expanded }: VizProps) {
     const c = useVizColors();          // 테마에 맞는 색 (라이트/다크 자동)
     // … 무대(Plot / canvas / Scene3D) → vz-pad(Note, Stat) → Controls 순서
   }
   ```

   틀(제목·설명·안내 문구·버튼)은 호스트가 그리므로 컴포넌트는 안쪽만 그린다.

2. **등록** — `src/data/viz/<그룹>.ts`

   ```ts
   {
     id: 'my-viz',                 // MDX에서 쓰는 고유 id
     title: '짧은 제목',
     tagline: '무엇을 해 볼 수 있는지 한 문장',
     icon: '🧪',
     component: '<그룹>/<이름>',
     chapter: 5,                   // 또는 lab: 3
     is3D: false,                  // WebGL이면 true → "▶ 3D로 실행하기" 뒤에 지연 로딩
     howTo: '안내 로봇 알고가 해 줄 한마디',
     aspect: 1.4                   // 로딩 전 자리 비율(가로/세로)
   }
   ```

3. **배치** — 해당 MDX에 한 번 `import Viz from '@/components/viz/Viz.tsx';`를 두고, 개념을 설명한 문단 바로 뒤에 `<Viz id="my-viz" client:visible />`.

4. **확인** — `npm run check:viz`, `npm run typecheck`, `npm run build` 후 `/AI/playground/my-viz/`에서 라이트·다크, 모바일 폭(390px)을 모두 본다.

### 공통 키트 (`src/components/viz/core/`)

- `ui` — `Slider`, `Button`, `Segmented`, `Toggle`, `Stat`, `Controls`, `Note`(알고의 말풍선), `PlayBar`, `Legend`, `VizLayout`
- `plot` — 반응형 SVG 좌표계 `Plot`(자식에 `x`, `y` 스케일 전달), `MiniChart`, `linear`, `linePath`
- `hooks` — `useElementSize`, `useRafLoop`(보이지 않으면 멈춤), `useInterval`, `useCanvas`, `useInView`, `usePrefersReducedMotion`
- `colors` — `useVizColors`, `ramp`, `withAlpha`, `mixHex`
- `math` — `clamp`, `lerp`, `linspace`, `softmax`, `sigmoid`, `rng(seed)`, `fmt`, `contourSegments`
- `three` — `Scene3D`, `Surface`, `Contours`, `Ball`, `Arrow3D`, `Label3D`, `DomainAxes`, `makeMapper`
- `story` — 코드랩용 단계별 설명 `CodeStory`와 SVG 블록 `Block`, `FlowArrow`, `Grid`

### 품질 기준

- **중학생 눈높이**: 짧고 친근한 한국어. 결과에 반응하는 `Note`로 칭찬·경고·다음 할 일을 알려 준다.
- **직접 만지기**: 끌기, 밀기, 누르기, 재생 — 정지된 그림이 아니라 실험이어야 한다. 단순화한 모형이면 그렇다고 밝힌다.
- **두 테마**: 색은 항상 `useVizColors()`에서. 라이트와 다크를 모두 확인한다.
- **반응형**: 390px 폭에서도 겹치지 않게, 크게 보기(`expanded`)에서는 무대를 키운다.
- **가볍게**: 애니메이션은 재생 중이고 화면에 보일 때만. 무거운 계산은 `useMemo`.
- **접근성**: 기본 입력 요소 사용, SVG에 `ariaLabel`, 움직임 줄이기 설정을 존중.

## 그림 요약 카드 (frontmatter)

```yaml
mascot: "알고가 질문 아래에서 건네는 한마디"
summary:
  - icon: "🏠"
    title: "짧은 제목"
    text: "중학생도 이해하는 1–2문장. **굵게** 가능"
    visual: "float"   # float | pulse | spin | shake | grow | orbit | slide | bounce
```

## 용어 사전

`src/data/glossary/part<N>.ts`에 `{ id, term, aliases?, short, long?, emoji, chapter }`를 추가한다. 페이지마다 **처음 나오는 한 번**에만 툴팁이 붙고, 용어 사전 페이지에도 자동으로 실린다. 일상어처럼 흔한 별칭은 피한다.
