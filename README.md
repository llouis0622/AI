# All of AI Hub

AI 정리와 학습을 하나로 모은 통합 페이지. 두 지식베이스([all-of-ai](https://github.com/llouis0622/all-of-ai), [AI-Engineer-Archive](https://github.com/llouis0622/AI-Engineer-Archive))를 통합하고, 학습 로드맵·실전 코드랩·복습 퀴즈를 더했다.

배포 URL: `https://llouis0622.github.io/AI/`

## 구성

| 트랙 | 구성 | 성격 |
| --- | --- | --- |
| 학습 로드맵 | 8단계 학습 경로 | 무엇을 어떤 순서로 공부할지 |
| 커리큘럼 (`docs/curriculum/`) | 6부 · 12챕터 · 28강 | 개념과 직관, 체계 |
| 핸드북 (`docs/handbook/`) | 14파트 · 166편 | 수식 유도와 이론, 실무 관점 |
| 코드랩 (`docs/practice/`) | 6영역 · 20편 | 처음부터 끝까지 실행되는 완전한 코드 |
| 복습 퀴즈 (`docs/review/`) | 9세트 · 90+ 문항 | 능동 회상 퀴즈, 맞힘/복습 기록(localStorage) |

퀴즈는 커스텀 Vue 컴포넌트(`docs/.vitepress/theme/Quiz.vue`)로 구현되어, 정답 확인과 자기 평가 기록이 브라우저에 저장된다.

## 로컬 실행

```bash
npm install
npm run dev        # http://localhost:5173/AI/
npm run build      # 정적 빌드
npm run preview    # 빌드 결과 미리보기
```

## 배포

`main` 브랜치에 푸시하면 GitHub Actions가 VitePress 빌드 후 GitHub Pages로 배포한다. 저장소 설정(Settings → Pages)에서 **Source를 GitHub Actions로** 지정해야 한다.

`docs/.vitepress/config.mts`의 `base`는 저장소 이름과 일치하는 `/AI/`로 설정되어 있다. 저장소 이름을 바꾸면 이 값도 함께 수정해야 한다.

## 기술 스택

- [VitePress](https://vitepress.dev/) — 정적 사이트 생성기
- markdown-it-mathjax3 — 수식 렌더링
- [Mermaid](https://mermaid.js.org/) — 다이어그램
- Vue 3 커스텀 컴포넌트 — 퀴즈/진행률
- GitHub Actions — 자동 배포
