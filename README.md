# All of AI

처음 배우는 사람이 이 페이지 하나로 인공지능을 완주하도록 쓴 책. 직선 하나를 긋는 학습 기계에서 GPT·확산 모델·강화학습·에이전트·서빙까지, 한 사람의 목소리로 쓴 9부 28장에 코드랩 20편과 부별 복습 퀴즈를 더했다.

배포 URL: `https://llouis0622.github.io/AI/`

## 구성

| | 위치 | 내용 |
| --- | --- | --- |
| 책 | `docs/book/` | 여는 글 + 28장. 모든 장은 질문 → 직관 → 필요한 자리의 수식 → 핵심 요약 → 스스로 점검 → 다음 장 예고로 흐른다 |
| 코드랩 | `docs/practice/` | 20편. NumPy 역전파, PyTorch 파이프라인, U-Net·탐지·ViT, BPE·Transformer·GPT, BERT/LoRA 파인튜닝, RAG, 에이전트, VAE/GAN, DDPM, DQN, PPO, CLIP, Whisper, 서빙 |
| 복습 퀴즈 | `docs/review/` | 9세트(부별). 커스텀 Vue 퀴즈 컴포넌트로 정답 확인과 맞힘/복습 기록(localStorage) |

## 로컬 실행

```bash
npm install
npm run dev        # http://localhost:5173/AI/
npm run build      # 정적 빌드 (데드링크 검사 포함)
npm run preview
```

## 배포

`main`에 푸시하면 GitHub Actions가 VitePress를 빌드해 GitHub Pages로 배포한다(Settings → Pages → Source: GitHub Actions). `docs/.vitepress/config.mts`의 `base`는 저장소 이름과 일치하는 `/AI/`다.

## 기술 스택

VitePress · markdown-it-mathjax3(수식) · Mermaid(다이어그램) · Vue 3 커스텀 컴포넌트(퀴즈) · GitHub Actions
