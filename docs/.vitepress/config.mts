import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

type Entry = [string, string]

function part(text: string, entries: Entry[]) {
  return {
    text,
    collapsed: false,
    items: entries.map(([file, title]) => ({ text: title, link: `/book/${file}` }))
  }
}

const bookSidebar = [
  { text: '여는 글', items: [{ text: '이 책을 읽는 법', link: '/book/' }] },
  part('1부. 기계가 배운다는 것', [
    ['01-what-is-ai', '1장. 인공지능이라는 문제'],
    ['02-first-learning-machine', '2장. 첫 번째 학습 기계'],
    ['03-math-for-learning', '3장. 학습을 떠받치는 수학'],
    ['04-generalization', '4장. 외운 것과 배운 것']
  ]),
  part('2부. 고전 머신러닝', [
    ['05-classification', '5장. 갈림길을 배우다 — 분류'],
    ['06-trees-and-ensembles', '6장. 질문의 나무 — 트리와 앙상블'],
    ['07-unsupervised', '7장. 정답 없이 배우기'],
    ['08-evaluation', '8장. 성적표를 읽는 법']
  ]),
  part('3부. 신경망', [
    ['09-neural-networks', '9장. 뉴런의 산수 — 신경망'],
    ['10-backpropagation', '10장. 오차의 강을 거슬러 — 역전파'],
    ['11-training-deep-nets', '11장. 깊이의 대가'],
    ['12-debugging-training', '12장. 병든 학습 고치기']
  ]),
  part('4부. 지각', [
    ['13-cnn', '13장. 본다는 것 — CNN'],
    ['14-vision-tasks', '14장. 비전의 과제들'],
    ['15-audio', '15장. 듣는다는 것 — 오디오']
  ]),
  part('5부. 언어', [
    ['16-tokens-and-embeddings', '16장. 말을 숫자로 — 토큰과 임베딩'],
    ['17-rnn-to-attention', '17장. 문맥이라는 문제 — 어텐션'],
    ['18-transformer', '18장. Transformer — 현대 AI의 심장']
  ]),
  part('6부. 생성', [
    ['19-generative-models', '19장. 만들어내는 기계 — VAE와 GAN'],
    ['20-diffusion', '20장. 노이즈에서 그림으로 — 확산 모델']
  ]),
  part('7부. 거대 언어 모델', [
    ['21-llm-pretraining', '21장. 다음 단어의 기적 — 사전학습'],
    ['22-alignment', '22장. 야생마 길들이기 — 정렬'],
    ['23-using-llms', '23장. LLM 활용의 기술'],
    ['24-agents', '24장. 행동하는 언어 — 에이전트']
  ]),
  part('8부. 행동', [
    ['25-reinforcement-learning', '25장. 당근과 채찍의 수학 — 강화학습'],
    ['26-policy-gradient', '26장. 정책 경사와 PPO']
  ]),
  part('9부. 현장', [
    ['27-to-production', '27장. 연구실에서 세상으로'],
    ['28-keep-learning', '28장. 계속 공부하는 법']
  ]),
  { text: '복습', items: [{ text: '복습 퀴즈', link: '/review/' }] }
]

const practiceSidebar = [
  {
    text: 'A. 기초',
    items: [
      { text: '코드랩 개요 · 환경 준비', link: '/practice/' },
      { text: '01. NumPy로 신경망 밑바닥 구현', link: '/practice/01-numpy-mlp-from-scratch' },
      { text: '02. PyTorch 학습 파이프라인', link: '/practice/02-pytorch-training-pipeline' },
      { text: '03. 전이학습과 파인튜닝', link: '/practice/03-transfer-learning' }
    ]
  },
  {
    text: 'B. 컴퓨터 비전',
    items: [
      { text: '04. U-Net 세그멘테이션', link: '/practice/04-unet-segmentation' },
      { text: '05. 객체 탐지', link: '/practice/05-object-detection' },
      { text: '06. ViT 밑바닥 구현', link: '/practice/06-vit-from-scratch' }
    ]
  },
  {
    text: 'C. NLP와 LLM',
    items: [
      { text: '07. BPE 토크나이저 구현', link: '/practice/07-bpe-tokenizer' },
      { text: '08. Transformer 밑바닥 구현', link: '/practice/08-transformer-from-scratch' },
      { text: '09. GPT 밑바닥 구현과 학습', link: '/practice/09-gpt-from-scratch' },
      { text: '10. Hugging Face로 BERT 파인튜닝', link: '/practice/10-bert-finetuning-hf' },
      { text: '11. LoRA로 LLM 파인튜닝', link: '/practice/11-lora-finetuning' },
      { text: '12. RAG 시스템 구축', link: '/practice/12-rag-system' },
      { text: '13. LLM 에이전트 밑바닥 구현', link: '/practice/13-agent-from-scratch' }
    ]
  },
  {
    text: 'D. 생성 모델',
    items: [
      { text: '14. VAE와 GAN 구현', link: '/practice/14-vae-gan' },
      { text: '15. Diffusion(DDPM) 밑바닥 구현', link: '/practice/15-ddpm-from-scratch' }
    ]
  },
  {
    text: 'E. 강화학습',
    items: [
      { text: '16. DQN으로 CartPole 정복', link: '/practice/16-dqn-cartpole' },
      { text: '17. PPO 구현', link: '/practice/17-ppo-from-scratch' }
    ]
  },
  {
    text: 'F. 멀티모달과 운영',
    items: [
      { text: '18. CLIP과 멀티모달', link: '/practice/18-clip-multimodal' },
      { text: '19. 음성 인식과 오디오', link: '/practice/19-whisper-audio' },
      { text: '20. 모델 서빙', link: '/practice/20-serving-fastapi' }
    ]
  }
]

const reviewSidebar = [
  {
    text: '복습 퀴즈',
    items: [
      { text: '복습 허브', link: '/review/' },
      { text: '1부. 기계가 배운다는 것', link: '/review/quiz-part1' },
      { text: '2부. 고전 머신러닝', link: '/review/quiz-part2' },
      { text: '3부. 신경망', link: '/review/quiz-part3' },
      { text: '4부. 지각', link: '/review/quiz-part4' },
      { text: '5부. 언어', link: '/review/quiz-part5' },
      { text: '6부. 생성', link: '/review/quiz-part6' },
      { text: '7부. 거대 언어 모델', link: '/review/quiz-part7' },
      { text: '8부. 행동', link: '/review/quiz-part8' },
      { text: '9부. 현장', link: '/review/quiz-part9' }
    ]
  }
]

export default withMermaid(
  defineConfig({
    title: 'All of AI',
    description: '처음 배우는 사람이 이 페이지 하나로 인공지능을 완주하는 책 — 28장의 강의, 20편의 코드랩, 9세트의 복습 퀴즈',
    lang: 'ko-KR',
    base: '/AI/',
    cleanUrls: true,
    lastUpdated: true,
    ignoreDeadLinks: false,

    vite: {
      build: { chunkSizeWarningLimit: 2000 }
    },

    markdown: {
      math: true,
      lineNumbers: false,
      theme: { light: 'github-light', dark: 'github-dark' }
    },

    themeConfig: {
      outline: [2, 3],
      search: {
        provider: 'local',
        options: {
          translations: {
            button: { buttonText: '검색', buttonAriaLabel: '검색' },
            modal: {
              displayDetails: '상세 보기',
              resetButtonTitle: '초기화',
              backButtonTitle: '뒤로',
              noResultsText: '결과 없음',
              footer: { selectText: '선택', navigateText: '이동', closeText: '닫기' }
            }
          }
        }
      },
      nav: [
        { text: '홈', link: '/' },
        { text: '책', link: '/book/' },
        { text: '부별 바로가기', items: [
          { text: '1부. 기계가 배운다는 것', link: '/book/01-what-is-ai' },
          { text: '2부. 고전 머신러닝', link: '/book/05-classification' },
          { text: '3부. 신경망', link: '/book/09-neural-networks' },
          { text: '4부. 지각', link: '/book/13-cnn' },
          { text: '5부. 언어', link: '/book/16-tokens-and-embeddings' },
          { text: '6부. 생성', link: '/book/19-generative-models' },
          { text: '7부. 거대 언어 모델', link: '/book/21-llm-pretraining' },
          { text: '8부. 행동', link: '/book/25-reinforcement-learning' },
          { text: '9부. 현장', link: '/book/27-to-production' }
        ]},
        { text: '코드랩', link: '/practice/' },
        { text: '복습 퀴즈', link: '/review/' }
      ],
      sidebar: {
        '/book/': bookSidebar,
        '/practice/': practiceSidebar,
        '/review/': reviewSidebar
      },
      socialLinks: [{ icon: 'github', link: 'https://github.com/llouis0622/AI' }],
      docFooter: { prev: '이전', next: '다음' },
      returnToTopLabel: '맨 위로',
      sidebarMenuLabel: '메뉴',
      darkModeSwitchLabel: '테마',
      lightModeSwitchTitle: '라이트 모드로',
      darkModeSwitchTitle: '다크 모드로',
      lastUpdatedText: '마지막 수정',
      outlineTitle: '이 장의 목차'
    }
  })
)
