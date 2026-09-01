import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

type Entry = [string, string]

function group(text: string, dir: string, entries: Entry[]) {
  return {
    text,
    collapsed: true,
    items: [
      { text: '파트 개요', link: `/handbook/${dir}/` },
      ...entries.map(([file, title]) => ({ text: title, link: `/handbook/${dir}/${file}` }))
    ]
  }
}

function chapter(text: string, dir: string, entries: Entry[]) {
  return {
    text,
    collapsed: false,
    items: entries.map(([file, title]) => ({ text: title, link: `/curriculum/${dir}/${file}` }))
  }
}

const curriculumSidebar = [
  {
    text: 'I. 인공지능 기초',
    items: [
      chapter('Ch 01. 인공지능과 학습', 'ch01', [
        ['lecture01', 'Lecture 01. 인공지능이란 무엇인가'],
        ['lecture02', 'Lecture 02. 학습의 개념'],
        ['lecture03', 'Lecture 03. 학습 패러다임']
      ]),
      chapter('Ch 02. 일반화와 이론적 기초', 'ch02', [
        ['lecture04', 'Lecture 04. 일반화 문제'],
        ['lecture05', 'Lecture 05. 편향과 분산']
      ])
    ]
  },
  {
    text: 'II. 머신러닝과 딥러닝 핵심',
    items: [
      chapter('Ch 03. 머신러닝의 기본 구조', 'ch03', [
        ['lecture06', 'Lecture 06. 경험적 위험 최소화'],
        ['lecture07', 'Lecture 07. 전통적 머신러닝 모델']
      ]),
      chapter('Ch 04. 딥러닝과 표현 학습', 'ch04', [
        ['lecture08', 'Lecture 08. 신경망의 원리'],
        ['lecture09', 'Lecture 09. 표현 학습'],
        ['lecture10', 'Lecture 10. 주요 신경망 구조']
      ])
    ]
  },
  {
    text: 'III. 생성 모델과 파운데이션 모델',
    items: [
      chapter('Ch 05. 생성 모델', 'ch05', [
        ['lecture11', 'Lecture 11. 생성이란 무엇인가'],
        ['lecture12', 'Lecture 12. 생성 모델의 주요 접근법']
      ]),
      chapter('Ch 06. 대규모 언어 모델', 'ch06', [
        ['lecture13', 'Lecture 13. LLM의 본질'],
        ['lecture14', 'Lecture 14. 스케일링과 능력'],
        ['lecture15', 'Lecture 15. 정렬과 미세조정']
      ])
    ]
  },
  {
    text: 'IV. 문제 영역과 의사결정',
    items: [
      chapter('Ch 07. 문제 영역별 인공지능', 'ch07', [
        ['lecture16', 'Lecture 16. 비전/언어/오디오 문제의 구조']
      ]),
      chapter('Ch 08. 강화학습', 'ch08', [
        ['lecture17', 'Lecture 17. 강화학습의 기본 개념'],
        ['lecture18', 'Lecture 18. 강화학습의 주요 방법론']
      ])
    ]
  },
  {
    text: 'V. AI 시스템과 에이전트',
    items: [
      chapter('Ch 09. LLM 시스템', 'ch09', [
        ['lecture19', 'Lecture 19. LLM 기반 시스템 설계']
      ]),
      chapter('Ch 10. 에이전트형 인공지능', 'ch10', [
        ['lecture20', 'Lecture 20. 에이전트형 인공지능']
      ])
    ]
  },
  {
    text: 'VI. 인공지능 수학과 컴퓨터 과학',
    items: [
      chapter('Ch 11. 인공지능 수학', 'ch11', [
        ['lecture21', 'Lecture 21. 선형대수와 표현'],
        ['lecture22', 'Lecture 22. 확률적 모델링'],
        ['lecture23', 'Lecture 23. 최적화와 학습']
      ]),
      chapter('Ch 12. 컴퓨터 과학 기초', 'ch12', [
        ['lecture24', 'Lecture 24. 자료구조'],
        ['lecture25', 'Lecture 25. 컴퓨터 구조'],
        ['lecture26', 'Lecture 26. 운영체제'],
        ['lecture27', 'Lecture 27. 네트워크'],
        ['lecture28', 'Lecture 28. 데이터베이스']
      ])
    ]
  },
  {
    text: '복습',
    items: [{ text: '복습 퀴즈 허브', link: '/review/' }]
  }
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
      { text: '퀴즈 01. 인공지능 기초', link: '/review/quiz-foundations' },
      { text: '퀴즈 02. 머신러닝', link: '/review/quiz-ml' },
      { text: '퀴즈 03. 딥러닝', link: '/review/quiz-dl' },
      { text: '퀴즈 04. 생성 모델과 LLM', link: '/review/quiz-genai-llm' },
      { text: '퀴즈 05. 강화학습과 에이전트', link: '/review/quiz-rl-agents' },
      { text: '퀴즈 06. AI 수학', link: '/review/quiz-math' },
      { text: '퀴즈 07. CS 기초', link: '/review/quiz-cs' },
      { text: '퀴즈 08. LLM 엔지니어링 실무', link: '/review/quiz-llm-engineering' },
      { text: '퀴즈 09. MLOps와 시스템', link: '/review/quiz-mlops-systems' }
    ]
  },
  {
    text: '학습 경로',
    items: [{ text: '학습 로드맵', link: '/roadmap' }]
  }
]

const part01: Entry[] = [
  ['01-type-hints-and-protocols', '타입 힌트와 정적 검사, 제네릭, Protocol'],
  ['02-decorators-and-closures', '데코레이터와 클로저, functools'],
  ['03-context-managers', '컨텍스트 매니저와 리소스 관리'],
  ['04-iterators-and-generators', '이터레이터, 제너레이터, 지연 평가'],
  ['05-classes-and-dataclasses', '클래스 심화: 매직 메서드, dataclass, __slots__'],
  ['06-exceptions-and-logging', '예외 설계와 로깅 전략'],
  ['07-concurrency', '동시성: GIL, threading, multiprocessing, asyncio'],
  ['08-environments-and-packaging', '가상환경, 의존성 관리, 패키징'],
  ['09-testing-with-pytest', 'pytest 기반 테스트'],
  ['10-numpy-internals', 'NumPy: 메모리 레이아웃와 브로드캐스팅'],
  ['11-pandas-internals', 'Pandas: 인덱싱, groupby, 메모리'],
  ['12-matplotlib', 'Matplotlib 체계 정리']
]

const part02: Entry[] = [
  ['01-vector-spaces-and-rank', '벡터 공간, 기저, 차원, 랭크'],
  ['02-inner-products-and-norms', '내적, 노름, 코사인 유사도'],
  ['03-matrix-multiplication-and-linear-maps', '행렬 곱의 세 해석과 선형변환'],
  ['04-eigendecomposition', '고유값 분해와 대각화'],
  ['05-svd-and-low-rank-approximation', 'SVD와 저랭크 근사, LoRA 연결'],
  ['06-pseudoinverse-and-least-squares', '의사역행렬과 최소제곱해'],
  ['07-gradients-and-directional-derivatives', '편미분, 그래디언트, 방향도함수'],
  ['08-jacobian-hessian-taylor', '야코비안, 헤시안, 테일러 전개'],
  ['09-chain-rule-and-computational-graphs', '연쇄법칙과 계산 그래프'],
  ['10-backpropagation-derivation', '역전파의 수식적 유도'],
  ['11-autodiff-modes', '전방 모드와 후방 모드 자동미분'],
  ['12-probability-and-distributions', '확률변수와 주요 분포'],
  ['13-bayes-mle-map', '베이즈 정리, MLE, MAP'],
  ['14-entropy-and-divergences', '엔트로피, 교차엔트로피, KL/JS 발산'],
  ['15-statistical-inference', '중심극한정리, 가설검정, 신뢰구간'],
  ['16-convexity-and-gradient-descent', '볼록성과 경사하강법의 수렴'],
  ['17-optimizers-and-schedules', '모멘텀부터 AdamW까지, 학습률 스케줄'],
  ['18-constrained-and-second-order-optimization', '제약 최적화와 2차 방법']
]

const part03: Entry[] = [
  ['01-learning-problem-taxonomy', '학습 문제의 정의와 분류'],
  ['02-generalization-and-bias-variance', '일반화 이론과 편향-분산 분해'],
  ['03-validation-and-data-leakage', '데이터 분할, 교차검증, 데이터 누수'],
  ['04-evaluation-metrics', '평가지표 총정리'],
  ['05-linear-regression', '선형회귀'],
  ['06-logistic-regression', '로지스틱회귀'],
  ['07-regularization', '정규화: L1, L2, ElasticNet'],
  ['08-knn-and-naive-bayes', 'kNN과 나이브베이즈'],
  ['09-svm', 'SVM: 마진, 쌍대, 커널'],
  ['10-decision-trees', '결정트리'],
  ['11-bagging-and-random-forest', '배깅과 랜덤포레스트'],
  ['12-boosting-and-gbdt', '부스팅과 Gradient Boosting'],
  ['13-gbdt-libraries-and-ensembling', 'XGBoost/LightGBM/CatBoost와 스태킹'],
  ['14-clustering-and-em', 'k-means, GMM과 EM, DBSCAN'],
  ['15-dimensionality-reduction', 'PCA, t-SNE, UMAP'],
  ['16-feature-engineering', '특징 공학과 파이프라인'],
  ['17-hyperparameter-search', '하이퍼파라미터 탐색과 Optuna']
]

const part04: Entry[] = [
  ['01-mlp-and-nonlinearity', '퍼셉트론에서 MLP로, 보편 근사 정리'],
  ['02-activation-functions', '활성함수'],
  ['03-loss-functions', '손실함수'],
  ['04-backpropagation-in-matrix-form', '역전파 완전 유도'],
  ['05-weight-initialization', '가중치 초기화'],
  ['06-normalization-layers', '정규화 계층'],
  ['07-dropout', '드롭아웃'],
  ['08-residual-connections-and-gradient-flow', '기울기 소실/폭발과 잔차 연결'],
  ['09-generalization-techniques', '일반화 기법'],
  ['10-training-failure-diagnosis', '학습이 안 될 때의 진단 순서']
]

const part05: Entry[] = [
  ['01-tensors', '텐서: dtype, device, 메모리 공유'],
  ['02-autograd-internals', 'autograd 내부 동작'],
  ['03-nn-module', 'nn.Module 구조'],
  ['04-dataset-and-dataloader', 'Dataset과 DataLoader'],
  ['05-training-loop-template', '표준 학습 루프 템플릿'],
  ['06-custom-components', '커스텀 레이어/손실/스케줄러'],
  ['07-torch-compile', 'torch.compile과 그래프 캡처'],
  ['08-export-torchscript-onnx', 'TorchScript와 ONNX 익스포트'],
  ['09-ddp', 'DataParallel과 DDP'],
  ['10-fsdp-and-zero', 'FSDP와 ZeRO, 그래디언트 체크포인팅'],
  ['11-profiling-and-memory', '프로파일링과 메모리 분석'],
  ['12-common-bug-patterns', '자주 만나는 버그 패턴']
]

const part06: Entry[] = [
  ['01-tensors-and-tf-function', 'tf.Tensor, eager, tf.function'],
  ['02-gradienttape', 'GradientTape 자동미분'],
  ['03-keras3-model-apis', 'Keras 3 모델 정의 세 가지 방식'],
  ['04-tf-data-pipeline', 'tf.data 파이프라인'],
  ['05-callbacks-and-custom-training-loop', '콜백과 커스텀 학습 루프'],
  ['06-framework-comparison', 'PyTorch와 TensorFlow 대조'],
  ['07-export-savedmodel-tflite', 'SavedModel과 TFLite']
]

const part07: Entry[] = [
  ['01-convolution-arithmetic', '합성곱 연산과 출력 크기 공식'],
  ['02-receptive-field-and-conv-variants', '수용영역과 합성곱 변형'],
  ['03-cnn-architectures-classic', 'LeNet, AlexNet, VGG'],
  ['04-resnet', 'ResNet과 잔차 학습'],
  ['05-efficient-architectures', 'Inception, DenseNet, MobileNet, EfficientNet, ConvNeXt'],
  ['06-transfer-learning', '전이학습과 파인튜닝 전략'],
  ['07-object-detection-fundamentals', 'IoU, NMS, 앵커, mAP'],
  ['08-detection-architectures', 'R-CNN, YOLO, DETR'],
  ['09-segmentation', 'FCN, U-Net, DeepLab, Mask R-CNN'],
  ['10-vision-transformer', 'ViT와 Swin'],
  ['11-self-supervised-vision', 'SimCLR, MoCo, BYOL, MAE'],
  ['12-image-augmentation', '이미지 증강 전략']
]

const part08: Entry[] = [
  ['01-tokenization', '토큰화: BPE, WordPiece, SentencePiece'],
  ['02-word-embeddings', 'Word2Vec, GloVe, FastText'],
  ['03-rnn-and-bptt', 'RNN과 BPTT'],
  ['04-lstm-and-gru', 'LSTM과 GRU'],
  ['05-seq2seq-and-attention', 'Seq2Seq와 어텐션의 등장'],
  ['06-scaled-dot-product-attention', 'Scaled Dot-Product Attention'],
  ['07-multi-head-attention', 'Multi-Head Attention'],
  ['08-positional-encoding', '위치 인코딩: 사인, 학습형, RoPE, ALiBi'],
  ['09-transformer-block-and-masking', 'FFN, 잔차, 정규화 위치, 마스킹'],
  ['10-transformer-from-scratch', 'Transformer 전체 구현'],
  ['11-bert-family', 'BERT 계열'],
  ['12-gpt-and-causal-lm', 'GPT 계열과 인과 언어 모델링'],
  ['13-t5-and-sentence-embeddings', 'T5와 문장 임베딩']
]

const part09: Entry[] = [
  ['01-autoencoders', '오토인코더와 표현 학습'],
  ['02-vae', 'VAE: ELBO와 재파라미터화'],
  ['03-gan-fundamentals', 'GAN: 미니맥스와 최적 판별자'],
  ['04-gan-failure-modes-and-wgan', '모드 붕괴와 WGAN'],
  ['05-conditional-and-image-translation-gans', 'cGAN, pix2pix, CycleGAN, StyleGAN'],
  ['06-diffusion-forward-process', '확산 순방향 과정'],
  ['07-ddpm-objective', 'DDPM 손실 유도'],
  ['08-ddim-and-fast-sampling', 'DDIM과 샘플링 가속'],
  ['09-latent-diffusion-and-guidance', 'Latent Diffusion과 CFG'],
  ['10-flow-matching', 'Flow Matching과 Rectified Flow'],
  ['11-generative-evaluation', 'FID, IS, CLIP Score']
]

const part10: Entry[] = [
  ['01-scaling-laws', '스케일링 법칙과 컴퓨트 배분'],
  ['02-attention-variants-and-kv-cache', 'MQA, GQA와 KV 캐시'],
  ['03-mixture-of-experts', 'MoE 라우팅과 로드 밸런싱'],
  ['04-modern-architecture-choices', 'SwiGLU, RMSNorm 채택 이유'],
  ['05-long-context', '긴 컨텍스트 확장 기법'],
  ['06-pretraining-data', '사전학습 데이터 구성과 중복 제거'],
  ['07-sft', 'SFT 데이터 설계'],
  ['08-rlhf', 'RLHF: 보상 모델과 PPO'],
  ['09-dpo', 'DPO 유도'],
  ['10-lora', 'LoRA 수식 유도'],
  ['11-peft-variants-and-qlora', 'QLoRA, Adapter, Prefix Tuning, IA3'],
  ['12-quantization', '양자화: PTQ, QAT, GPTQ, AWQ'],
  ['13-flashattention', 'FlashAttention의 타일링'],
  ['14-inference-serving-and-batching', 'PagedAttention, vLLM, 연속 배칭'],
  ['15-speculative-decoding', 'Speculative Decoding'],
  ['16-decoding-strategies', '디코딩 전략'],
  ['17-prompt-engineering', '프롬프트 엔지니어링'],
  ['18-rag-pipeline', 'RAG 파이프라인과 청킹'],
  ['19-vector-indexes', '벡터 인덱스: Flat, IVF, HNSW, PQ'],
  ['20-rag-evaluation', 'RAG 평가와 실패 분석'],
  ['21-agents', '에이전트: 도구 사용과 실행 제어'],
  ['22-llm-evaluation', 'LLM 평가와 환각 측정']
]

const part11: Entry[] = [
  ['01-recommender-systems', '추천 시스템'],
  ['02-time-series', '시계열'],
  ['03-anomaly-detection', '이상 탐지'],
  ['04-graph-neural-networks', '그래프 신경망'],
  ['05-rl-foundations', '강화학습 기초: MDP와 Q-learning'],
  ['06-policy-gradient-and-ppo', '정책 경사와 PPO']
]

const part12: Entry[] = [
  ['01-experiment-tracking', '실험 추적: MLflow와 W&B'],
  ['02-data-versioning', '데이터 버전 관리와 계보'],
  ['03-feature-store', '피처 스토어와 학습-서빙 왜곡'],
  ['04-pipeline-orchestration', '학습 파이프라인 오케스트레이션'],
  ['05-model-registry', '모델 레지스트리와 승격 절차'],
  ['06-serving-basics', 'FastAPI 기반 최소 서빙'],
  ['07-serving-infrastructure', 'TorchServe, Triton, 동적 배칭'],
  ['08-containers-and-kubernetes', '컨테이너와 쿠버네티스, GPU 스케줄링'],
  ['09-cicd-for-ml', 'CI/CD와 모델 검증 게이트'],
  ['10-monitoring-and-drift', '모니터링과 드리프트 탐지'],
  ['11-deployment-strategies', '섀도, 카나리, A/B 테스트'],
  ['12-cost-and-reproducibility', '비용 최적화와 재현성 체크리스트']
]

const part13: Entry[] = [
  ['01-gpu-architecture', 'GPU 아키텍처: SM, 워프, 텐서코어'],
  ['02-cuda-execution-model', 'CUDA 실행 모델과 메모리 계층'],
  ['03-roofline-model', '연산 강도와 루프라인 모델'],
  ['04-kernel-fusion', '커널 퓨전'],
  ['05-collective-communication', 'NCCL과 집합 통신 알고리즘'],
  ['06-parallelism-strategies', '데이터/텐서/파이프라인/3D 병렬'],
  ['07-data-loading-and-io', '데이터 로딩과 I/O 병목'],
  ['08-memory-accounting', '메모리 사용량 계산']
]

const part14: Entry[] = [
  ['01-experiment-management', '실험 관리 습관'],
  ['02-reading-and-reproducing-papers', '논문 읽기와 재현 절차'],
  ['03-code-review-for-ml', 'ML 코드 리뷰에서 보는 지점'],
  ['04-training-debug-checklist', '학습 실패 디버깅 체크리스트'],
  ['05-performance-optimization-priority', '성능 개선 우선순위 결정'],
  ['06-code-recipes', '자주 쓰는 코드 레시피']
]

export default withMermaid(
  defineConfig({
    title: 'All of AI Hub',
    description: 'AI 개념 커리큘럼과 실무 핸드북, 학습 로드맵과 복습 퀴즈를 하나로 모은 정리·학습 페이지',
    lang: 'ko-KR',
    base: '/AI/',
    cleanUrls: true,
    lastUpdated: true,
    ignoreDeadLinks: false,

    vite: {
      build: {
        chunkSizeWarningLimit: 2000
      }
    },

    markdown: {
      math: true,
      lineNumbers: true,
      theme: {
        light: 'github-light',
        dark: 'github-dark'
      }
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
              footer: {
                selectText: '선택',
                navigateText: '이동',
                closeText: '닫기'
              }
            }
          }
        }
      },
      nav: [
        { text: '홈', link: '/' },
        { text: '학습 로드맵', link: '/roadmap' },
        { text: '커리큘럼', items: [
          { text: 'I. 인공지능 기초', link: '/curriculum/ch01/lecture01' },
          { text: 'II. 머신러닝과 딥러닝 핵심', link: '/curriculum/ch03/lecture06' },
          { text: 'III. 생성 모델과 파운데이션 모델', link: '/curriculum/ch05/lecture11' },
          { text: 'IV. 문제 영역과 의사결정', link: '/curriculum/ch07/lecture16' },
          { text: 'V. AI 시스템과 에이전트', link: '/curriculum/ch09/lecture19' },
          { text: 'VI. 인공지능 수학과 컴퓨터 과학', link: '/curriculum/ch11/lecture21' }
        ]},
        { text: '핸드북', items: [
          { text: 'Python', link: '/handbook/01-python-foundations/' },
          { text: '수학', link: '/handbook/02-mathematics/' },
          { text: '머신러닝', link: '/handbook/03-machine-learning/' },
          { text: '딥러닝', link: '/handbook/04-deep-learning/' },
          { text: 'PyTorch', link: '/handbook/05-pytorch/' },
          { text: 'TensorFlow', link: '/handbook/06-tensorflow/' },
          { text: 'Computer Vision', link: '/handbook/07-computer-vision/' },
          { text: 'Sequence & NLP', link: '/handbook/08-sequence-nlp/' },
          { text: 'Generative Models', link: '/handbook/09-generative-models/' },
          { text: 'LLM Engineering', link: '/handbook/10-llm-engineering/' },
          { text: 'Other Domains', link: '/handbook/11-other-domains/' },
          { text: 'MLOps', link: '/handbook/12-mlops/' },
          { text: 'Systems & Hardware', link: '/handbook/13-systems-hardware/' },
          { text: 'Practitioner Guide', link: '/handbook/14-practitioner-guide/' }
        ]},
        { text: '코드랩', link: '/practice/' },
        { text: '복습 퀴즈', link: '/review/' }
      ],
      sidebar: {
        '/curriculum/': curriculumSidebar,
        '/practice/': practiceSidebar,
        '/handbook/': [
          group('Part 01 · Python Foundations', '01-python-foundations', part01),
          group('Part 02 · Mathematics for AI', '02-mathematics', part02),
          group('Part 03 · Machine Learning', '03-machine-learning', part03),
          group('Part 04 · Deep Learning Fundamentals', '04-deep-learning', part04),
          group('Part 05 · PyTorch', '05-pytorch', part05),
          group('Part 06 · TensorFlow and Keras', '06-tensorflow', part06),
          group('Part 07 · Computer Vision', '07-computer-vision', part07),
          group('Part 08 · Sequence Models and NLP', '08-sequence-nlp', part08),
          group('Part 09 · Generative Models', '09-generative-models', part09),
          group('Part 10 · LLM Engineering', '10-llm-engineering', part10),
          group('Part 11 · Other Domains', '11-other-domains', part11),
          group('Part 12 · MLOps and Production', '12-mlops', part12),
          group('Part 13 · Systems and Hardware', '13-systems-hardware', part13),
          group('Part 14 · Practitioner Guide', '14-practitioner-guide', part14)
        ],
        '/review/': reviewSidebar,
        '/roadmap': reviewSidebar
      },
      socialLinks: [
        { icon: 'github', link: 'https://github.com/llouis0622/AI' }
      ],
      docFooter: { prev: '이전', next: '다음' },
      returnToTopLabel: '맨 위로',
      sidebarMenuLabel: '메뉴',
      darkModeSwitchLabel: '테마',
      lightModeSwitchTitle: '라이트 모드로',
      darkModeSwitchTitle: '다크 모드로',
      lastUpdatedText: '마지막 수정',
      outlineTitle: '이 문서의 목차'
    }
  })
)
