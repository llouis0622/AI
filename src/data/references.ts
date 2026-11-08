/**
 * Further reading shown at the end of each chapter and code lab.
 * Papers link to arXiv or their DOI; books and courses to their official pages.
 */
export type RefKind = 'book' | 'paper' | 'course' | 'article' | 'docs' | 'video' | 'code';

export interface Ref {
  title: string;
  /** Authors, organisation or publisher. */
  by?: string;
  year?: number;
  url: string;
  kind: RefKind;
  /** One line on what to read it for. */
  note?: string;
}

export const REF_KIND_LABEL: Record<RefKind, string> = {
  book: '책',
  paper: '논문',
  course: '강의',
  article: '글',
  docs: '문서',
  video: '영상',
  code: '코드'
};

const DL_BOOK = 'https://www.deeplearningbook.org/';
const ISLR = 'https://www.statlearning.com/';
const UDL = 'https://udlbook.github.io/udlbook/';

/** Keyed by chapter number. */
export const chapterRefs: Record<number, Ref[]> = {
  1: [
    { title: 'Artificial Intelligence: A Modern Approach (4th ed.)', by: 'Stuart Russell, Peter Norvig', year: 2020, url: 'https://aima.cs.berkeley.edu/', kind: 'book', note: 'AI 전반을 넓게 다루는 대표 교과서.' },
    { title: 'Computing Machinery and Intelligence', by: 'A. M. Turing', year: 1950, url: 'https://doi.org/10.1093/mind/LIX.236.433', kind: 'paper', note: '기계가 생각할 수 있는지를 처음 체계적으로 물은 글.' },
    { title: 'Machine Learning', by: 'Tom M. Mitchell', year: 1997, url: 'http://www.cs.cmu.edu/~tom/mlbook.html', kind: 'book', note: '작업, 성능, 경험으로 학습을 정의한 고전.' }
  ],
  2: [
    { title: 'CS229 Lecture Notes', by: 'Andrew Ng, Stanford', url: 'https://cs229.stanford.edu/main_notes.pdf', kind: 'course', note: '선형 회귀와 경사하강법을 수식으로 정리한 강의 노트.' },
    { title: 'Deep Learning, 4~5장', by: 'Ian Goodfellow, Yoshua Bengio, Aaron Courville', year: 2016, url: DL_BOOK, kind: 'book', note: '수치 계산과 머신러닝 기초.' },
    { title: 'An overview of gradient descent optimization algorithms', by: 'Sebastian Ruder', year: 2016, url: 'https://arxiv.org/abs/1609.04747', kind: 'paper', note: '경사하강법의 여러 변형을 한 번에 비교한다.' }
  ],
  3: [
    { title: 'Mathematics for Machine Learning', by: 'Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong', year: 2020, url: 'https://mml-book.github.io/', kind: 'book', note: '선형대수, 미적분, 확률을 머신러닝 관점에서 정리한 무료 교재.' },
    { title: 'Essence of linear algebra', by: '3Blue1Brown', url: 'https://www.3blue1brown.com/topics/linear-algebra', kind: 'video', note: '벡터와 행렬을 그림으로 이해하는 영상 시리즈.' },
    { title: 'Essence of calculus', by: '3Blue1Brown', url: 'https://www.3blue1brown.com/topics/calculus', kind: 'video', note: '미분과 연쇄법칙을 직관으로 잡아 준다.' }
  ],
  4: [
    { title: 'An Introduction to Statistical Learning (2nd ed.), 2·5장', by: 'Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani', year: 2021, url: ISLR, kind: 'book', note: '편향과 분산, 교차검증을 차근차근 설명한다.' },
    { title: 'A Few Useful Things to Know About Machine Learning', by: 'Pedro Domingos', year: 2012, url: 'https://doi.org/10.1145/2347736.2347755', kind: 'article', note: '과적합과 일반화에 대한 실무 감각을 짧게 정리한 글.' },
    { title: 'Cross-validation: evaluating estimator performance', by: 'scikit-learn', url: 'https://scikit-learn.org/stable/modules/cross_validation.html', kind: 'docs', note: '교차검증을 코드로 해 보는 공식 문서.' }
  ],
  5: [
    { title: 'Speech and Language Processing (3rd ed. draft), 로지스틱 회귀 장', by: 'Daniel Jurafsky, James H. Martin', url: 'https://web.stanford.edu/~jurafsky/slp3/', kind: 'book', note: '시그모이드, 교차 엔트로피, 소프트맥스를 예제로 풀어 쓴다.' },
    { title: 'An Introduction to Statistical Learning, 4장 분류', by: 'Gareth James 외', year: 2021, url: ISLR, kind: 'book' },
    { title: 'Logistic regression', by: 'scikit-learn', url: 'https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression', kind: 'docs' }
  ],
  6: [
    { title: 'Random Forests', by: 'Leo Breiman', year: 2001, url: 'https://doi.org/10.1023/A:1010933404324', kind: 'paper', note: '랜덤 포레스트를 처음 제안한 논문.' },
    { title: 'XGBoost: A Scalable Tree Boosting System', by: 'Tianqi Chen, Carlos Guestrin', year: 2016, url: 'https://arxiv.org/abs/1603.02754', kind: 'paper', note: '표 데이터 대회에서 널리 쓰인 부스팅 구현.' },
    { title: 'Ensembles: Gradient boosting, random forests, bagging, voting, stacking', by: 'scikit-learn', url: 'https://scikit-learn.org/stable/modules/ensemble.html', kind: 'docs' }
  ],
  7: [
    { title: 'An Introduction to Statistical Learning, 12장 비지도 학습', by: 'Gareth James 외', year: 2021, url: ISLR, kind: 'book', note: 'PCA와 군집화를 그림과 함께 설명한다.' },
    { title: 'Visualizing Data using t-SNE', by: 'Laurens van der Maaten, Geoffrey Hinton', year: 2008, url: 'https://www.jmlr.org/papers/v9/vandermaaten08a.html', kind: 'paper' },
    { title: 'Clustering', by: 'scikit-learn', url: 'https://scikit-learn.org/stable/modules/clustering.html', kind: 'docs', note: 'k-평균을 비롯한 군집 알고리즘을 비교한다.' }
  ],
  8: [
    { title: 'Metrics and scoring: quantifying the quality of predictions', by: 'scikit-learn', url: 'https://scikit-learn.org/stable/modules/model_evaluation.html', kind: 'docs', note: '정밀도, 재현율, ROC 등 평가 지표 총정리.' },
    { title: 'The Relationship Between Precision-Recall and ROC Curves', by: 'Jesse Davis, Mark Goadrich', year: 2006, url: 'https://doi.org/10.1145/1143844.1143874', kind: 'paper', note: '불균형 데이터에서 PR 곡선이 왜 필요한지.' }
  ],
  9: [
    { title: 'Neural Networks and Deep Learning', by: 'Michael Nielsen', year: 2015, url: 'http://neuralnetworksanddeeplearning.com/', kind: 'book', note: '손글씨 숫자 인식으로 신경망을 처음부터 만드는 무료 책.' },
    { title: 'Neural networks', by: '3Blue1Brown', url: 'https://www.3blue1brown.com/topics/neural-networks', kind: 'video', note: '뉴런과 층이 무엇을 하는지 그림으로 보여 준다.' },
    { title: 'Approximation by superpositions of a sigmoidal function', by: 'George Cybenko', year: 1989, url: 'https://doi.org/10.1007/BF02551274', kind: 'paper', note: '보편 근사 정리의 원 논문.' }
  ],
  10: [
    { title: 'Learning representations by back-propagating errors', by: 'David Rumelhart, Geoffrey Hinton, Ronald Williams', year: 1986, url: 'https://doi.org/10.1038/323533a0', kind: 'paper', note: '역전파를 널리 알린 논문.' },
    { title: 'Yes you should understand backprop', by: 'Andrej Karpathy', year: 2016, url: 'https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b', kind: 'article', note: '역전파를 직접 이해해야 하는 이유.' },
    { title: 'micrograd', by: 'Andrej Karpathy', url: 'https://github.com/karpathy/micrograd', kind: 'code', note: '100줄 남짓한 자동미분 엔진.' },
    { title: 'CS231n: Backpropagation, Intuitions', by: 'Stanford', url: 'https://cs231n.github.io/optimization-2/', kind: 'course' }
  ],
  11: [
    { title: 'Adam: A Method for Stochastic Optimization', by: 'Diederik P. Kingma, Jimmy Ba', year: 2014, url: 'https://arxiv.org/abs/1412.6980', kind: 'paper' },
    { title: 'Batch Normalization', by: 'Sergey Ioffe, Christian Szegedy', year: 2015, url: 'https://arxiv.org/abs/1502.03167', kind: 'paper' },
    { title: 'Deep Residual Learning for Image Recognition', by: 'Kaiming He 외', year: 2015, url: 'https://arxiv.org/abs/1512.03385', kind: 'paper', note: '잔차 연결(ResNet)을 제안한 논문.' },
    { title: 'Dropout: A Simple Way to Prevent Neural Networks from Overfitting', by: 'Nitish Srivastava 외', year: 2014, url: 'https://jmlr.org/papers/v15/srivastava14a.html', kind: 'paper' }
  ],
  12: [
    { title: 'A Recipe for Training Neural Networks', by: 'Andrej Karpathy', year: 2019, url: 'https://karpathy.github.io/2019/04/25/recipe/', kind: 'article', note: '학습이 안 될 때 하나씩 점검하는 순서.' },
    { title: 'Deep Learning Tuning Playbook', by: 'Google Research', year: 2023, url: 'https://github.com/google-research/tuning_playbook', kind: 'docs', note: '하이퍼파라미터 조정을 체계적으로 하는 법.' },
    { title: 'Understanding the difficulty of training deep feedforward neural networks', by: 'Xavier Glorot, Yoshua Bengio', year: 2010, url: 'https://proceedings.mlr.press/v9/glorot10a.html', kind: 'paper', note: '가중치 초기화와 기울기 소실을 다룬 논문.' }
  ],
  13: [
    { title: 'CS231n: Convolutional Neural Networks', by: 'Stanford', url: 'https://cs231n.github.io/convolutional-networks/', kind: 'course', note: '합성곱, 풀링, 수용 영역을 자세히 설명한다.' },
    { title: 'Gradient-based learning applied to document recognition', by: 'Yann LeCun 외', year: 1998, url: 'https://doi.org/10.1109/5.726791', kind: 'paper', note: 'LeNet으로 손글씨를 읽은 논문.' },
    { title: 'ImageNet Classification with Deep Convolutional Neural Networks', by: 'Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton', year: 2012, url: 'https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html', kind: 'paper', note: 'AlexNet. 딥러닝 붐의 시작점.' }
  ],
  14: [
    { title: 'You Only Look Once: Unified, Real-Time Object Detection', by: 'Joseph Redmon 외', year: 2015, url: 'https://arxiv.org/abs/1506.02640', kind: 'paper' },
    { title: 'U-Net: Convolutional Networks for Biomedical Image Segmentation', by: 'Olaf Ronneberger, Philipp Fischer, Thomas Brox', year: 2015, url: 'https://arxiv.org/abs/1505.04597', kind: 'paper' },
    { title: 'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale', by: 'Alexey Dosovitskiy 외', year: 2020, url: 'https://arxiv.org/abs/2010.11929', kind: 'paper', note: 'Vision Transformer(ViT).' }
  ],
  15: [
    { title: 'Robust Speech Recognition via Large-Scale Weak Supervision', by: 'Alec Radford 외', year: 2022, url: 'https://arxiv.org/abs/2212.04356', kind: 'paper', note: 'Whisper.' },
    { title: 'SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition', by: 'Daniel S. Park 외', year: 2019, url: 'https://arxiv.org/abs/1904.08779', kind: 'paper' },
    { title: 'librosa', by: 'librosa 개발팀', url: 'https://librosa.org/doc/latest/index.html', kind: 'docs', note: '스펙트로그램과 멜 필터를 직접 만들어 볼 수 있는 파이썬 라이브러리.' }
  ],
  16: [
    { title: 'Efficient Estimation of Word Representations in Vector Space', by: 'Tomas Mikolov 외', year: 2013, url: 'https://arxiv.org/abs/1301.3781', kind: 'paper', note: 'word2vec.' },
    { title: 'Neural Machine Translation of Rare Words with Subword Units', by: 'Rico Sennrich, Barry Haddow, Alexandra Birch', year: 2015, url: 'https://arxiv.org/abs/1508.07909', kind: 'paper', note: 'BPE 토큰화를 번역에 도입한 논문.' },
    { title: 'Hugging Face LLM Course, 토크나이저 장', by: 'Hugging Face', url: 'https://huggingface.co/learn/llm-course/chapter6/1', kind: 'course' }
  ],
  17: [
    { title: 'Long Short-Term Memory', by: 'Sepp Hochreiter, Jürgen Schmidhuber', year: 1997, url: 'https://doi.org/10.1162/neco.1997.9.8.1735', kind: 'paper' },
    { title: 'Neural Machine Translation by Jointly Learning to Align and Translate', by: 'Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio', year: 2014, url: 'https://arxiv.org/abs/1409.0473', kind: 'paper', note: '어텐션을 처음 번역에 쓴 논문.' },
    { title: 'The Unreasonable Effectiveness of Recurrent Neural Networks', by: 'Andrej Karpathy', year: 2015, url: 'https://karpathy.github.io/2015/05/21/rnn-effectiveness/', kind: 'article' }
  ],
  18: [
    { title: 'Attention Is All You Need', by: 'Ashish Vaswani 외', year: 2017, url: 'https://arxiv.org/abs/1706.03762', kind: 'paper', note: 'Transformer를 제안한 논문.' },
    { title: 'The Illustrated Transformer', by: 'Jay Alammar', year: 2018, url: 'https://jalammar.github.io/illustrated-transformer/', kind: 'article', note: '그림으로 따라가는 Transformer.' },
    { title: 'The Annotated Transformer', by: 'Harvard NLP', url: 'https://nlp.seas.harvard.edu/annotated-transformer/', kind: 'code', note: '논문을 한 줄씩 코드로 옮긴 글.' },
    { title: 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding', by: 'Jacob Devlin 외', year: 2018, url: 'https://arxiv.org/abs/1810.04805', kind: 'paper' }
  ],
  19: [
    { title: 'Auto-Encoding Variational Bayes', by: 'Diederik P. Kingma, Max Welling', year: 2013, url: 'https://arxiv.org/abs/1312.6114', kind: 'paper', note: 'VAE.' },
    { title: 'Generative Adversarial Networks', by: 'Ian Goodfellow 외', year: 2014, url: 'https://arxiv.org/abs/1406.2661', kind: 'paper', note: 'GAN.' },
    { title: 'Understanding Deep Learning, 생성 모델 장', by: 'Simon J. D. Prince', year: 2023, url: UDL, kind: 'book', note: 'VAE, GAN, 확산 모델을 한 흐름으로 설명하는 무료 교재.' }
  ],
  20: [
    { title: 'Denoising Diffusion Probabilistic Models', by: 'Jonathan Ho, Ajay Jain, Pieter Abbeel', year: 2020, url: 'https://arxiv.org/abs/2006.11239', kind: 'paper', note: 'DDPM.' },
    { title: 'High-Resolution Image Synthesis with Latent Diffusion Models', by: 'Robin Rombach 외', year: 2021, url: 'https://arxiv.org/abs/2112.10752', kind: 'paper', note: 'Stable Diffusion의 바탕이 된 논문.' },
    { title: 'What are Diffusion Models?', by: 'Lilian Weng', year: 2021, url: 'https://lilianweng.github.io/posts/2021-07-11-diffusion-models/', kind: 'article' }
  ],
  21: [
    { title: 'Language Models are Few-Shot Learners', by: 'Tom B. Brown 외', year: 2020, url: 'https://arxiv.org/abs/2005.14165', kind: 'paper', note: 'GPT-3.' },
    { title: 'Scaling Laws for Neural Language Models', by: 'Jared Kaplan 외', year: 2020, url: 'https://arxiv.org/abs/2001.08361', kind: 'paper' },
    { title: 'Training Compute-Optimal Large Language Models', by: 'Jordan Hoffmann 외', year: 2022, url: 'https://arxiv.org/abs/2203.15556', kind: 'paper', note: 'Chinchilla.' },
    { title: 'nanoGPT', by: 'Andrej Karpathy', url: 'https://github.com/karpathy/nanoGPT', kind: 'code', note: '작은 GPT를 직접 사전학습해 볼 수 있는 코드.' }
  ],
  22: [
    { title: 'Training language models to follow instructions with human feedback', by: 'Long Ouyang 외', year: 2022, url: 'https://arxiv.org/abs/2203.02155', kind: 'paper', note: 'InstructGPT와 RLHF.' },
    { title: 'Direct Preference Optimization: Your Language Model is Secretly a Reward Model', by: 'Rafael Rafailov 외', year: 2023, url: 'https://arxiv.org/abs/2305.18290', kind: 'paper', note: 'DPO.' },
    { title: 'Constitutional AI: Harmlessness from AI Feedback', by: 'Yuntao Bai 외', year: 2022, url: 'https://arxiv.org/abs/2212.08073', kind: 'paper' }
  ],
  23: [
    { title: 'Chain-of-Thought Prompting Elicits Reasoning in Large Language Models', by: 'Jason Wei 외', year: 2022, url: 'https://arxiv.org/abs/2201.11903', kind: 'paper' },
    { title: 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks', by: 'Patrick Lewis 외', year: 2020, url: 'https://arxiv.org/abs/2005.11401', kind: 'paper', note: 'RAG.' },
    { title: 'LoRA: Low-Rank Adaptation of Large Language Models', by: 'Edward J. Hu 외', year: 2021, url: 'https://arxiv.org/abs/2106.09685', kind: 'paper' }
  ],
  24: [
    { title: 'ReAct: Synergizing Reasoning and Acting in Language Models', by: 'Shunyu Yao 외', year: 2022, url: 'https://arxiv.org/abs/2210.03629', kind: 'paper' },
    { title: 'Toolformer: Language Models Can Teach Themselves to Use Tools', by: 'Timo Schick 외', year: 2023, url: 'https://arxiv.org/abs/2302.04761', kind: 'paper' },
    { title: 'LLM Powered Autonomous Agents', by: 'Lilian Weng', year: 2023, url: 'https://lilianweng.github.io/posts/2023-06-23-agent/', kind: 'article', note: '계획, 기억, 도구 사용을 한 번에 정리한 글.' }
  ],
  25: [
    { title: 'Reinforcement Learning: An Introduction (2nd ed.)', by: 'Richard S. Sutton, Andrew G. Barto', year: 2018, url: 'http://incompleteideas.net/book/the-book-2nd.html', kind: 'book', note: '강화학습의 표준 교과서. 무료로 읽을 수 있다.' },
    { title: 'Human-level control through deep reinforcement learning', by: 'Volodymyr Mnih 외', year: 2015, url: 'https://doi.org/10.1038/nature14236', kind: 'paper', note: 'DQN으로 아타리 게임을 배운 논문.' },
    { title: 'Lectures on Reinforcement Learning', by: 'David Silver, UCL', year: 2015, url: 'https://www.davidsilver.uk/teaching/', kind: 'course' }
  ],
  26: [
    { title: 'Simple statistical gradient-following algorithms for connectionist reinforcement learning', by: 'Ronald J. Williams', year: 1992, url: 'https://doi.org/10.1007/BF00992696', kind: 'paper', note: 'REINFORCE.' },
    { title: 'Proximal Policy Optimization Algorithms', by: 'John Schulman 외', year: 2017, url: 'https://arxiv.org/abs/1707.06347', kind: 'paper', note: 'PPO.' },
    { title: 'Spinning Up in Deep RL', by: 'OpenAI', url: 'https://spinningup.openai.com/', kind: 'course', note: '정책 경사 계열을 코드와 함께 설명한다.' }
  ],
  27: [
    { title: 'Hidden Technical Debt in Machine Learning Systems', by: 'D. Sculley 외', year: 2015, url: 'https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html', kind: 'paper', note: '모델 코드는 전체 시스템의 작은 일부라는 것을 보여 준 논문.' },
    { title: 'Rules of Machine Learning', by: 'Martin Zinkevich, Google', url: 'https://developers.google.com/machine-learning/guides/rules-of-ml', kind: 'article' },
    { title: 'Designing Machine Learning Systems', by: 'Chip Huyen', year: 2022, url: 'https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/', kind: 'book' }
  ],
  28: [
    { title: 'Dive into Deep Learning (한국어판)', by: 'Aston Zhang 외', url: 'https://ko.d2l.ai/', kind: 'book', note: '코드와 함께 읽는 딥러닝 교재의 한국어 번역.' },
    { title: 'Understanding Deep Learning', by: 'Simon J. D. Prince', year: 2023, url: UDL, kind: 'book', note: '이 책 다음에 읽기 좋은 무료 교재.' },
    { title: 'Practical Deep Learning for Coders', by: 'fast.ai', url: 'https://course.fast.ai/', kind: 'course', note: '실전 프로젝트 중심의 무료 강의.' },
    { title: 'Hugging Face LLM Course', by: 'Hugging Face', url: 'https://huggingface.co/learn/llm-course', kind: 'course', note: 'Transformer 모델을 직접 다뤄 보는 강의.' }
  ]
};

/** Keyed by code-lab number. */
export const labRefs: Record<number, Ref[]> = {
  1: [
    { title: 'CS231n: Neural Networks Part 1', by: 'Stanford', url: 'https://cs231n.github.io/neural-networks-1/', kind: 'course' },
    { title: 'Neural Networks and Deep Learning', by: 'Michael Nielsen', year: 2015, url: 'http://neuralnetworksanddeeplearning.com/', kind: 'book' }
  ],
  2: [{ title: 'PyTorch Tutorials: Learn the Basics', by: 'PyTorch', url: 'https://pytorch.org/tutorials/beginner/basics/intro.html', kind: 'docs' }],
  3: [
    { title: 'Transfer Learning for Computer Vision Tutorial', by: 'PyTorch', url: 'https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html', kind: 'docs' },
    { title: 'How transferable are features in deep neural networks?', by: 'Jason Yosinski 외', year: 2014, url: 'https://arxiv.org/abs/1411.1792', kind: 'paper' }
  ],
  4: [{ title: 'U-Net: Convolutional Networks for Biomedical Image Segmentation', by: 'Olaf Ronneberger 외', year: 2015, url: 'https://arxiv.org/abs/1505.04597', kind: 'paper' }],
  5: [
    { title: 'You Only Look Once: Unified, Real-Time Object Detection', by: 'Joseph Redmon 외', year: 2015, url: 'https://arxiv.org/abs/1506.02640', kind: 'paper' },
    { title: 'Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks', by: 'Shaoqing Ren 외', year: 2015, url: 'https://arxiv.org/abs/1506.01497', kind: 'paper' }
  ],
  6: [{ title: 'An Image is Worth 16x16 Words', by: 'Alexey Dosovitskiy 외', year: 2020, url: 'https://arxiv.org/abs/2010.11929', kind: 'paper' }],
  7: [
    { title: 'Neural Machine Translation of Rare Words with Subword Units', by: 'Rico Sennrich 외', year: 2015, url: 'https://arxiv.org/abs/1508.07909', kind: 'paper' },
    { title: 'minbpe', by: 'Andrej Karpathy', url: 'https://github.com/karpathy/minbpe', kind: 'code' }
  ],
  8: [
    { title: 'Attention Is All You Need', by: 'Ashish Vaswani 외', year: 2017, url: 'https://arxiv.org/abs/1706.03762', kind: 'paper' },
    { title: 'The Annotated Transformer', by: 'Harvard NLP', url: 'https://nlp.seas.harvard.edu/annotated-transformer/', kind: 'code' }
  ],
  9: [
    { title: 'Language Models are Unsupervised Multitask Learners', by: 'Alec Radford 외', year: 2019, url: 'https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf', kind: 'paper', note: 'GPT-2.' },
    { title: 'nanoGPT', by: 'Andrej Karpathy', url: 'https://github.com/karpathy/nanoGPT', kind: 'code' }
  ],
  10: [
    { title: 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding', by: 'Jacob Devlin 외', year: 2018, url: 'https://arxiv.org/abs/1810.04805', kind: 'paper' },
    { title: 'Hugging Face LLM Course: Fine-tuning a pretrained model', by: 'Hugging Face', url: 'https://huggingface.co/learn/llm-course/chapter3/1', kind: 'course' }
  ],
  11: [
    { title: 'LoRA: Low-Rank Adaptation of Large Language Models', by: 'Edward J. Hu 외', year: 2021, url: 'https://arxiv.org/abs/2106.09685', kind: 'paper' },
    { title: 'PEFT', by: 'Hugging Face', url: 'https://huggingface.co/docs/peft/index', kind: 'docs' }
  ],
  12: [{ title: 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks', by: 'Patrick Lewis 외', year: 2020, url: 'https://arxiv.org/abs/2005.11401', kind: 'paper' }],
  13: [{ title: 'ReAct: Synergizing Reasoning and Acting in Language Models', by: 'Shunyu Yao 외', year: 2022, url: 'https://arxiv.org/abs/2210.03629', kind: 'paper' }],
  14: [
    { title: 'Auto-Encoding Variational Bayes', by: 'Diederik P. Kingma, Max Welling', year: 2013, url: 'https://arxiv.org/abs/1312.6114', kind: 'paper' },
    { title: 'Generative Adversarial Networks', by: 'Ian Goodfellow 외', year: 2014, url: 'https://arxiv.org/abs/1406.2661', kind: 'paper' }
  ],
  15: [
    { title: 'Denoising Diffusion Probabilistic Models', by: 'Jonathan Ho 외', year: 2020, url: 'https://arxiv.org/abs/2006.11239', kind: 'paper' },
    { title: 'The Annotated Diffusion Model', by: 'Hugging Face', url: 'https://huggingface.co/blog/annotated-diffusion', kind: 'article' }
  ],
  16: [
    { title: 'Human-level control through deep reinforcement learning', by: 'Volodymyr Mnih 외', year: 2015, url: 'https://doi.org/10.1038/nature14236', kind: 'paper' },
    { title: 'Cart Pole', by: 'Gymnasium', url: 'https://gymnasium.farama.org/environments/classic_control/cart_pole/', kind: 'docs' }
  ],
  17: [
    { title: 'Proximal Policy Optimization Algorithms', by: 'John Schulman 외', year: 2017, url: 'https://arxiv.org/abs/1707.06347', kind: 'paper' },
    { title: 'The 37 Implementation Details of Proximal Policy Optimization', by: 'Shengyi Huang 외', year: 2022, url: 'https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/', kind: 'article' }
  ],
  18: [{ title: 'Learning Transferable Visual Models From Natural Language Supervision', by: 'Alec Radford 외', year: 2021, url: 'https://arxiv.org/abs/2103.00020', kind: 'paper', note: 'CLIP.' }],
  19: [
    { title: 'Robust Speech Recognition via Large-Scale Weak Supervision', by: 'Alec Radford 외', year: 2022, url: 'https://arxiv.org/abs/2212.04356', kind: 'paper' },
    { title: 'openai/whisper', by: 'OpenAI', url: 'https://github.com/openai/whisper', kind: 'code' }
  ],
  20: [{ title: 'FastAPI', by: 'Sebastián Ramírez', url: 'https://fastapi.tiangolo.com/', kind: 'docs' }]
};
