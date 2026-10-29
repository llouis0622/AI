import { useEffect, useMemo, useState, type CSSProperties, type ReactNode } from 'react';
import type { VizProps } from '../Viz';
import { Button, Controls, Note, Segmented, Stat } from '../core/ui';
import { useElementSize, usePrefersReducedMotion } from '../core/hooks';
import { useVizColors, withAlpha } from '../core/colors';
import { rng } from '../core/math';
import { loadProgress, onProgress, type ProgressState } from '@/lib/progress';
import { TRACKS, url } from '@/data/curriculum';

/*
 * "What to learn next" for chapter 28: a skill tree that grows from the reader's real progress
 * (code labs marked done) plus checklists kept in this browser, a sorting game for
 * "what changes vs what lasts", and a paper-reading guide built on the book's five questions.
 */
const SAVE_KEY = 'aiology:skilltree:v1';

const CH: [string, string][] = [
  ['00-intro', '여는 글'],
  ['01-what-is-ai', 'AI라는 문제'],
  ['02-first-learning-machine', '첫 학습 기계'],
  ['03-math-for-learning', '학습의 수학'],
  ['04-generalization', '일반화'],
  ['05-classification', '분류'],
  ['06-trees-and-ensembles', '트리와 앙상블'],
  ['07-unsupervised', '비지도 학습'],
  ['08-evaluation', '평가'],
  ['09-neural-networks', '신경망'],
  ['10-backpropagation', '역전파'],
  ['11-training-deep-nets', '깊은 신경망'],
  ['12-debugging-training', '학습 디버깅'],
  ['13-cnn', 'CNN'],
  ['14-vision-tasks', '비전 과제'],
  ['15-audio', '오디오'],
  ['16-tokens-and-embeddings', '토큰과 임베딩'],
  ['17-rnn-to-attention', '어텐션'],
  ['18-transformer', 'Transformer'],
  ['19-generative-models', '생성 모델'],
  ['20-diffusion', '확산 모델'],
  ['21-llm-pretraining', 'LLM 사전학습'],
  ['22-alignment', '정렬'],
  ['23-using-llms', 'LLM 활용'],
  ['24-agents', '에이전트'],
  ['25-reinforcement-learning', '강화학습'],
  ['26-policy-gradient', '정책 경사'],
  ['27-to-production', '배포와 운영'],
  ['28-keep-learning', '계속 공부하기']
];
const LABS: [string, string][] = [
  ['01-numpy-mlp-from-scratch', 'NumPy 신경망'],
  ['02-pytorch-training-pipeline', 'PyTorch 파이프라인'],
  ['03-transfer-learning', '전이학습'],
  ['04-unet-segmentation', 'U-Net'],
  ['05-object-detection', '객체 탐지'],
  ['06-vit-from-scratch', 'ViT'],
  ['07-bpe-tokenizer', 'BPE 토크나이저'],
  ['08-transformer-from-scratch', 'Transformer'],
  ['09-gpt-from-scratch', 'GPT'],
  ['10-bert-finetuning-hf', 'BERT 파인튜닝'],
  ['11-lora-finetuning', 'LoRA'],
  ['12-rag-system', 'RAG'],
  ['13-agent-from-scratch', '에이전트'],
  ['14-vae-gan', 'VAE·GAN'],
  ['15-ddpm-from-scratch', 'DDPM'],
  ['16-dqn-cartpole', 'DQN'],
  ['17-ppo-from-scratch', 'PPO'],
  ['18-clip-multimodal', 'CLIP'],
  ['19-whisper-audio', '음성 인식'],
  ['20-serving-fastapi', '모델 서빙']
];
const chHref = (n: number) => url(`/book/${CH[n][0]}/`);
const labHref = (n: number) => url(`/practice/${LABS[n - 1][0]}/`);

const STEP2: { text: string; ch?: number }[] = [
  { text: '📝 문제 정하기: 작업·지표·데이터 세 질문', ch: 1 },
  { text: '📏 기준선부터 세우기', ch: 8 },
  { text: '🛠️ 모델 만들고 학습시키기 (막히면 처방전)', ch: 12 },
  { text: '🔍 틀린 사례를 직접 읽기', ch: 8 },
  { text: '🚀 배포까지 해 보기', ch: 27 },
  { text: '✍️ 만든 과정을 글로 남기기' }
];
const STEP3 = ['📜 그 분야의 고전 논문 읽기', '🔁 최신 논문 하나 작게 재현하기', '💬 커뮤니티에서 토론하기'];
const STEP4 = ['👥 스터디 열기', '✍️ 배운 것을 글로 쓰기', '🙋 다른 사람의 질문에 답하기'];

type FieldKey = 'vision' | 'language' | 'gen' | 'rl' | 'sys';
const FIELDS: Record<FieldKey, { label: string; chapters: number[]; labs: number[]; papers: [string, string][] }> = {
  vision: { label: '👀 비전', chapters: [13, 14], labs: [4, 5, 6, 18], papers: [['AlexNet (2012)', '딥러닝 붐을 연 이미지 분류'], ['ResNet (2015)', '지름길로 152층까지'], ['ViT (2020)', '이미지를 조각 토큰으로']] },
  language: { label: '💬 언어', chapters: [16, 17, 18, 21, 22, 23, 24], labs: [7, 8, 9, 10, 11, 12, 13], papers: [['Attention Is All You Need (2017)', 'Transformer의 탄생'], ['BERT (2018)', '빈칸 채우기로 사전학습'], ['GPT-3 (2020)', '규모가 키운 퓨샷 능력']] },
  gen: { label: '🎨 생성', chapters: [19, 20], labs: [14, 15], papers: [['VAE (2013)', '잠재 공간에서 뽑아 그리기'], ['GAN (2014)', '위조범과 감별사의 대결'], ['DDPM (2020)', '노이즈를 걷어 내며 그리기']] },
  rl: { label: '🎮 강화학습', chapters: [25, 26], labs: [16, 17], papers: [['DQN (2015)', '화면만 보고 아타리 게임 정복'], ['AlphaGo (2016)', '최정상 바둑 기사를 이긴 AI'], ['PPO (2017)', '안정적인 정책 경사']] },
  sys: { label: '🚀 시스템', chapters: [8, 12, 27], labs: [2, 20], papers: [['Hidden Technical Debt in ML Systems (2015)', '모델은 시스템의 작은 조각'], ['FlashAttention (2022)', '메모리를 아끼는 어텐션 계산'], ['PagedAttention (2023)', 'KV 캐시를 페이지로 관리']] }
};

const CONCEPTS: [string, number][] = [
  ['경사하강법', 2],
  ['과적합', 4],
  ['정밀도와 재현율', 8],
  ['역전파', 10],
  ['합성곱', 13],
  ['임베딩', 16],
  ['어텐션', 17],
  ['Transformer', 18],
  ['확산 모델', 20],
  ['RLHF', 22],
  ['RAG', 23],
  ['Q-학습', 25],
  ['데이터 드리프트', 27]
];

const CARDS: { text: string; stay: boolean; why: string }[] = [
  { text: '학습은 손실을 줄이는 것', stay: true, why: '2장의 직선부터 GPT까지, 모두 손실을 줄이며 배워요.' },
  { text: '진짜 목표는 일반화', stay: true, why: '외운 문제가 아니라 처음 보는 데이터에서 잘해야 해요 (4장).' },
  { text: '최고 성능 모델의 이름', stay: false, why: '몇 달마다 새 이름이 1등을 해요. 외울 필요 없어요.' },
  { text: '데이터가 모델을 만든다', stay: true, why: '어떤 모델이든 데이터에 없는 것은 배우지 못해요.' },
  { text: '프레임워크 함수 이름(API)', stay: false, why: '버전마다 바뀌어요. 필요할 때 문서를 찾아보면 돼요.' },
  { text: '평가 없는 성적은 소문', stay: true, why: '재 보지 않은 성능은 믿지 않아요 (8장).' },
  { text: '벤치마크 순위표', stay: false, why: '순위는 매주 바뀌어요.' },
  { text: '곱셈의 사슬은 소실하고, 지름길이 살린다', stay: true, why: '깊은 신경망에서 신호가 사라지는 이유와 잔차 연결이 살리는 이유 (10~11장).' },
  { text: '서빙 엔진', stay: false, why: '더 빠른 엔진이 계속 나와요. 배칭·KV 캐시 같은 원리는 그대로!' },
  { text: '표현이 바뀌면 도구가 옮겨 간다', stay: true, why: '소리도 그림도 토큰으로 바꾸면 Transformer를 쓸 수 있어요.' },
  { text: '이번 달의 유행 기법', stay: false, why: '대부분 오래된 원리의 새 조합이에요.' }
];

const QUESTIONS = ['🔤 표현 — 입력을 어떤 숫자로 바꿨나?', '🏗️ 구조 — 어떤 아키텍처인가?', '🎯 목표 — 무엇을 맞히도록 배웠나?', '📚 데이터 — 무엇으로 배웠나?', '📏 평가 — 무엇으로 잘한다고 했나?'];
const PAPERS: Record<string, { label: string; answers: string[] }> = {
  transformer: {
    label: 'Attention Is All You Need (2017)',
    answers: ['단어 조각(토큰) 임베딩 + 위치 정보 (16·18장)', '순환 없이 어텐션만 쌓은 인코더-디코더 Transformer', '번역: 다음 토큰 맞히기 (교차 엔트로피)', 'WMT 2014 영어→독일어 약 450만 문장 쌍, 영어→프랑스어', 'BLEU 점수 (영→독 28.4) + 학습 비용 비교']
  },
  ddpm: {
    label: 'DDPM (2020)',
    answers: ['이미지의 픽셀 값', '노이즈를 예측하는 U-Net (20장)', '이미지에 더해진 노이즈 맞히기 (MSE)', 'CIFAR-10, CelebA-HQ, LSUN 사진들', 'FID·IS 점수 (CIFAR-10에서 FID 3.17)']
  },
  gpt3: {
    label: 'GPT-3 (2020)',
    answers: ['BPE 토큰 (16장)', '디코더만 있는 Transformer, 파라미터 1750억 개', '다음 토큰 맞히기 (21장)', '걸러 낸 웹 문서·책·위키백과 등 약 3000억 토큰', '수십 개 과제를 예시 몇 개만 주고(퓨샷) 풀게 함']
  }
};
const READ_STEPS: { min: string; w: number; title: string; body: string; checks?: string[] }[] = [
  { min: '5분', w: 5, title: '초록 + 그림 1 + 결론', body: '무엇을 주장하나? 다섯 질문으로 정리해 보고, 흥미가 없으면 덮어도 돼요. 모든 논문을 읽을 필요는 없어요.' },
  { min: '30분', w: 30, title: '방법: 수식은 말로 번역하며', body: '모든 기호를 나온 자리에서 한국어로 읽어요 (3장의 습관). 막히면 숫자 두세 개로 아주 작은 예를 직접 계산해 봐요.' },
  { min: '20분', w: 20, title: '실험: 의심하며', body: '좋은 논문일수록 아래 질문에 정직하게 답해 둬요.', checks: ['기준선은 공정한가? (8장)', '비교 모델도 같은 정성으로 튜닝했나?', '데이터 누수 가능성은 없나?', '시드를 바꿔 여러 번 돌렸나?'] },
  { min: '선택', w: 14, title: '직접 재현', body: '코드가 있으면 돌려 보고, 없으면 핵심만 작게 구현해요. 코드랩이 바로 이 훈련이었어요. 최고의 공부!' }
];

interface Saved {
  s2: boolean[];
  field: FieldKey | '';
  s3: boolean[];
  s4: boolean[];
  wins: number;
  todo: number[];
}
const fresh = (): Saved => ({ s2: STEP2.map(() => false), field: '', s3: STEP3.map(() => false), s4: STEP4.map(() => false), wins: 0, todo: [] });

function loadSaved(): Saved {
  try {
    const raw = localStorage.getItem(SAVE_KEY);
    if (!raw) return fresh();
    const p = JSON.parse(raw) as Partial<Saved>;
    const base = fresh();
    const bools = (v: unknown, n: number) => (Array.isArray(v) ? Array.from({ length: n }, (_, i) => v[i] === true) : new Array<boolean>(n).fill(false));
    return {
      s2: bools(p.s2, base.s2.length),
      field: p.field && p.field in FIELDS ? p.field : '',
      s3: bools(p.s3, base.s3.length),
      s4: bools(p.s4, base.s4.length),
      wins: typeof p.wins === 'number' ? p.wins : 0,
      todo: Array.isArray(p.todo) ? p.todo.filter((v): v is number => typeof v === 'number' && v >= 0 && v < CONCEPTS.length) : []
    };
  } catch {
    return fresh();
  }
}

const chip = (bg: string, fg: string, border: string): CSSProperties => ({
  display: 'inline-flex',
  alignItems: 'center',
  gap: 4,
  padding: '4px 10px',
  borderRadius: 999,
  fontSize: 13,
  fontWeight: 600,
  background: bg,
  color: fg,
  border: `1.5px solid ${border}`,
  textDecoration: 'none',
  lineHeight: 1.35
});

function Check({ checked, onChange, children }: { checked: boolean; onChange: (v: boolean) => void; children: ReactNode }) {
  return (
    <label style={{ display: 'flex', gap: 8, alignItems: 'flex-start', fontSize: 14, cursor: 'pointer', lineHeight: 1.45 }}>
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} style={{ marginTop: 3, width: 16, height: 16, accentColor: 'var(--brand)', flex: 'none' }} />
      <span style={{ textDecoration: checked ? 'line-through' : undefined, opacity: checked ? 0.7 : 1 }}>{children}</span>
    </label>
  );
}

export default function SkillTree({ expanded }: VizProps) {
  const c = useVizColors();
  const reduce = usePrefersReducedMotion();
  const [treeBox, { width: treeW }] = useElementSize<HTMLDivElement>();
  // the tree is drawn in a fixed viewBox; enlarge its labels on phones so they stay ≥ 11px
  const fk = treeW > 0 && treeW < 480 ? 1.3 : 1;
  const [tab, setTab] = useState<'tree' | 'sort' | 'paper'>('tree');
  const [prog, setProg] = useState<ProgressState>({ chapters: {}, labs: {}, viz: {} });
  const [saved, setSaved] = useState<Saved>(fresh);
  const [loaded, setLoaded] = useState(false);
  const [stage, setStage] = useState<number | null>(null);

  useEffect(() => {
    setProg(loadProgress());
    setSaved(loadSaved());
    setLoaded(true);
    return onProgress(() => setProg(loadProgress()));
  }, []);
  useEffect(() => {
    if (!loaded) return;
    try {
      localStorage.setItem(SAVE_KEY, JSON.stringify(saved));
    } catch {
      /* storage blocked: the tree still works for this visit */
    }
  }, [saved, loaded]);
  const patch = (p: Partial<Saved>) => setSaved((s) => ({ ...s, ...p }));

  // ---- tree data ----
  const labDone = (n: number) => !!prog.labs[n]?.done;
  const labSeen = (n: number) => !!prog.labs[n]?.visited;
  const chDone = (n: number) => !!prog.chapters[n]?.done;
  const s1 = LABS.map((_, i) => (labDone(i + 1) ? 1 : labSeen(i + 1) ? 0.5 : 0));
  const s3leaves = [saved.field ? 1 : 0, ...saved.s3.map((v) => (v ? 1 : 0))];
  const s4leaves = [...saved.s4.map((v) => (v ? 1 : 0)), saved.wins >= 3 ? 1 : 0];
  const stages = [
    { name: '코드랩 완주', color: c.series[1], leaves: s1, done: s1.filter((v) => v === 1).length },
    { name: '내 프로젝트', color: c.series[3], leaves: saved.s2.map((v) => (v ? 1 : 0)), done: saved.s2.filter(Boolean).length },
    { name: '한 분야 깊이', color: c.series[0], leaves: s3leaves, done: s3leaves.filter((v) => v === 1).length },
    { name: '가르치기', color: c.series[5], leaves: s4leaves, done: s4leaves.filter((v) => v === 1).length }
  ];
  const allLeaves = stages.reduce((a, s) => a + s.leaves.length, 0);
  const lit = stages.reduce((a, s) => a + s.done, 0);
  const firstOpen = stages.findIndex((s) => s.done < s.leaves.length);
  const active = stage ?? (firstOpen < 0 ? 3 : firstOpen);
  const complete = firstOpen < 0;
  const chaptersRead = Object.keys(prog.chapters).filter((k) => prog.chapters[k]?.done).length;
  const nextLab = LABS.findIndex((_, i) => !labDone(i + 1)) + 1;

  // ---- sort game ----
  const [round, setRound] = useState(0);
  const order = useMemo(() => {
    const r = rng(7 + round * 31);
    const idx = CARDS.map((_, i) => i);
    for (let i = idx.length - 1; i > 0; i--) {
      const j = Math.floor(r.next() * (i + 1));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    return idx;
  }, [round]);
  const [picks, setPicks] = useState<Record<number, boolean>>({});
  const [pos, setPos] = useState(0);
  const cur = order[pos];
  const answered = cur !== undefined && cur in picks;
  const sortDone = pos >= order.length;
  const score = Object.entries(picks).filter(([k, v]) => CARDS[Number(k)].stay === v).length;

  // ---- paper tab ----
  const [paper, setPaper] = useState<keyof typeof PAPERS>('transformer');
  const [flipped, setFlipped] = useState<boolean[]>(QUESTIONS.map(() => false));
  const [readStep, setReadStep] = useState(0);
  const [doubt, setDoubt] = useState<boolean[]>([false, false, false, false]);

  // ---- explain challenge ----
  const [challenge, setChallenge] = useState<number | null>(null);
  const [chSeed, setChSeed] = useState(1);
  const drawConcept = () => {
    const r = rng(chSeed * 97 + 13);
    setChSeed((s) => s + 1);
    let k = Math.floor(r.next() * CONCEPTS.length);
    if (k === challenge) k = (k + 1) % CONCEPTS.length;
    setChallenge(k);
  };

  const trunk = c.theme === 'dark' ? '#b98d72' : '#9b6a4f';
  const W = 420;
  const H = 236;
  const cx = 160;
  const levels = [
    { y: 192, w: 135 },
    { y: 148, w: 110 },
    { y: 106, w: 86 },
    { y: 66, w: 60 }
  ];

  const tree = (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: expanded ? 640 : 520, display: 'block', margin: '0 auto' }} role="img" aria-label={`실력 나무: 잎 ${allLeaves}개 중 ${lit}개가 켜졌어요`}>
      <ellipse cx={cx} cy={H - 16} rx={128} ry={10} fill={withAlpha(c.pos, 0.16)} />
      <path d={`M${cx - 11},${H - 18} C${cx - 7},170 ${cx - 5},100 ${cx - 3},42 L${cx + 3},42 C${cx + 5},100 ${cx + 7},170 ${cx + 11},${H - 18} Z`} fill={trunk} />
      {stages.map((s, si) => {
        const { y, w } = levels[si];
        const on = si === active;
        const els: ReactNode[] = [];
        const sides = [-1, 1];
        const perSide = [Math.ceil(s.leaves.length / 2), Math.floor(s.leaves.length / 2)];
        let li = 0;
        sides.forEach((side, k) => {
          const p0 = [cx, y];
          const p1 = [cx + side * w * 0.55, y - 2];
          const p2 = [cx + side * w, y - 24];
          els.push(<path key={`b${k}`} d={`M${p0[0]},${p0[1]} Q${p1[0]},${p1[1]} ${p2[0]},${p2[1]}`} fill="none" stroke={trunk} strokeWidth={4.2 - si * 0.6} strokeLinecap="round" />);
          const m = perSide[k];
          for (let j = 0; j < m; j++, li++) {
            const t = 0.24 + (0.76 * (j + 0.5)) / m;
            const bx = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0];
            const by = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1];
            const tx = 2 * (1 - t) * (p1[0] - p0[0]) + 2 * t * (p2[0] - p1[0]);
            const ty = 2 * (1 - t) * (p1[1] - p0[1]) + 2 * t * (p2[1] - p1[1]);
            const len = Math.hypot(tx, ty) || 1;
            const up = j % 2 === 0 ? -1 : 1;
            const lx = bx + (-ty / len) * 7 * up * side;
            const ly = by + (tx / len) * 7 * up * side;
            const ang = (Math.atan2(ty, tx) * 180) / Math.PI + (up > 0 ? 38 : -38);
            const v = s.leaves[li];
            els.push(
              <ellipse key={`l${k}-${j}`} cx={lx} cy={ly} rx={6.4} ry={3.6} transform={`rotate(${ang.toFixed(1)} ${lx.toFixed(1)} ${ly.toFixed(1)})`} fill={v === 1 ? s.color : v > 0 ? withAlpha(s.color, 0.35) : 'none'} stroke={s.color} strokeWidth={1.3} opacity={v > 0 ? 1 : 0.5}>
                {!reduce && v === 1 && <animate attributeName="ry" values="3.6;4.2;3.6" dur={`${2.4 + ((li * 7) % 10) / 10}s`} repeatCount="indefinite" />}
              </ellipse>
            );
          }
        });
        const full = s.done === s.leaves.length;
        return (
          <g key={si} role="button" tabIndex={0} aria-label={`${si + 1}단계 ${s.name}: ${s.done}/${s.leaves.length}`} onClick={() => setStage(si)} onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), setStage(si))} style={{ cursor: 'pointer', outline: 'none' }}>
            <rect x={cx - w - 12} y={y - 40} width={w * 2 + 120} height={42} fill="transparent" />
            {on && <rect x={cx - w - 14} y={y - 38} width={w * 2 + 28} height={44} rx={14} fill={withAlpha(s.color, 0.1)} stroke={withAlpha(s.color, 0.5)} strokeDasharray="4 4" />}
            {els}
            <circle cx={cx} cy={y} r={9.5} fill={full ? s.color : 'var(--surface-solid, #fff)'} stroke={s.color} strokeWidth={2} />
            <text x={cx} y={y + 4} textAnchor="middle" fontSize={11 * Math.min(fk, 1.1)} fontWeight={800} fill={full ? '#ffffff' : s.color}>
              {si + 1}
            </text>
            <text x={cx + w + 10} y={y - 27} fontSize={12 * fk} fontWeight={800} fill="var(--viz-text)">
              {s.name}
            </text>
            <text x={cx + w + 10} y={y - 27 + 15 * fk} fontSize={11 * fk} fill="var(--viz-text)" opacity={0.75}>
              {full ? '완성 ✓' : `${s.done} / ${s.leaves.length}`}
            </text>
          </g>
        );
      })}
      {complete ? (
        <text x={cx} y={34} textAnchor="middle" fontSize={24}>
          ⭐
        </text>
      ) : (
        <circle cx={cx} cy={36} r={5} fill={withAlpha(c.series[5], 0.35)} stroke={c.series[5]} strokeWidth={1.2} />
      )}
    </svg>
  );

  const stagePanel = (() => {
    if (active === 0)
      return (
        <div style={{ display: 'grid', gap: 10 }}>
          <div style={{ fontSize: 14 }}>
            <strong>1단계: 코드랩을 전부 완주하라.</strong> 코드랩 맨 아래의 <strong>"다 읽었어요 ✓"</strong>를 누르면 여기 잎이 돋아요. (👀 = 열어 봄, ✓ = 완주)
          </div>
          {TRACKS.map((t) => (
            <div key={t.id} style={{ display: 'flex', flexWrap: 'wrap', gap: 6, alignItems: 'center' }}>
              <span style={{ fontSize: 13, fontWeight: 800, minWidth: 128 }}>
                {t.emoji} {t.title}
              </span>
              {t.labs.map((n) => {
                const d = labDone(n);
                const seen = labSeen(n);
                return (
                  <a key={n} href={labHref(n)} style={chip(d ? withAlpha(c.series[1], 0.2) : 'transparent', 'var(--viz-text)', d ? c.series[1] : withAlpha(c.series[1], 0.45))}>
                    {d ? '✓' : seen ? '👀' : '○'} {String(n).padStart(2, '0')} {LABS[n - 1][1]}
                  </a>
                );
              })}
            </div>
          ))}
        </div>
      );
    if (active === 1)
      return (
        <div style={{ display: 'grid', gap: 8 }}>
          <div style={{ fontSize: 14 }}>
            <strong>2단계: 자기 문제를 하나 골라 끝까지 만들어라.</strong> 완주한 프로젝트 하나가 튜토리얼 백 개보다 많이 가르쳐요.
          </div>
          {STEP2.map((st, i) => (
            <Check key={i} checked={saved.s2[i]} onChange={(v) => patch({ s2: saved.s2.map((x, k) => (k === i ? v : x)) })}>
              {st.text}
              {st.ch !== undefined && (
                <>
                  {' '}
                  <a href={chHref(st.ch)} onClick={(e) => e.stopPropagation()} style={{ fontSize: 12 }}>
                    ({st.ch}장)
                  </a>
                </>
              )}
            </Check>
          ))}
        </div>
      );
    if (active === 2) {
      const f = saved.field ? FIELDS[saved.field] : null;
      return (
        <div style={{ display: 'grid', gap: 10 }}>
          <div style={{ fontSize: 14 }}>
            <strong>3단계: 한 분야를 깊이 파라.</strong> 깊이 하나가 생기면 다른 영토로 옮기기는 훨씬 쉬워요 — 원리는 공통이니까요. 어디를 파 볼까요?
          </div>
          <Segmented options={(Object.keys(FIELDS) as FieldKey[]).map((k) => ({ value: k, label: FIELDS[k].label }))} value={saved.field as FieldKey} onChange={(v) => patch({ field: v })} />
          {f && (
            <div style={{ display: 'grid', gap: 8 }}>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, alignItems: 'center' }}>
                <span style={{ fontSize: 13, fontWeight: 800, minWidth: 64 }}>📖 다시 볼 장</span>
                {f.chapters.map((n) => (
                  <a key={n} href={chHref(n)} style={chip(chDone(n) ? withAlpha(c.series[0], 0.18) : 'transparent', 'var(--viz-text)', withAlpha(c.series[0], chDone(n) ? 1 : 0.45))}>
                    {chDone(n) ? '✓' : '○'} {n}장 {CH[n][1]}
                  </a>
                ))}
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, alignItems: 'center' }}>
                <span style={{ fontSize: 13, fontWeight: 800, minWidth: 64 }}>💻 코드랩</span>
                {f.labs.map((n) => (
                  <a key={n} href={labHref(n)} style={chip(labDone(n) ? withAlpha(c.series[1], 0.2) : 'transparent', 'var(--viz-text)', withAlpha(c.series[1], labDone(n) ? 1 : 0.45))}>
                    {labDone(n) ? '✓' : '○'} {String(n).padStart(2, '0')} {LABS[n - 1][1]}
                  </a>
                ))}
              </div>
              <div style={{ display: 'grid', gap: 4 }}>
                <span style={{ fontSize: 13, fontWeight: 800 }}>📜 오래 살아남은 논문부터</span>
                {f.papers.map(([name, what]) => (
                  <div key={name} style={{ fontSize: 13 }}>
                    <strong>{name}</strong> — {what}
                  </div>
                ))}
              </div>
              {STEP3.map((t, i) => (
                <Check key={i} checked={saved.s3[i]} onChange={(v) => patch({ s3: saved.s3.map((x, k) => (k === i ? v : x)) })}>
                  {t}
                </Check>
              ))}
            </div>
          )}
        </div>
      );
    }
    return (
      <div style={{ display: 'grid', gap: 10 }}>
        <div style={{ fontSize: 14 }}>
          <strong>4단계: 가르쳐라.</strong> 설명하다 막히는 곳이 바로 다음 공부 목록이에요.
        </div>
        {STEP4.map((t, i) => (
          <Check key={i} checked={saved.s4[i]} onChange={(v) => patch({ s4: saved.s4.map((x, k) => (k === i ? v : x)) })}>
            {t}
          </Check>
        ))}
        <div style={{ border: `1.5px dashed ${withAlpha(c.series[5], 0.6)}`, borderRadius: 14, padding: 12, display: 'grid', gap: 8 }}>
          <div style={{ fontSize: 14, fontWeight: 800 }}>
            🎲 30초 설명 챌린지 <span style={{ fontWeight: 600, opacity: 0.75 }}>(성공 {Math.min(saved.wins, 3)}/3)</span>
          </div>
          {challenge === null ? (
            <div style={{ fontSize: 14 }}>개념 하나를 뽑아서, 친구에게 말하듯 30초 안에 설명해 보세요.</div>
          ) : (
            <div style={{ fontSize: 15 }}>
              <strong style={{ fontSize: 20 }}>"{CONCEPTS[challenge][0]}"</strong>을(를) 설명해 보세요! <a href={chHref(CONCEPTS[challenge][1])}>({CONCEPTS[challenge][1]}장)</a>
            </div>
          )}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            <Button size="sm" variant="primary" onClick={drawConcept}>
              {challenge === null ? '🎲 개념 뽑기' : '🎲 다른 개념'}
            </Button>
            {challenge !== null && (
              <>
                <Button size="sm" onClick={() => (patch({ wins: saved.wins + 1, todo: saved.todo.filter((t) => t !== challenge) }), setChallenge(null))}>
                  😎 설명했어요
                </Button>
                <Button size="sm" onClick={() => (patch({ todo: saved.todo.includes(challenge) ? saved.todo : [...saved.todo, challenge] }), setChallenge(null))}>
                  😵 막혔어요
                </Button>
              </>
            )}
          </div>
          {saved.todo.length > 0 && (
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, alignItems: 'center' }}>
              <span style={{ fontSize: 13, fontWeight: 800 }}>📌 다음 공부 목록</span>
              {saved.todo.map((k) => (
                <a key={k} href={chHref(CONCEPTS[k][1])} style={chip(withAlpha(c.series[5], 0.14), 'var(--viz-text)', withAlpha(c.series[5], 0.6))}>
                  {CONCEPTS[k][0]} · {CONCEPTS[k][1]}장
                </a>
              ))}
            </div>
          )}
        </div>
      </div>
    );
  })();

  let note: { t: ReactNode; mood: 'smile' | 'happy' | 'think' | 'wow' | 'cheer' | 'wink'; tone?: 'good' | 'warn' };
  if (tab === 'tree') {
    if (complete) note = { t: <>🎉 실력 나무 완성! 이제 당신이 누군가의 나무에 물을 줄 차례예요. 그래도 배움은 계속돼요 — 기계처럼, 경험이 쌓일수록 나아지면서.</>, mood: 'cheer', tone: 'good' };
    else if (firstOpen === 0)
      note = {
        t: (
          <>
            가지를 누르면 그 단계의 할 일이 보여요. 아래부터 차근차근! 다음 추천: <a href={labHref(nextLab)}>코드랩 {String(nextLab).padStart(2, '0')} 「{LABS[nextLab - 1][1]}」</a>
          </>
        ),
        mood: 'smile'
      };
    else note = { t: <>{stages[firstOpen].name} 단계를 키울 차례예요. 체크한 내용은 이 브라우저에만 저장돼요. 순서대로 안 해도 괜찮아요 — 가지를 눌러 골라 보세요.</>, mood: 'happy' };
  } else if (tab === 'sort') {
    if (sortDone) note = { t: <>{CARDS.length}장 중 <strong>{score}장</strong> 정답! 변하지 않는 원리는 <strong>이해</strong>하고, 변하는 표면은 필요할 때 <strong>찾아보면</strong> 돼요. 새것이 나오면 "아, 그 원리의 변주구나!"라고 말할 수 있게요.</>, mood: score >= CARDS.length - 2 ? 'cheer' : 'think', tone: 'good' };
    else if (answered) {
      const card = CARDS[cur];
      const ok = picks[cur] === card.stay;
      note = { t: <>{ok ? '정답! ' : '아쉬워요. '}"{card.text}"은(는) <strong>{card.stay ? '변하지 않는 원리' : '빠르게 변하는 표면'}</strong>이에요. {card.why}</>, mood: ok ? 'happy' : 'think', tone: ok ? 'good' : 'warn' };
    } else note = { t: <>이 분야는 빨라요. 하지만 빠르게 변하는 건 <strong>표면</strong>뿐이에요. 카드를 보고, 10년 뒤에도 참일지 골라 보세요!</>, mood: 'wink' };
  } else {
    const n = flipped.filter(Boolean).length;
    note =
      n === QUESTIONS.length
        ? { t: <>다섯 질문에 모두 답이 나왔어요 — 이해한 거예요! 🎉 새 모델 소식을 들으면 이 다섯 칸부터 채워 보세요. 안 채워지면 아직 "발표 자료"일 뿐이에요.</>, mood: 'cheer', tone: 'good' }
        : { t: <>논문은 무섭지 않아요 — 읽는 순서만 알면요. 아래 <strong>다섯 질문 카드</strong>를 눌러 뒤집어 보세요 ({n}/5).</>, mood: 'smile' };
  }

  const card = cur !== undefined ? CARDS[cur] : null;
  const binList = (stay: boolean) =>
    order
      .slice(0, pos + (answered ? 1 : 0))
      .filter((i) => i in picks && picks[i] === stay)
      .map((i) => (
        <span key={i} style={chip(CARDS[i].stay === stay ? withAlpha(stay ? c.series[0] : c.series[3], 0.16) : withAlpha(c.neg, 0.14), 'var(--viz-text)', CARDS[i].stay === stay ? withAlpha(stay ? c.series[0] : c.series[3], 0.6) : c.neg)}>
          {CARDS[i].stay === stay ? '✓' : '✗'} {CARDS[i].text}
        </span>
      ));

  return (
    <div>
      <div className="vz-pad" style={{ display: 'grid', gap: 14 }}>
        <Segmented
          options={[
            { value: 'tree', label: '🌳 실력 나무' },
            { value: 'sort', label: '🔄 변하는 것 vs 안 변하는 것' },
            { value: 'paper', label: '📄 논문 읽는 법' }
          ]}
          value={tab}
          onChange={setTab}
        />
        <Note mood={note.mood} tone={note.tone}>
          {note.t}
        </Note>
        {tab === 'tree' && (
          <>
            <div ref={treeBox}>{tree}</div>
            <Segmented options={stages.map((s, i) => ({ value: i, label: `${i + 1}. ${s.name}` }))} value={active} onChange={(v) => setStage(v)} />
            <div style={{ border: `1.5px solid ${withAlpha(stages[active].color, 0.5)}`, background: withAlpha(stages[active].color, 0.06), borderRadius: 16, padding: 14 }}>{stagePanel}</div>
            <div className="vz-stats">
              <Stat label="나무 전체" value={`${Math.round((lit / allLeaves) * 100)}%`} sub={`잎 ${lit} / ${allLeaves}`} tone={complete ? 'good' : undefined} />
              <Stat label="완주한 코드랩" value={`${stages[0].done} / 20`} />
              <Stat label="다 읽은 장" value={`${chaptersRead} / 28`} sub="각 장 끝 '다 읽었어요 ✓'" />
              <Stat label="고른 분야" value={saved.field ? FIELDS[saved.field].label : '아직'} />
            </div>
          </>
        )}
        {tab === 'sort' && (
          <>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 10 }}>
              <div style={{ border: `1.5px solid ${withAlpha(c.series[0], 0.5)}`, borderRadius: 14, padding: 10, minHeight: 90 }}>
                <div style={{ fontWeight: 800, fontSize: 14, marginBottom: 6 }}>🪨 변하지 않는 것 (원리)</div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>{binList(true)}</div>
              </div>
              <div style={{ border: `1.5px solid ${withAlpha(c.series[3], 0.5)}`, borderRadius: 14, padding: 10, minHeight: 90 }}>
                <div style={{ fontWeight: 800, fontSize: 14, marginBottom: 6 }}>🌊 변하는 것 (표면)</div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>{binList(false)}</div>
              </div>
            </div>
            {!sortDone && card && (
              <div style={{ display: 'grid', justifyItems: 'center', gap: 10, padding: '6px 0' }}>
                <div style={{ fontSize: 12, opacity: 0.7 }}>
                  카드 {pos + 1} / {order.length}
                </div>
                <div key={cur} style={{ minWidth: 240, maxWidth: 420, padding: '18px 22px', borderRadius: 18, textAlign: 'center', fontSize: 19, fontWeight: 800, background: 'var(--surface-solid, #fff)', border: `2px solid ${answered ? (picks[cur] === card.stay ? c.pos : c.neg) : withAlpha(c.text, 0.2)}`, boxShadow: 'var(--shadow-sm)', animation: 'vz-note-in .35s ease' }}>
                  {card.text}
                </div>
                {!answered ? (
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center' }}>
                    <Button onClick={() => setPicks((p) => ({ ...p, [cur]: true }))}>🪨 안 변해요 (원리)</Button>
                    <Button onClick={() => setPicks((p) => ({ ...p, [cur]: false }))}>🌊 변해요 (표면)</Button>
                  </div>
                ) : (
                  <Button variant="primary" onClick={() => setPos((p) => p + 1)}>
                    {pos + 1 < order.length ? '다음 카드 →' : '결과 보기 🎉'}
                  </Button>
                )}
              </div>
            )}
            {sortDone && (
              <div style={{ display: 'flex', justifyContent: 'center' }}>
                <Button
                  variant="primary"
                  onClick={() => {
                    setRound((r) => r + 1);
                    setPicks({});
                    setPos(0);
                  }}
                >
                  ↺ 다시 하기
                </Button>
              </div>
            )}
          </>
        )}
        {tab === 'paper' && (
          <>
            <div>
              <div className="vz-panel-title">읽는 순서 (눌러 보세요)</div>
              <div style={{ display: 'flex', gap: 4 }} role="tablist">
                {READ_STEPS.map((s, i) => (
                  <button
                    key={i}
                    type="button"
                    role="tab"
                    aria-selected={readStep === i}
                    onClick={() => setReadStep(i)}
                    style={{ flex: `${s.w} 1 0`, minWidth: 58, border: `2px solid ${readStep === i ? c.series[i] : 'transparent'}`, borderRadius: 10, cursor: 'pointer', padding: '9px 6px', fontWeight: 800, fontSize: 13, color: 'var(--viz-text)', background: withAlpha(c.series[i], readStep === i ? 0.38 : 0.14), transition: 'background .2s' }}
                  >
                    {i + 1}. {s.min}
                  </button>
                ))}
              </div>
              <div style={{ marginTop: 10, padding: 12, borderRadius: 14, background: withAlpha(c.series[readStep], 0.08), border: `1.5px solid ${withAlpha(c.series[readStep], 0.4)}`, display: 'grid', gap: 8 }}>
                <div style={{ fontWeight: 800, fontSize: 15 }}>
                  {readStep + 1}단계 · {READ_STEPS[readStep].title} <span style={{ opacity: 0.7, fontWeight: 600 }}>({READ_STEPS[readStep].min})</span>
                </div>
                <div style={{ fontSize: 14 }}>{READ_STEPS[readStep].body}</div>
                {READ_STEPS[readStep].checks?.map((q, i) => (
                  <Check key={i} checked={doubt[i]} onChange={(v) => setDoubt((d) => d.map((x, k) => (k === i ? v : x)))}>
                    🧐 {q}
                  </Check>
                ))}
              </div>
            </div>
            <div style={{ display: 'grid', gap: 8 }}>
              <div className="vz-panel-title">다섯 질문으로 논문 정리하기</div>
              <Segmented
                options={(Object.keys(PAPERS) as (keyof typeof PAPERS)[]).map((k) => ({ value: k, label: PAPERS[k].label }))}
                value={paper}
                onChange={(v) => {
                  setPaper(v);
                  setFlipped(QUESTIONS.map(() => false));
                }}
              />
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: 8 }}>
                {QUESTIONS.map((q, i) => (
                  <button
                    key={`${paper}-${i}`}
                    type="button"
                    onClick={() => setFlipped((f) => f.map((x, k) => (k === i ? !x : x)))}
                    aria-pressed={flipped[i]}
                    style={{ textAlign: 'left', cursor: 'pointer', borderRadius: 14, padding: 12, minHeight: 84, fontSize: 13.5, lineHeight: 1.45, color: 'var(--viz-text)', border: `1.5px solid ${withAlpha(c.series[i], flipped[i] ? 0.9 : 0.45)}`, background: flipped[i] ? withAlpha(c.series[i], 0.14) : 'var(--surface-solid, #fff)', transition: 'background .25s' }}
                  >
                    <div style={{ fontWeight: 800, marginBottom: 4 }}>{q.split(' — ')[0]}</div>
                    {flipped[i] ? <div style={{ animation: 'vz-note-in .3s ease' }}>{PAPERS[paper].answers[i]}</div> : <div style={{ opacity: 0.7 }}>{q.split(' — ')[1]} 👆</div>}
                  </button>
                ))}
              </div>
            </div>
          </>
        )}
      </div>
      <Controls>
        {tab === 'tree' && (
          <Button variant="ghost" onClick={() => (setSaved(fresh()), setChallenge(null))}>
            ↺ 체크 기록 지우기
          </Button>
        )}
        {tab === 'paper' && (
          <Button variant="ghost" onClick={() => setFlipped(QUESTIONS.map(() => false))}>
            ↺ 카드 다시 덮기
          </Button>
        )}
        {tab === 'sort' && (
          <Button
            variant="ghost"
            onClick={() => {
              setRound((r) => r + 1);
              setPicks({});
              setPos(0);
            }}
          >
            ↺ 카드 섞어서 처음부터
          </Button>
        )}
      </Controls>
    </div>
  );
}
