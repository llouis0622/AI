import { useMemo, useRef, useState } from 'react';
import { useFrame, type ThreeEvent } from '@react-three/fiber';
import * as THREE from 'three';
import type { VizProps } from '../Viz';
import { Scene3D, Label3D, Line } from '../core/three';
import { Controls, Note, PlayBar, Segmented, Toggle } from '../core/ui';
import { useInterval } from '../core/hooks';
import { useVizColors } from '../core/colors';

type Variant = 'gpt' | 'bert';
type Kind = 'embed' | 'ln1' | 'attn' | 'add1' | 'ln2' | 'ffn' | 'add2' | 'head';

const XS = [-2.4, -0.8, 0.8, 2.4];
const BLOCK_H = 3.2;
const base = (b: number) => 1.2 + b * BLOCK_H;
const OFFSET: Record<Exclude<Kind, 'embed' | 'head'>, number> = { ln1: 0, attn: 0.8, add1: 1.6, ln2: 2.0, ffn: 2.6, add2: 3.0 };
const HEAD_Y = base(2) + 0.4;

interface Stage {
  kind: Kind;
  block: number;
  y: number;
}

const STAGES: Stage[] = [
  { kind: 'embed', block: -1, y: 0.4 },
  ...[0, 1].flatMap((b) => (['ln1', 'attn', 'add1', 'ln2', 'ffn', 'add2'] as const).map((k) => ({ kind: k, block: b, y: base(b) + OFFSET[k] }))),
  { kind: 'head', block: -1, y: HEAD_Y }
];

const TOKENS: Record<Variant, string[]> = {
  gpt: ['나는', '고양이를', '정말', '좋아'],
  bert: ['나는', '[MASK]', '정말', '좋아해']
};

const OUTPUTS: Record<Variant, [string, number][]> = {
  gpt: [
    ['해', 0.62],
    ['한다', 0.21],
    ['했다', 0.11],
    ['?', 0.06]
  ],
  bert: [
    ['고양이를', 0.48],
    ['너를', 0.27],
    ['공부를', 0.15],
    ['빵을', 0.1]
  ]
};

/** attention[block][query][key]; GPT is causal (keys ≤ query only). */
const ATTN: Record<Variant, number[][][]> = {
  gpt: [
    [
      [1, 0, 0, 0],
      [0.6, 0.4, 0, 0],
      [0.3, 0.3, 0.4, 0],
      [0.15, 0.55, 0.2, 0.1]
    ],
    [
      [1, 0, 0, 0],
      [0.45, 0.55, 0, 0],
      [0.15, 0.5, 0.35, 0],
      [0.1, 0.6, 0.15, 0.15]
    ]
  ],
  bert: [
    [
      [0.4, 0.2, 0.15, 0.25],
      [0.35, 0.1, 0.15, 0.4],
      [0.15, 0.3, 0.3, 0.25],
      [0.3, 0.35, 0.2, 0.15]
    ],
    [
      [0.35, 0.3, 0.1, 0.25],
      [0.3, 0.05, 0.2, 0.45],
      [0.2, 0.3, 0.25, 0.25],
      [0.25, 0.45, 0.15, 0.15]
    ]
  ]
};

const STAGE_NAME: Record<Kind, string> = {
  embed: '입력 임베딩',
  ln1: '정규화(LayerNorm)',
  attn: '멀티헤드 어텐션',
  add1: '더하기(잔차 연결)',
  ln2: '정규화(LayerNorm)',
  ffn: '피드포워드 망',
  add2: '더하기(잔차 연결)',
  head: '출력: 단어 확률'
};

function explain(kind: Kind, variant: Variant): string {
  switch (kind) {
    case 'embed':
      return '단어 하나하나가 숫자 묶음(벡터) 상자가 돼요. 상자 4개가 블록을 타고 위로 올라가며 점점 똑똑해져요.';
    case 'ln1':
    case 'ln2':
      return '다음 단계에 들어가기 전에 숫자들의 크기를 고르게 맞춰요(정규화). 학습이 안정적으로 되게 해 주는 손질이에요.';
    case 'attn':
      return variant === 'gpt'
        ? '단어들이 서로를 바라보며 정보를 주고받아요. 선이 굵을수록 많이 참고해요. GPT는 앞 단어만 볼 수 있어서 선이 모두 왼쪽(과거)에서 와요.'
        : '단어들이 서로를 바라보며 정보를 주고받아요. BERT는 앞뒤를 모두 봐요. [MASK] 상자가 양쪽 단어를 보며 빈칸을 추리해요.';
    case 'add1':
    case 'add2':
      return '방금 계산한 결과에 원래 상자를 지름길(잔차 연결)로 더해 줘요. 층이 깊어져도 정보와 기울기가 사라지지 않게 해요.';
    case 'ffn':
      return '이번엔 각 단어가 혼자서 생각을 정리해요. 어텐션이 "대화"라면, 피드포워드는 각자의 "필기 정리" 시간이에요.';
    case 'head':
      return variant === 'gpt'
        ? '맨 위에 도착한 마지막 단어의 상자로 "다음에 올 단어"의 확률을 계산해요. 이렇게 한 글자씩 이어 쓰는 게 GPT예요.'
        : '[MASK] 자리의 상자로 "빈칸에 들어갈 단어"의 확률을 계산해요. 이렇게 빈칸 맞히기로 배우는 게 BERT예요.';
  }
}

/**
 * An exploded Transformer: token boxes climb through two blocks (LayerNorm →
 * attention → add → LayerNorm → feed-forward → add) to the output head.
 */
export default function Transformer3D({ expanded }: VizProps) {
  const c = useVizColors();
  const [variant, setVariant] = useState<Variant>('gpt');
  const [stage, setStage] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [rails, setRails] = useState(true);

  useInterval(
    () => {
      if (stage >= STAGES.length - 1) setPlaying(false);
      else setStage((s) => s + 1);
    },
    playing ? 1500 : null
  );

  const cur = STAGES[stage];
  const tokenColors = [c.series[0], c.series[1], c.series[2], c.series[4]];
  const focus = variant === 'gpt' ? 3 : 1;
  const stageH = expanded ? 'min(74vh, 820px)' : 'clamp(400px, 90cqw, 600px)';
  const plateColor = c.theme === 'dark' ? '#5ce8ff' : '#7c6cf6';
  const ink = c.theme === 'dark' ? '#b4c3ff' : '#5b5570';

  const goTo = (i: number) => (e: ThreeEvent<MouseEvent>) => {
    if (e.delta > 6) return;
    e.stopPropagation();
    setPlaying(false);
    setStage(i);
  };

  return (
    <div>
      <Scene3D height={stageH} camera={[9.5, 7.5, 11.5]} target={[0, 4.1, 0]} minDistance={6} maxDistance={26}>
        {/* token labels under the input */}
        {TOKENS[variant].map((t, i) => (
          <Label3D key={`${variant}-${t}-${i}`} position={[XS[i], -0.35, 0.9]}>
            {t}
          </Label3D>
        ))}
        <Plate y={0.1} label="입력 임베딩" active={cur.kind === 'embed'} color={plateColor} onClick={goTo(0)} />

        {[0, 1].map((b) => {
          const idx = (k: Kind) => STAGES.findIndex((s) => s.block === b && s.kind === k);
          const at = (k: Kind) => cur.block === b && cur.kind === k;
          return (
            <group key={b}>
              <Plate y={base(b) + OFFSET.ln1} label={`블록 ${b + 1} · 정규화`} thin active={at('ln1')} color={plateColor} onClick={goTo(idx('ln1'))} />
              <Plate y={base(b) + OFFSET.attn} label={`블록 ${b + 1} · 어텐션`} active={at('attn')} color={c.series[1]} onClick={goTo(idx('attn'))} />
              <Arcs y={base(b) + OFFSET.attn + 0.12} weights={ATTN[variant][b]} colors={tokenColors} on={at('attn')} />
              <Plate y={base(b) + OFFSET.ln2} label={`블록 ${b + 1} · 정규화`} thin active={at('ln2')} color={plateColor} onClick={goTo(idx('ln2'))} />
              {XS.map((x, i) => (
                <mesh key={i} position={[x, base(b) + OFFSET.ffn, 0]} onClick={goTo(idx('ffn'))}>
                  <boxGeometry args={[1.0, 0.36, 0.9]} />
                  <meshStandardMaterial color={c.series[2]} transparent opacity={at('ffn') ? 0.55 : 0.22} emissive={c.series[2]} emissiveIntensity={at('ffn') ? 0.5 : 0} />
                </mesh>
              ))}
              <Label3D position={[-4.1, base(b) + OFFSET.ffn, 0]}>{`블록 ${b + 1} · 피드포워드`}</Label3D>
              {rails &&
                XS.map((x, i) => (
                  <group key={`r${i}`}>
                    <Line
                      points={[
                        [x + 0.62, base(b) - 0.35, 0.3],
                        [x + 0.62, base(b) + OFFSET.add1, 0.3],
                        [x + 0.2, base(b) + OFFSET.add1, 0.3]
                      ]}
                      color={c.series[3]}
                      lineWidth={at('add1') ? 4 : 1.5}
                      transparent
                      opacity={at('add1') ? 1 : 0.45}
                    />
                    <Line
                      points={[
                        [x + 0.62, base(b) + OFFSET.add1 + 0.05, 0.3],
                        [x + 0.62, base(b) + OFFSET.add2, 0.3],
                        [x + 0.2, base(b) + OFFSET.add2, 0.3]
                      ]}
                      color={c.series[3]}
                      lineWidth={at('add2') ? 4 : 1.5}
                      transparent
                      opacity={at('add2') ? 1 : 0.45}
                    />
                  </group>
                ))}
            </group>
          );
        })}

        <Plate y={HEAD_Y - 0.2} label={variant === 'gpt' ? '출력 · 다음 단어 확률' : '출력 · 빈칸 단어 확률'} active={cur.kind === 'head'} color={c.series[5]} onClick={goTo(STAGES.length - 1)} />
        {cur.kind === 'head' &&
          OUTPUTS[variant].map(([w, p], k) => {
            const x = -2.4 + k * 1.6;
            const h = 0.25 + p * 2.2;
            return (
              <group key={w}>
                <mesh position={[x, HEAD_Y - 0.15 + h / 2, -0.2]}>
                  <boxGeometry args={[0.7, h, 0.5]} />
                  <meshStandardMaterial color={k === 0 ? c.series[5] : ink} emissive={k === 0 ? c.series[5] : '#000000'} emissiveIntensity={k === 0 ? 0.5 : 0} transparent opacity={k === 0 ? 0.95 : 0.55} />
                </mesh>
                <Label3D position={[x, HEAD_Y + h + 0.15, -0.2]}>{`${w} ${Math.round(p * 100)}%`}</Label3D>
              </group>
            );
          })}

        <TokenBoxes stage={cur} colors={tokenColors} focus={focus} />
      </Scene3D>

      <div className="vz-pad" style={{ display: 'grid', gap: 12 }}>
        <PlayBar
          playing={playing}
          onToggle={() => {
            if (stage >= STAGES.length - 1) setStage(0);
            setPlaying((p) => !p);
          }}
          onStep={stage < STAGES.length - 1 ? () => setStage((s) => s + 1) : undefined}
          onReset={() => {
            setPlaying(false);
            setStage(0);
          }}
        />
        <Note mood={cur.kind === 'head' ? 'cheer' : cur.kind === 'attn' ? 'wow' : 'happy'} tone={cur.kind === 'head' ? 'good' : undefined}>
          <strong>
            {cur.block >= 0 ? `블록 ${cur.block + 1} · ` : ''}
            {STAGE_NAME[cur.kind]} ({stage + 1}/{STAGES.length})
          </strong>
          <br />
          {explain(cur.kind, variant)}
        </Note>
      </div>

      <Controls>
        <Segmented
          label="모델 종류"
          options={[
            { value: 'gpt', label: 'GPT (앞만 보기)' },
            { value: 'bert', label: 'BERT (앞뒤 보기)' }
          ]}
          value={variant}
          onChange={(v) => {
            setVariant(v as Variant);
            setStage(0);
            setPlaying(false);
          }}
        />
        <Toggle label="잔차 연결(지름길) 보기" checked={rails} onChange={setRails} />
      </Controls>
    </div>
  );
}

function Plate({
  y,
  label,
  active,
  color,
  thin = false,
  onClick
}: {
  y: number;
  label: string;
  active: boolean;
  color: string;
  thin?: boolean;
  onClick: (e: ThreeEvent<MouseEvent>) => void;
}) {
  return (
    <group>
      <mesh position={[0, y, 0]} onClick={onClick}>
        <boxGeometry args={[6.6, thin ? 0.05 : 0.09, 1.5]} />
        <meshStandardMaterial color={color} transparent opacity={active ? 0.5 : 0.16} emissive={color} emissiveIntensity={active ? 0.45 : 0.05} depthWrite={false} />
      </mesh>
      <Label3D position={[-4.1, y, 0]}>{label}</Label3D>
    </group>
  );
}

/** Attention arcs from each key token to each query token (thickness = weight). */
function Arcs({ y, weights, colors, on }: { y: number; weights: number[][]; colors: string[]; on: boolean }) {
  const arcs = useMemo(() => {
    const out: { pts: [number, number, number][]; w: number; q: number }[] = [];
    weights.forEach((row, q) =>
      row.forEach((w, k) => {
        if (k === q || w < 0.12) return;
        const a = new THREE.Vector3(XS[k], y, 0);
        const b = new THREE.Vector3(XS[q], y, 0);
        const mid = new THREE.Vector3((XS[k] + XS[q]) / 2, y + 0.35 + Math.abs(XS[q] - XS[k]) * 0.18, -0.25 + q * 0.12);
        const curve = new THREE.QuadraticBezierCurve3(a, mid, b);
        out.push({ pts: curve.getPoints(24).map((p) => [p.x, p.y, p.z] as [number, number, number]), w, q });
      })
    );
    return out;
  }, [weights, y]);
  return (
    <group>
      {arcs.map((a, i) => (
        <Line key={i} points={a.pts} color={colors[a.q]} lineWidth={on ? 1.5 + a.w * 7 : 1} transparent opacity={on ? 0.35 + a.w * 0.65 : 0.12} />
      ))}
    </group>
  );
}

/** The four token boxes climbing to the current stage (they swell inside the FFN). */
function TokenBoxes({ stage, colors, focus }: { stage: Stage; colors: string[]; focus: number }) {
  const refs = useRef<(THREE.Group | null)[]>([]);
  const swell = useRef(0);
  useFrame((_, dt) => {
    swell.current = stage.kind === 'ffn' ? Math.min(1, swell.current + dt * 1.5) : Math.max(0, swell.current - dt * 2);
    refs.current.forEach((g, i) => {
      if (!g) return;
      const targetY = stage.kind === 'head' ? (i === focus ? HEAD_Y - 0.55 : base(2) - 0.35) : stage.y + 0.3;
      g.position.y += (targetY - g.position.y) * (1 - Math.exp(-dt * 5));
      const s = 1 + Math.sin(swell.current * Math.PI) * 0.45;
      g.scale.setScalar(s);
    });
  });
  return (
    <group>
      {XS.map((x, i) => (
        <group key={i} ref={(el) => (refs.current[i] = el)} position={[x, 0.7, 0]}>
          <mesh>
            <boxGeometry args={[0.42, 0.42, 0.42]} />
            <meshStandardMaterial color={colors[i]} emissive={colors[i]} emissiveIntensity={0.55} roughness={0.3} />
          </mesh>
        </group>
      ))}
    </group>
  );
}
