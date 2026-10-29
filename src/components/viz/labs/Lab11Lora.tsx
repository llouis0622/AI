import { useState } from 'react';
import type { VizProps } from '../Viz';
import { CodeBlock } from '../core/story';
import { Controls, Note, Segmented, Stat } from '../core/ui';
import { useElementSize } from '../core/hooks';
import { useVizColors, type VizColors } from '../core/colors';
import { FS, alpha, T, Arrow, Pulse } from './_LabKitB';

type Mode = 'structure' | 'init' | 'forward' | 'merge';

const CODE = String.raw`class LoRALinear(nn.Module):
    """기존 Linear를 감싸 W를 동결하고 BA만 학습한다."""
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False                      # W, b 동결
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))  # B=0 → 초기엔 ΔW=0
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T @ self.B.T)

    def merge(self):
        """추론용: ΔW를 W에 합쳐 지연 없는 일반 Linear로 되돌린다."""
        self.base.weight.data += self.scale * (self.B @ self.A)
        return self.base`;

const HIGHLIGHT: Record<Mode, number[]> = {
  structure: [6, 7, 8, 9],
  init: [8, 9],
  forward: [12, 13],
  merge: [15, 16, 17, 18]
};

const NOTES: Record<Mode, { text: string; mood: 'smile' | 'think' | 'happy' | 'wow' }> = {
  structure: { text: '원래 가중치 W는 얼려 두고(🧊), 옆에 가느다란 길 A → B를 하나 더 내요. 학습되는 건 이 가느다란 길뿐이라 파라미터가 확 줄어요.', mood: 'smile' },
  init: { text: 'B를 0으로 시작하면 ΔW = B·A = 0 → 처음엔 원본 모델과 출력이 똑같아요. 사전학습 실력에서 출발해 조금씩 움직이는 거예요.', mood: 'think' },
  forward: { text: '입력 x가 두 길로 나뉘어 W·x와 (α/r)·B·A·x를 계산한 뒤 더해요. r을 줄이면 가운데 병목이 좁아지는 걸 보세요.', mood: 'happy' },
  merge: { text: '다 배운 뒤에는 ΔW를 W에 더해 하나로 합쳐요(merge). 옆길이 사라져서 추론 속도는 원래 모델과 똑같아요.', mood: 'wow' }
};

/* 7B Llama-style model, LoRA on q,k,v,o,gate,up,down in 32 layers ≈ 2.5M params per unit of rank. */
const PARAMS_7B = 7e9;
const ADAPTER_PER_R = 2.5e6;

const fmtInt = (n: number) => Math.round(n).toLocaleString('ko-KR');
const gb = (bytes: number) => bytes / 1e9;

export default function Lab11Lora({ expanded }: VizProps) {
  const c = useVizColors();
  const [mode, setMode] = useState<Mode>('structure');
  const [d, setD] = useState(4096);
  const [r, setR] = useState(16);
  const [box, { width }] = useElementSize<HTMLDivElement>();
  const wide = width > 720;

  const full = d * d;
  const lora = 2 * d * r;
  const ratio = lora / full;
  const adapterParams = ADAPTER_PER_R * r;
  const mem = {
    full: gb(PARAMS_7B * 16),
    lora: gb(PARAMS_7B * 2 + adapterParams * 16),
    qlora: gb(PARAMS_7B * 0.5 + 0.1e9 + adapterParams * 16)
  };
  const note = NOTES[mode];

  return (
    <div ref={box}>
      <div style={{ display: 'grid', gridTemplateColumns: wide ? 'minmax(0, 1.3fr) minmax(0, 1fr)' : '1fr' }}>
        <div style={{ padding: '14px 14px 4px' }}>
          <svg className="vz-svg" viewBox="0 0 420 260" role="img" aria-label="LoRA 구조: 동결된 W와 저랭크 A, B" style={{ touchAction: 'pan-y' }}>
            <Diagram c={c} mode={mode} r={r} />
          </svg>
        </div>
        <div style={{ padding: wide ? '14px 14px 4px 0' : '0 14px 4px', minWidth: 0, maxHeight: expanded ? 'none' : 420, overflow: 'auto' }}>
          <CodeBlock code={CODE} highlight={HIGHLIGHT[mode]} c={c} />
        </div>
      </div>

      <div className="vz-pad" style={{ display: 'grid', gap: 12 }}>
        <Note mood={note.mood}>{note.text}</Note>
        <div className="vz-stats">
          <Stat label={`W 한 장 (${d}×${d})`} value={fmtInt(full)} sub="동결 — 학습 안 함" />
          <Stat label={`LoRA A+B (r=${r})`} value={fmtInt(lora)} tone="brand" sub="2 × d × r" />
          <Stat label="학습하는 비율" value={`${(ratio * 100).toFixed(2)}%`} tone="good" sub="2r / d" />
          <Stat label="7B 어댑터 파일" value={`약 ${Math.round((adapterParams * 2) / 1e6)}MB`} sub="bf16 기준" />
        </div>
        <MemoryBars c={c} mem={mem} />
      </div>

      <Controls>
        <Segmented
          label="보기"
          options={[
            { value: 'structure', label: '구조' },
            { value: 'init', label: 'B=0 시작' },
            { value: 'forward', label: '순전파' },
            { value: 'merge', label: '병합' }
          ]}
          value={mode}
          onChange={(v) => setMode(v as Mode)}
        />
        <Segmented
          label="랭크 r"
          options={[4, 8, 16, 32, 64].map((v) => ({ value: v, label: String(v) }))}
          value={r}
          onChange={(v) => setR(Number(v))}
        />
        <Segmented
          label="층 크기 d"
          options={[1024, 2048, 4096].map((v) => ({ value: v, label: String(v) }))}
          value={d}
          onChange={(v) => setD(Number(v))}
        />
      </Controls>
    </div>
  );
}

function Diagram({ c, mode, r }: { c: VizColors; mode: Mode; r: number }) {
  const hb = Math.min(44, Math.max(8, 6 + 6 * Math.log2(r)));
  const cy = 196;
  const merged = mode === 'merge';
  const zero = mode === 'init';
  const branchOpacity = merged ? 0.22 : 1;
  const wCol = merged ? c.series[2] : c.series[3];
  const train = c.series[0];
  return (
    <g>
      {/* input x */}
      <rect x={10} y={56} width={18} height={168} rx={6} fill={alpha(c.series[1], 0.16)} stroke={c.series[1]} strokeWidth={2} />
      <T x={19} y={244} anchor="middle" size={FS.s} weight={850}>
        x
      </T>
      <Arrow x1={30} y1={84} x2={66} y2={74} color={c.series[1]} />
      <Arrow x1={30} y1={190} x2={66} y2={cy} color={c.series[1]} t={merged ? 0 : 1} />

      {/* frozen W */}
      <rect x={70} y={20} width={140} height={104} rx={12} fill={alpha(wCol, merged ? 0.3 : 0.16)} stroke={wCol} strokeWidth={2.5} />
      {Array.from({ length: 6 }, (_, i) =>
        Array.from({ length: 9 }, (_, k) => (
          <rect key={`${i}-${k}`} x={80 + k * 14} y={32 + i * 13} width={11} height={10} rx={2} fill={alpha(wCol, 0.18 + 0.12 * ((i * 3 + k * 5) % 4))} />
        ))
      )}
      <T x={140} y={114} anchor="middle" size={FS.s} weight={850}>
        {merged ? "W' = W + (α/r)·B·A" : 'W (d×d)  🧊 동결'}
      </T>

      {/* low-rank branch: A (narrowing) → bottleneck r → B (widening) */}
      <g opacity={branchOpacity}>
        <path d={`M70 ${cy - 40} L136 ${cy - hb / 2} L136 ${cy + hb / 2} L70 ${cy + 40} Z`} fill={alpha(train, 0.2)} stroke={train} strokeWidth={2} strokeLinejoin="round" />
        <T x={100} y={cy + 5} anchor="middle" size={FS.s} weight={900}>
          A
        </T>
        <rect x={139} y={cy - hb / 2} width={12} height={hb} rx={3} fill={train} />
        <path
          d={`M154 ${cy - hb / 2} L220 ${cy - 40} L220 ${cy + 40} L154 ${cy + hb / 2} Z`}
          fill={zero ? 'none' : alpha(train, 0.2)}
          stroke={train}
          strokeWidth={2}
          strokeDasharray={zero ? '5 4' : undefined}
          strokeLinejoin="round"
        />
        <T x={190} y={cy + 5} anchor="middle" size={FS.s} weight={900}>
          {zero ? 'B=0' : 'B'}
        </T>
        <T x={145} y={cy + hb / 2 + 18} anchor="middle" size={FS.m} weight={850} color={train}>
          r={r}
        </T>
        <T x={145} y={252} anchor="middle" size={11.5} weight={750} color="var(--text-2)">
          ✏️ 학습: A (r×d), B (d×r)
        </T>
      </g>

      {/* merge point and output */}
      <circle cx={262} cy={120} r={15} fill="var(--surface-solid)" stroke={c.series[1]} strokeWidth={2.5} />
      <T x={262} y={126} anchor="middle" size={18} weight={900}>
        +
      </T>
      <Arrow x1={212} y1={74} x2={250} y2={110} color={wCol} />
      <g opacity={branchOpacity}>
        <Arrow x1={222} y1={cy} x2={254} y2={134} color={train} label={zero ? '+0' : '× α/r'} labelDy={18} />
      </g>
      <Arrow x1={278} y1={120} x2={300} y2={120} color={c.series[1]} />
      <rect x={302} y={56} width={18} height={168} rx={6} fill={alpha(c.series[1], 0.16)} stroke={c.series[1]} strokeWidth={2} />
      <T x={311} y={244} anchor="middle" size={FS.s} weight={850}>
        h
      </T>
      {mode === 'forward' && (
        <>
          <Pulse path="M30 84 L70 74 L210 74 L250 110 L300 120" color={c.series[1]} dur={2} />
          <Pulse path={`M30 190 L70 ${cy} L222 ${cy} L254 134 L300 120`} color={train} dur={2} delay={0.4} />
        </>
      )}

      {/* side legend */}
      <g>
        <T x={334} y={70} size={FS.m} weight={850} color={c.series[3]}>
          🧊 그대로
        </T>
        <T x={334} y={88} size={11.5} weight={700} color="var(--text-2)">
          requires_grad
        </T>
        <T x={334} y={104} size={11.5} weight={700} color="var(--text-2)">
          = False
        </T>
        <T x={334} y={150} size={FS.m} weight={850} color={train}>
          ✏️ 학습
        </T>
        <T x={334} y={168} size={11.5} weight={700} color="var(--text-2)">
          A는 작은 무작위,
        </T>
        <T x={334} y={184} size={11.5} weight={700} color="var(--text-2)">
          B는 0에서 출발
        </T>
      </g>
    </g>
  );
}

function MemoryBars({ c, mem }: { c: VizColors; mem: { full: number; lora: number; qlora: number } }) {
  const max = mem.full;
  const rows = [
    { label: '풀 파인튜닝', v: mem.full, color: c.neg },
    { label: 'LoRA', v: mem.lora, color: c.series[0] },
    { label: 'QLoRA (4비트)', v: mem.qlora, color: c.pos }
  ];
  const gpu = 24;
  return (
    <div>
      <div className="vz-panel-title">7B 모델 학습에 드는 GPU 메모리 (대략, 활성값 제외)</div>
      <div style={{ position: 'relative', display: 'grid', gap: 8, marginTop: 8 }}>
        {rows.map((row) => (
          <div key={row.label} style={{ display: 'grid', gridTemplateColumns: '96px 1fr 64px', alignItems: 'center', gap: 8, fontSize: 13, fontWeight: 700 }}>
            <span>{row.label}</span>
            <span style={{ position: 'relative', height: 14, borderRadius: 99, background: 'var(--surface-3)' }}>
              <span
                style={{
                  position: 'absolute',
                  inset: 0,
                  width: `${Math.max(1.5, (row.v / max) * 100)}%`,
                  borderRadius: 99,
                  background: row.color,
                  transition: 'width .4s var(--ease-out)'
                }}
              />
              <span
                aria-hidden="true"
                style={{ position: 'absolute', top: -4, bottom: -4, left: `${(gpu / max) * 100}%`, borderLeft: '2px dashed var(--text-3)' }}
              />
            </span>
            <span style={{ textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{row.v.toFixed(1)}GB</span>
          </div>
        ))}
        <div style={{ fontSize: 12, color: 'var(--text-2)', paddingLeft: 104 }}>┆ 점선 = 소비자 GPU 한 장 (24GB)</div>
      </div>
    </div>
  );
}
