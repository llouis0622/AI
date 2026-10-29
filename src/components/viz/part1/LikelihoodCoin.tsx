import { useMemo, useRef, useState } from 'react';
import type { VizProps } from '../Viz';
import { Plot, linePath } from '../core/plot';
import { Button, Controls, Note, Slider, Stat, Toggle } from '../core/ui';
import { useVizColors } from '../core/colors';
import { linspace, rng } from '../core/math';

const INITIAL = [1, 1, 0, 1, 1, 1, 0, 1, 0, 1]; // 7 heads, 3 tails

/** Maximum likelihood with coins: which p makes the observed flips most probable? */
export default function LikelihoodCoin({ expanded }: VizProps) {
  const c = useVizColors();
  const [flips, setFlips] = useState<number[]>(INITIAL);
  const [p, setP] = useState(0.4);
  const [asLoss, setAsLoss] = useState(false);
  const r = useRef(rng(7));

  const n = flips.length;
  const h = flips.filter((f) => f === 1).length;
  const tails = n - h;
  const pHat = n ? h / n : 0.5;
  const logL = (q: number) => h * Math.log(Math.max(q, 1e-9)) + tails * Math.log(Math.max(1 - q, 1e-9));
  const L = Math.exp(logL(p));
  const maxLog = logL(Math.min(0.999, Math.max(0.001, pHat)));
  const grid = useMemo(() => linspace(0.005, 0.995, 160), []);
  const curve = grid.map((q) => [q, asLoss ? -logL(q) / Math.max(1, n) : Math.exp(logL(q) - maxLog)] as [number, number]);
  const yMax = asLoss ? Math.max(1.5, ...curve.map((d) => Math.min(d[1], 4))) : 1.08;
  const near = Math.abs(p - pHat) < 0.025;
  const shown = flips.slice(0, 12);

  return (
    <div>
      <div className="vz-pad" style={{ display: 'grid', gap: 10, paddingBottom: 4 }}>
        <div className="vz-panel-title">관찰한 동전 (눌러서 앞/뒤를 바꿀 수 있어요) — 앞면 {h}번, 뒷면 {tails}번</div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          {flips.map((f, i) => (
            <button
              key={i}
              type="button"
              onClick={() => setFlips((fs) => fs.map((v, k) => (k === i ? 1 - v : v)))}
              aria-label={`${i + 1}번째 동전: ${f ? '앞면' : '뒷면'} (눌러서 바꾸기)`}
              style={{
                width: 38,
                height: 38,
                borderRadius: '50%',
                display: 'grid',
                placeItems: 'center',
                fontSize: 13,
                fontWeight: 900,
                color: f ? '#3b2a00' : 'var(--text)',
                background: f ? 'radial-gradient(circle at 35% 30%, #ffe48a, #f2b417)' : 'radial-gradient(circle at 35% 30%, var(--surface-solid), var(--surface-3))',
                border: `2px solid ${f ? '#d39a00' : 'var(--border-strong)'}`,
                boxShadow: 'var(--shadow-sm)'
              }}
            >
              {f ? '앞' : '뒤'}
            </button>
          ))}
        </div>
      </div>

      <div className="vz-pad" style={{ paddingTop: 4, paddingBottom: 0 }}>
        <Plot
          xDomain={[0, 1]}
          yDomain={[0, yMax]}
          xLabel="p = 모델이 생각하는 '앞면이 나올 확률'"
          yLabel={asLoss ? '손실 (-log 가능도)' : '가능도 (상대 크기)'}
          aspect={expanded ? 2.2 : 1.7}
          maxHeight={expanded ? 500 : 340}
          xFormat={(v) => v.toFixed(1)}
          yFormat={(v) => v.toFixed(1)}
          ariaLabel="동전 확률 p에 따른 가능도 곡선"
        >
          {({ x, y, clip }) => {
            const curY = asLoss ? -logL(p) / Math.max(1, n) : Math.exp(logL(p) - maxLog);
            return (
              <g>
                <g clipPath={clip}>
                  <path d={linePath(curve.map(([a, b]) => [x(a), y(Math.min(b, yMax))]))} fill="none" stroke={c.series[0]} strokeWidth={3.5} />
                  <line x1={x(pHat)} x2={x(pHat)} y1={y(0)} y2={y(yMax)} stroke={c.pos} strokeWidth={2} strokeDasharray="5 4" />
                  <line x1={x(p)} x2={x(p)} y1={y(0)} y2={y(Math.min(curY, yMax))} strokeWidth={2} style={{ stroke: 'var(--lemon)' }} />
                </g>
                <circle cx={x(p)} cy={y(Math.min(curY, yMax))} r={9} stroke={c.surface} strokeWidth={3} style={{ fill: 'var(--lemon)' }} />
                <text x={x(pHat)} y={y(yMax) + 14} textAnchor={pHat > 0.5 ? 'end' : 'start'} dx={pHat > 0.5 ? -6 : 6} style={{ fontSize: 12, fontWeight: 850, fill: c.pos }}>
                  🏆 {asLoss ? '손실 최소' : '가능도 최대'} = {h}/{n} = {pHat.toFixed(2)}
                </text>
              </g>
            );
          }}
        </Plot>
      </div>

      <div className="vz-pad" style={{ display: 'grid', gap: 12 }}>
        <div style={{ padding: '10px 14px', borderRadius: 14, background: 'var(--surface-2)', border: '1px solid var(--border)', fontSize: 14, lineHeight: 1.8, overflowX: 'auto' }}>
          <span style={{ fontWeight: 750 }}>이 동전들이 나올 확률 = </span>
          {shown.map((f, i) => (
            <span key={i} style={{ color: f ? '#c58a00' : 'var(--text-2)', fontWeight: 800 }}>
              {f ? p.toFixed(2) : (1 - p).toFixed(2)}
              {i < shown.length - 1 ? ' × ' : ''}
            </span>
          ))}
          {flips.length > shown.length && <span> × …</span>}
          <span style={{ fontWeight: 900 }}> = {L < 1e-4 ? L.toExponential(2) : L.toFixed(5)}</span>
        </div>
        <div className="vz-stats">
          <Stat label="앞면 비율 (관찰)" value={`${h}/${n}`} sub={`= ${pHat.toFixed(2)}`} />
          <Stat label="내가 고른 p" value={p.toFixed(2)} tone={near ? 'good' : 'brand'} />
          <Stat label="가능도 (최고 대비)" value={`${Math.round(100 * Math.exp(logL(p) - maxLog))}%`} tone={near ? 'good' : undefined} sub={`= ${L.toPrecision(2)}`} />
        </div>
        <Note mood={near ? 'cheer' : 'think'} tone={near ? 'good' : undefined}>
          {near
            ? `찾았어요! 🎉 p = ${pHat.toFixed(2)}일 때 관찰한 결과가 나올 확률이 가장 커요. 이렇게 "데이터를 가장 그럴듯하게 만드는 값"을 고르는 게 최대가능도 원리예요.`
            : `슬라이더로 p를 바꿔 노란 점을 곡선의 꼭대기로 옮겨 보세요. 앞면이 ${h}번 나왔다면, p는 얼마쯤일 때 가장 그럴듯할까요?`}
          {asLoss && ' 가능도를 가장 크게 = 마이너스 로그(손실)를 가장 작게. 손실 함수는 이렇게 "도출"돼요!'}
        </Note>
      </div>

      <Controls>
        <Slider label="p (앞면 확률)" value={p} min={0.01} max={0.99} step={0.01} onChange={setP} format={(v) => v.toFixed(2)} accent="var(--lemon)" />
        <Toggle label="손실(-log)로 뒤집어 보기" checked={asLoss} onChange={setAsLoss} />
        <Button size="sm" onClick={() => setFlips((fs) => [...fs, r.current.next() < 0.7 ? 1 : 0].slice(0, 24))} disabled={flips.length >= 24}>
          🪙 하나 더 던지기
        </Button>
        <Button size="sm" variant="ghost" onClick={() => setFlips((fs) => (fs.length > 1 ? fs.slice(0, -1) : fs))}>
          하나 빼기
        </Button>
        <Button size="sm" variant="ghost" onClick={() => { setFlips(INITIAL); setP(0.4); }}>
          ↺ 처음으로
        </Button>
      </Controls>
    </div>
  );
}
