import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { Html } from '@react-three/drei';
import * as THREE from 'three';
import type { VizProps } from '../Viz';
import { Scene3D, Arrow3D, Line } from '../core/three';
import { Button, Controls, Note, Segmented, Toggle } from '../core/ui';
import { usePrefersReducedMotion } from '../core/hooks';
import { useVizColors } from '../core/colors';
import type { AlgoMood } from '@/components/mascot/Algo';
import { ANALOGIES, CATEGORIES, WORDS, add, indexOf, nearest, sub, type CategoryKey, type Vec3 } from './_embed';

const catColor = (cat: CategoryKey, theme: 'light' | 'dark') => {
  const h = CATEGORIES[cat].hue;
  if (cat === 'ambiguous') return theme === 'dark' ? '#e8ecff' : '#3b3550';
  return theme === 'dark' ? `hsl(${h}, 95%, 64%)` : `hsl(${h}, 68%, 50%)`;
};

/** Three.js wants hex; convert an hsl() string once per theme. */
const toHex = (css: string) => `#${new THREE.Color(css).getHexString()}`;

export default function EmbeddingSpace3D({ expanded }: VizProps) {
  const c = useVizColors();
  const reduce = usePrefersReducedMotion();
  const [hover, setHover] = useState<number | null>(null);
  const [sel, setSel] = useState<number>(indexOf('고양이'));
  const [preset, setPreset] = useState(0);
  const [phase, setPhase] = useState(0);
  const [showLabels, setShowLabels] = useState(true);
  const [spin, setSpin] = useState(false);
  const [focus, setFocus] = useState<CategoryKey | null>(null);
  const timers = useRef<number[]>([]);

  const colors = useMemo(() => {
    const out = {} as Record<CategoryKey, string>;
    (Object.keys(CATEGORIES) as CategoryKey[]).forEach((k) => (out[k] = toHex(catColor(k, c.theme))));
    return out;
  }, [c.theme]);

  const active = hover ?? sel;
  const neighbors = useMemo(() => nearest(WORDS[active].p, [active], 5), [active]);

  const an = ANALOGIES[preset];
  const ia = indexOf(an.a);
  const ib = indexOf(an.b);
  const ic = indexOf(an.c);
  const target: Vec3 = add(sub(WORDS[ia].p, WORDS[ib].p), WORDS[ic].p);
  const answer = nearest(target, [ia, ib, ic], 1)[0];

  const clearTimers = () => {
    timers.current.forEach((t) => window.clearTimeout(t));
    timers.current = [];
  };
  useEffect(() => clearTimers, []);

  const runAnalogy = () => {
    clearTimers();
    if (reduce) {
      setPhase(4);
      return;
    }
    setPhase(1);
    [2, 3, 4].forEach((p, k) => timers.current.push(window.setTimeout(() => setPhase(p), 900 * (k + 1))));
  };

  const inAnalogy = phase > 0 ? new Set([ia, ib, ic, ...(phase >= 4 ? [answer.i] : [])]) : new Set<number>();
  const neighborSet = new Set(neighbors.slice(0, 4).map((n) => n.i));
  // viewport-based height: no feedback loop with the page scrollbar
  const stageH = expanded ? 'min(66vh, 760px)' : 'clamp(340px, 50vw, 520px)';
  const lineCol = c.theme === 'dark' ? '#b4c3ff' : '#5b5570';

  let note: { text: ReactNode; mood: AlgoMood; tone?: 'good' } = {
    text: (
      <>
        「<strong>{WORDS[active].w}</strong>」 가까이에는 {neighbors.slice(0, 3).map((n) => `「${WORDS[n.i].w}」`).join(', ')}가 있어요. 비슷한 문맥에 나오는 낱말끼리 가까이 모여요 —
        <strong> 거리가 곧 닮음</strong>이에요.
      </>
    ),
    mood: 'happy'
  };
  if (WORDS[active].w === '배')
    note = {
      text: (
        <>
          「<strong>배</strong>」는 타는 배·먹는 배·몸의 배… 그런데 벡터는 <strong>하나뿐</strong>이라 음식과 탈것 사이 어중간한 곳에 있어요. 문맥을 보고 뜻을 정하는 장치(어텐션)가 다음 장의 주제예요!
        </>
      ),
      mood: 'think'
    };
  if (phase >= 4)
    note = {
      text: (
        <>
          {an.a} - {an.b} + {an.c} 의 도착점(⭐) 가장 가까이에 「<strong>{WORDS[answer.i].w}</strong>」가 있어요! (코사인 유사도 {answer.sim.toFixed(2)}) 「{an.b}→{an.a}」 화살표와
          「{an.c}→{WORDS[answer.i].w}」 화살표가 나란하죠? 관계가 <strong>같은 방향</strong>으로 새겨져 있기 때문이에요.
        </>
      ),
      mood: 'cheer',
      tone: 'good'
    };
  else if (phase > 0)
    note = {
      text: (
        <>
          {phase === 1 && `「${an.b}」에서 「${an.a}」로 가는 화살표를 찾는 중…`}
          {phase === 2 && `이게 ${an.relation}이에요. 이번엔 같은 화살표를 「${an.c}」에 붙여 볼게요…`}
          {phase === 3 && '화살표 끝(⭐)은 어디일까요?'}
        </>
      ),
      mood: 'wow'
    };

  return (
    <div>
      <Scene3D height={stageH} camera={[8.2, 5.2, 9.2]} target={[0, 0, 0]} autoRotate={spin} minDistance={5} maxDistance={20}>
        <SpaceBox color={lineCol} />
        {/* the selected word as a vector from the origin */}
        {phase === 0 && <Line points={[[0, 0, 0], WORDS[active].p]} color={colors[WORDS[active].cat]} lineWidth={2} dashed dashSize={0.18} gapSize={0.12} />}
        <mesh>
          <sphereGeometry args={[0.08, 12, 12]} />
          <meshBasicMaterial color={lineCol} />
        </mesh>
        {phase === 0 &&
          neighbors.slice(0, 4).map((n) => (
            <Line key={n.i} points={[WORDS[active].p, WORDS[n.i].p]} color={c.series[0]} lineWidth={2.2} transparent opacity={0.85} />
          ))}
        {phase >= 2 && <Arrow3D from={WORDS[ib].p} to={WORDS[ia].p} color={c.series[2]} width={4} head={0.32} />}
        {phase >= 3 && <Arrow3D from={WORDS[ic].p} to={target} color={c.series[2]} width={4} head={0.32} />}
        {phase >= 4 && (
          <>
            <Line points={[WORDS[ib].p, WORDS[ic].p]} color={c.series[2]} lineWidth={1.4} dashed dashSize={0.12} gapSize={0.1} transparent opacity={0.6} />
            <Line points={[WORDS[ia].p, target]} color={c.series[2]} lineWidth={1.4} dashed dashSize={0.12} gapSize={0.1} transparent opacity={0.6} />
            <TargetStar position={target} color={c.series[4]} />
          </>
        )}
        {WORDS.map((wd, i) => {
          const dim = (focus && wd.cat !== focus) || (phase > 0 && !inAnalogy.has(i));
          const strong = i === active || inAnalogy.has(i) || (phase === 0 && neighborSet.has(i));
          return (
            <group key={wd.w} position={wd.p}>
              <mesh
                onPointerOver={(e) => {
                  e.stopPropagation();
                  setHover(i);
                }}
                onPointerOut={() => setHover((h) => (h === i ? null : h))}
                onClick={(e) => {
                  e.stopPropagation();
                  if (e.delta > 6) return;
                  setSel(i);
                  setPhase(0);
                  clearTimers();
                }}
                scale={i === active ? 1.55 : strong ? 1.2 : 1}
              >
                <sphereGeometry args={[0.15, 18, 18]} />
                <meshStandardMaterial
                  color={colors[wd.cat]}
                  emissive={colors[wd.cat]}
                  emissiveIntensity={c.theme === 'dark' ? (i === active ? 0.9 : 0.45) : i === active ? 0.35 : 0.08}
                  transparent
                  opacity={dim ? 0.18 : 1}
                  roughness={0.35}
                />
              </mesh>
              {(showLabels || strong) && (
                <Html center zIndexRange={[20, 0]} pointerEvents="none" position={[0, 0.34, 0]}>
                  <span
                    style={{
                      display: 'inline-block',
                      padding: strong ? '2px 8px' : '1px 6px',
                      borderRadius: 8,
                      whiteSpace: 'nowrap',
                      fontFamily: 'var(--font-sans)',
                      fontSize: strong ? 13 : 11,
                      fontWeight: strong ? 850 : 650,
                      color: strong ? 'var(--text)' : 'var(--text-2)',
                      background: strong ? 'var(--surface-solid)' : 'color-mix(in srgb, var(--surface-solid) 70%, transparent)',
                      border: `1px solid ${strong ? colors[wd.cat] : 'var(--border)'}`,
                      opacity: dim ? 0.25 : 1,
                      boxShadow: i === active ? `0 0 12px ${colors[wd.cat]}` : 'none',
                      userSelect: 'none'
                    }}
                  >
                    {wd.w}
                  </span>
                </Html>
              )}
            </group>
          );
        })}
      </Scene3D>

      <div className="vz-pad" style={{ display: 'grid', gap: 12 }}>
        <Note mood={note.mood} tone={note.tone}>
          {note.text}
        </Note>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 14 }}>
          <div>
            <div className="vz-panel-title" style={{ marginBottom: 8 }}>
              「{WORDS[active].w}」와 닮은 낱말 (코사인 유사도)
            </div>
            <div style={{ display: 'grid', gap: 6 }}>
              {neighbors.map((n) => (
                <button
                  key={n.i}
                  type="button"
                  onClick={() => {
                    setSel(n.i);
                    setPhase(0);
                  }}
                  style={{ display: 'grid', gridTemplateColumns: '78px 1fr 40px', gap: 8, alignItems: 'center', fontSize: 13.5, textAlign: 'left' }}
                >
                  <span style={{ fontWeight: 800 }}>{WORDS[n.i].w}</span>
                  <span style={{ height: 10, borderRadius: 99, background: 'var(--surface-3)', overflow: 'hidden' }}>
                    <span style={{ display: 'block', height: '100%', width: `${Math.max(0, n.sim) * 100}%`, background: colors[WORDS[n.i].cat], borderRadius: 99 }} />
                  </span>
                  <span style={{ fontVariantNumeric: 'tabular-nums', color: 'var(--text-2)', fontWeight: 700 }}>{n.sim.toFixed(2)}</span>
                </button>
              ))}
            </div>
          </div>
          <div style={{ display: 'grid', gap: 10, alignContent: 'start' }}>
            <div className="vz-panel-title">낱말 계산기 — 벡터로 더하고 빼기</div>
            <Segmented
              options={ANALOGIES.map((x, i) => ({ value: i, label: `${x.a}-${x.b}+${x.c}` }))}
              value={preset}
              onChange={(v) => {
                clearTimers();
                setPreset(v);
                setPhase(0);
              }}
            />
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
              <Button variant="primary" onClick={runAnalogy}>
                ▶ 계산해 보기
              </Button>
              <span style={{ fontWeight: 800, fontSize: 15 }}>
                {an.a} - {an.b} + {an.c} ≈ {phase >= 4 ? <span style={{ color: c.series[2] }}>{WORDS[answer.i].w} ✨</span> : '?'}
              </span>
            </div>
          </div>
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          {(Object.keys(CATEGORIES) as CategoryKey[]).map((k) => (
            <button
              key={k}
              type="button"
              onClick={() => setFocus((f) => (f === k ? null : k))}
              aria-pressed={focus === k}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 6,
                padding: '4px 10px',
                borderRadius: 99,
                border: `1.5px solid ${focus === k ? colors[k] : 'var(--border)'}`,
                background: focus === k ? 'var(--surface-solid)' : 'var(--surface-2)',
                fontSize: 12.5,
                fontWeight: 750,
                color: 'var(--text-2)'
              }}
            >
              <span style={{ width: 10, height: 10, borderRadius: 99, background: colors[k] }} />
              {CATEGORIES[k].emoji} {CATEGORIES[k].label}
            </button>
          ))}
        </div>
      </div>

      <Controls>
        <Toggle label="이름표 모두 보기" checked={showLabels} onChange={setShowLabels} />
        <Toggle label="천천히 돌리기" checked={spin} onChange={setSpin} />
        <Button variant="ghost" onClick={() => { clearTimers(); setPhase(0); setFocus(null); setSel(indexOf('고양이')); }}>
          ↺ 처음으로
        </Button>
      </Controls>
    </div>
  );
}

/** Faint cube around the space for depth cues. */
function SpaceBox({ color }: { color: string }) {
  const geom = useMemo(() => new THREE.EdgesGeometry(new THREE.BoxGeometry(10, 10, 10)), []);
  return (
    <lineSegments geometry={geom}>
      <lineBasicMaterial color={color} transparent opacity={0.16} />
    </lineSegments>
  );
}

function TargetStar({ position, color }: { position: Vec3; color: string }) {
  return (
    <group position={position}>
      <mesh rotation={[Math.PI / 4, Math.PI / 4, 0]}>
        <octahedronGeometry args={[0.24, 0]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.7} />
      </mesh>
      <mesh>
        <sphereGeometry args={[0.5, 20, 20]} />
        <meshBasicMaterial color={color} transparent opacity={0.14} depthWrite={false} />
      </mesh>
    </group>
  );
}
