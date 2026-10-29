import { useState } from 'react';
import type { VizProps } from '../Viz';
import { Button, Controls, Note } from '../core/ui';
import { useVizColors, withAlpha } from '../core/colors';

type RingId = 'ai' | 'ml' | 'dl' | 'gpt' | 'none';

const RINGS: { id: Exclude<RingId, 'none'>; name: string; short: string; cx: number; cy: number; r: number; era: string; def: string; examples: string }[] = [
  { id: 'ai', name: '인공지능 (AI)', short: 'AI', cx: 300, cy: 178, r: 164, era: '1950년대~', def: '사람이 하는 똑똑한 일을 기계가 해내게 하려는 모든 시도예요. 가장 큰 우산!', examples: '규칙으로 짠 전문가 시스템, 체스 프로그램' },
  { id: 'ml', name: '머신러닝 (ML)', short: 'ML', cx: 326, cy: 200, r: 124, era: '1980~90년대~', def: '규칙을 사람이 적지 않고, 데이터와 정답을 보여 주면 기계가 규칙을 스스로 찾는 방법이에요.', examples: '스팸 필터, 결정 트리, 상품 추천' },
  { id: 'dl', name: '딥러닝 (DL)', short: 'DL', cx: 352, cy: 226, r: 82, era: '2012년경~', def: '여러 층을 쌓은 인공 신경망으로 규칙을 찾는 머신러닝이에요. "딥"은 층이 깊다(많다)는 뜻!', examples: '얼굴 인식, 음성 인식, 번역' },
  { id: 'gpt', name: 'GPT', short: 'GPT', cx: 370, cy: 246, r: 30, era: '2018년~', def: '아주아주 큰 딥러닝 모델이에요. 엄청난 양의 글로 "다음 단어 맞히기"를 하며 배웠어요.', examples: 'ChatGPT 같은 챗봇' }
];

const ITEMS: { id: string; label: string; emoji: string; ring: RingId; why: string; at: [number, number] }[] = [
  { id: 'chess', emoji: '♟️', label: '규칙으로 짠 체스 프로그램', ring: 'ai', at: [180, 112], why: '사람이 짠 규칙과 수읽기로 두는 체스 프로그램은 AI예요. 하지만 데이터로 배우지 않으니 머신러닝은 아니에요.' },
  { id: 'spam', emoji: '📧', label: '단어 점수로 거르는 스팸 필터', ring: 'ml', at: [250, 148], why: '예시 메일에서 단어 점수를 스스로 배우니 머신러닝! 신경망을 쓰지 않으면 딥러닝은 아니에요.' },
  { id: 'tree', emoji: '🌳', label: '결정 트리', ring: 'ml', at: [252, 262], why: '2부에서 배울 결정 트리는 신경망 없이도 훌륭하게 배우는 머신러닝이에요.' },
  { id: 'cat', emoji: '🖼️', label: '사진 속 고양이 찾기', ring: 'dl', at: [318, 212], why: '여러 층의 신경망이 픽셀에서 고양이 모양을 찾아내요. 이건 딥러닝!' },
  { id: 'voice', emoji: '🎙️', label: '말을 글자로 바꾸는 음성 인식', ring: 'dl', at: [398, 196], why: '요즘 음성 인식은 신경망을 깊게 쌓은 딥러닝으로 만들어요.' },
  { id: 'gpt', emoji: '💬', label: 'ChatGPT', ring: 'gpt', at: [370, 248], why: 'GPT는 딥러닝 ⊂ 머신러닝 ⊂ 인공지능 — 네 개의 동그라미에 모두 들어가요!' },
  { id: 'calc', emoji: '🔢', label: '전자계산기', ring: 'none', at: [506, 44], why: '계산기는 정해진 계산을 그대로 할 뿐, 사람처럼 판단하는 일을 흉내 내지 않아서 보통 AI라고 부르지 않아요.' }
];

export default function AiMlDlNest(_: VizProps) {
  const c = useVizColors();
  const [sel, setSel] = useState<RingId>('ml');
  const [placed, setPlaced] = useState<string[]>([]);
  const [last, setLast] = useState<string | null>(null);
  const color: Record<Exclude<RingId, 'none'>, string> = { ai: c.series[4], ml: c.series[1], dl: c.series[0], gpt: c.series[5] };
  const ring = RINGS.find((r) => r.id === sel);
  const lastItem = ITEMS.find((i) => i.id === last);

  const place = (id: string) => {
    setPlaced((p) => (p.includes(id) ? p : [...p, id]));
    setLast(id);
    const it = ITEMS.find((i) => i.id === id);
    if (it && it.ring !== 'none') setSel(it.ring);
  };

  return (
    <div>
      <div style={{ padding: '12px 12px 0' }}>
        <svg className="vz-svg" viewBox="112 2 430 348" role="img" aria-label="인공지능 안에 머신러닝, 그 안에 딥러닝, 그 안에 GPT가 들어 있는 포함 관계 그림" style={{ touchAction: 'manipulation', maxWidth: 560, margin: '0 auto' }}>
          {RINGS.map((r) => {
            const active = sel === r.id;
            return (
              <g key={r.id} style={{ cursor: 'pointer' }} onClick={() => setSel(r.id)} role="button" tabIndex={0} aria-label={r.name} onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && setSel(r.id)}>
                <circle
                  cx={r.cx}
                  cy={r.cy}
                  r={r.r}
                  fill={withAlpha(color[r.id], active ? 0.3 : 0.14)}
                  stroke={color[r.id]}
                  strokeWidth={active ? 4 : 2}
                  style={{ transition: 'all .3s', filter: active && c.theme === 'dark' ? `drop-shadow(0 0 10px ${color[r.id]})` : undefined }}
                />
                {r.id !== 'gpt' && (
                  <text x={r.cx} y={r.cy - r.r + 26} textAnchor="middle" style={{ fontSize: r.id === 'ai' ? 17 : 15, fontWeight: 850, fill: 'var(--text)' }}>
                    {r.name}
                  </text>
                )}
              </g>
            );
          })}
          <text x={370} y={291} textAnchor="middle" style={{ fontSize: 12, fontWeight: 850, fill: 'var(--text)', pointerEvents: 'none' }}>
            GPT
          </text>
          {ITEMS.filter((i) => placed.includes(i.id)).map((i) => (
            <g key={i.id} transform={`translate(${i.at[0]} ${i.at[1]})`} style={{ pointerEvents: 'none' }}>
              <circle r={17} fill="var(--surface-solid)" stroke={i.ring === 'none' ? c.neg : color[i.ring as Exclude<RingId, 'none'>]} strokeWidth={2.5} />
              <text y={6} textAnchor="middle" style={{ fontSize: 17 }}>
                {i.emoji}
              </text>
              {last === i.id && <circle r={22} fill="none" stroke={c.brand} strokeWidth={2} strokeDasharray="4 4" />}
            </g>
          ))}
          {placed.includes('calc') && (
            <text x={506} y={80} textAnchor="middle" style={{ fontSize: 11.5, fontWeight: 750, fill: c.neg }}>
              AI 바깥
            </text>
          )}
        </svg>
      </div>

      <div className="vz-pad" style={{ display: 'grid', gap: 12, paddingTop: 6 }}>
        {ring && (
          <div style={{ padding: '12px 14px', borderRadius: 16, border: `2px solid ${color[ring.id]}`, background: withAlpha(color[ring.id], 0.1) }}>
            <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'baseline', gap: 8, marginBottom: 4 }}>
              <strong style={{ fontSize: 17 }}>{ring.name}</strong>
              <span className="vz-badge" style={{ background: 'var(--surface-solid)', color: 'var(--text-2)' }}>
                {ring.era}
              </span>
            </div>
            <div style={{ fontSize: 14.5, lineHeight: 1.65 }}>{ring.def}</div>
            <div style={{ fontSize: 13, color: 'var(--text-2)', marginTop: 4 }}>예: {ring.examples}</div>
          </div>
        )}

        <div>
          <div className="vz-panel-title" style={{ marginBottom: 8 }}>
            이건 어느 동그라미에 들어갈까? 눌러서 놓아 보세요 ({placed.length}/{ITEMS.length})
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {ITEMS.map((i) => (
              <button
                key={i.id}
                type="button"
                className={`vz-btn vz-btn-sm${placed.includes(i.id) ? ' is-active' : ''}`}
                onClick={() => place(i.id)}
                aria-pressed={placed.includes(i.id)}
              >
                <span aria-hidden="true">{i.emoji}</span> {i.label}
              </button>
            ))}
          </div>
        </div>

        {lastItem ? (
          <Note mood={lastItem.ring === 'none' ? 'think' : 'happy'} tone={lastItem.ring === 'none' ? 'warn' : 'good'}>
            <strong>{lastItem.label}</strong> → {lastItem.why}
          </Note>
        ) : (
          <Note mood="smile">동그라미를 누르면 설명이 나와요. 큰 동그라미일수록 넓은 개념이에요: GPT ⊂ 딥러닝 ⊂ 머신러닝 ⊂ AI.</Note>
        )}
      </div>

      <Controls>
        <Button variant="primary" onClick={() => { setPlaced(ITEMS.map((i) => i.id)); setLast('gpt'); setSel('gpt'); }}>
          ✨ 모두 놓아 보기
        </Button>
        <Button variant="ghost" onClick={() => { setPlaced([]); setLast(null); setSel('ml'); }}>
          ↺ 다시 하기
        </Button>
      </Controls>
    </div>
  );
}
