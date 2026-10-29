import { useEffect, useMemo, useRef, useState } from 'react';
import type { VizProps } from '../Viz';
import { Plot, linePath } from '../core/plot';
import { Button, Controls, Legend, Note, Slider, Stat, Toggle } from '../core/ui';
import { useElementSize, useInterval, useRafLoop, usePrefersReducedMotion } from '../core/hooks';
import { useVizColors, withAlpha } from '../core/colors';
import { range } from '../core/math';
import { FieldCanvas, ClassLegend } from './_Field';
import { makeDataset, splitTrainTest } from './_data';
import { accuracy, buildTree, gini, leaves, predictTree, type TreeNode } from './_tree';

const MAX_DEPTH = 8;
const FEAT = ['크기', '울퉁불퉁'];
const DATA = makeDataset('noisy', 120, 2);
const { train: TRAIN, test: TEST } = splitTrainTest(DATA);
const TREES = range(MAX_DEPTH + 1).map((d) => buildTree(TRAIN, { maxDepth: d }));
const ACC = TREES.map((t) => ({
  train: accuracy((x, y) => predictTree(t, x, y), TRAIN),
  test: accuracy((x, y) => predictTree(t, x, y), TEST)
}));

const q = (n: TreeNode) => `${FEAT[n.feature!]} < ${(n.threshold! * 10).toFixed(1)}`;

interface Placed {
  node: TreeNode;
  x: number;
  y: number;
  /** horizontal room (px) this subtree owns */
  room: number;
  path: string;
  parent?: Placed;
  isLeft?: boolean;
}

function place(root: TreeNode, width: number, top: number, levelH: number): Placed[] {
  const ls = leaves(root);
  const slot = width / ls.length;
  const out: Placed[] = [];
  let leafIdx = 0;
  const walk = (n: TreeNode, path: string, parent?: Placed, isLeft?: boolean): Placed => {
    if (n.feature === undefined) {
      const p: Placed = { node: n, x: (leafIdx + 0.5) * slot, y: top + n.depth * levelH, room: slot, path, parent, isLeft };
      leafIdx++;
      out.push(p);
      return p;
    }
    const me: Placed = { node: n, x: 0, y: top + n.depth * levelH, room: 0, path, parent, isLeft };
    out.push(me);
    const a = walk(n.left!, `${path}L`, me, true);
    const b = walk(n.right!, `${path}R`, me, false);
    me.x = (a.x + b.x) / 2;
    me.room = (leaves(n).length) * slot;
    return me;
  };
  walk(root, '');
  return out;
}

function findLeafPath(root: TreeNode, x: number, y: number) {
  let n = root;
  let path = '';
  while (n.feature !== undefined) {
    const goLeft = (n.feature ? y : x) < n.threshold!;
    path += goLeft ? 'L' : 'R';
    n = goLeft ? n.left! : n.right!;
  }
  return path;
}

/** Decision tree on 2D points: axis-aligned splits on the map + the growing tree diagram. */
export default function DecisionTree({ expanded }: VizProps) {
  const c = useVizColors();
  const reduce = usePrefersReducedMotion();
  const [depth, setDepth] = useState(0);
  const [showTest, setShowTest] = useState(false);
  const [sel, setSel] = useState<string | null>(null);
  const [growing, setGrowing] = useState(false);
  const [grow, setGrow] = useState(1);
  const prevDepth = useRef(0);
  const [box, { width }] = useElementSize<HTMLDivElement>();

  const tree = TREES[depth];
  const wide = width >= 700;
  const mapW = wide ? Math.floor(width * 0.5) : width;
  const mapH = Math.round(expanded ? Math.min(560, Math.max(300, mapW * 0.8)) : Math.min(400, Math.max(260, mapW * 0.72)));

  useEffect(() => {
    if (depth > prevDepth.current && !reduce) setGrow(0);
    else setGrow(1);
    prevDepth.current = depth;
    setSel(null);
  }, [depth, reduce]);
  useRafLoop((dt) => setGrow((g) => Math.min(1, g + dt * 1.8)), grow < 1);
  useInterval(
    () => {
      if (depth >= MAX_DEPTH) setGrowing(false);
      else setDepth((d) => d + 1);
    },
    growing ? 1300 : null
  );

  // map a path string to its node
  const nodeAt = (path: string | null) => {
    if (path === null) return null;
    let n: TreeNode | undefined = tree;
    for (const ch of path) n = ch === 'L' ? n?.left : n?.right;
    return n ?? null;
  };
  const selNode = nodeAt(sel);

  const ruleText = useMemo(() => {
    if (sel === null) return null;
    const parts: string[] = [];
    let n: TreeNode = tree;
    for (const ch of sel) {
      const thr = (n.threshold! * 10).toFixed(1);
      parts.push(`${FEAT[n.feature!]} ${ch === 'L' ? '<' : '≥'} ${thr}`);
      n = ch === 'L' ? n.left! : n.right!;
    }
    return { parts, node: n };
  }, [sel, tree]);

  const acc = ACC[depth];
  const internal = (function count(n: TreeNode): number {
    return n.feature === undefined ? 0 : 1 + count(n.left!) + count(n.right!);
  })(tree);
  const nLeaves = leaves(tree).length;
  const bestTest = Math.max(...ACC.map((a) => a.test));

  let note: { text: string; mood: 'smile' | 'happy' | 'think' | 'wow' | 'cheer'; tone?: 'good' | 'warn' | 'bad' };
  if (depth === 0)
    note = { text: `아직 질문이 하나도 없어요. 전부 한 덩어리라 다수결로 "${tree.counts[1] > tree.counts[0] ? '▲ 악성' : '● 양성'}"이라고만 답해요. 깊이를 1로 올려 첫 질문을 던져 보세요!`, mood: 'smile' };
  else if (depth === 1) {
    const after = (tree.left!.counts[0] + tree.left!.counts[1]) * tree.left!.gini + (tree.right!.counts[0] + tree.right!.counts[1]) * tree.right!.gini;
    const n = tree.counts[0] + tree.counts[1];
    note = {
      text: `첫 질문은 "${q(tree)}?" — 가능한 모든 칼질 위치를 다 훑어서, 불순도(뒤섞인 정도)를 ${tree.gini.toFixed(2)}에서 ${(after / n).toFixed(2)}로 가장 많이 줄이는 질문을 골랐어요.`,
      mood: 'happy'
    };
  } else if (acc.test >= bestTest - 0.001 || depth <= 3)
    note = { text: '질문을 몇 번 더 하니 시험 성적도 좋아졌어요. 경계가 계단 모양인 건 질문이 "크기 < ?"처럼 축에 나란한 칼질이라서예요.', mood: 'cheer', tone: 'good' };
  else
    note = {
      text: `훈련 정확도는 ${Math.round(acc.train * 100)}%로 오르는데, 처음 보는 시험 데이터 정확도는 ${Math.round(acc.test * 100)}%로 떨어졌어요. 작은 상자가 잡음 점 하나씩을 감싸는 "암기" — 과적합이에요!`,
      mood: 'wow',
      tone: 'bad'
    };

  // ---- tree diagram ----
  const diagW = wide ? width - mapW - 16 : width;
  const tDepth = Math.max(1, (function d(n: TreeNode): number {
    return n.feature === undefined ? n.depth : Math.max(d(n.left!), d(n.right!));
  })(tree));
  const diagH = wide ? mapH : Math.round(Math.min(320, 70 + tDepth * 44));
  const top = 26;
  const levelH = Math.min(70, (diagH - top - 30) / tDepth);
  const placed = width > 0 ? place(tree, diagW, top, levelH) : [];
  const onPath = (p: string) => sel !== null && sel.startsWith(p);

  const mapDraw = (ctx: CanvasRenderingContext2D, w: number, h: number) => {
    const lemon = getComputedStyle(document.documentElement).getPropertyValue('--lemon').trim() || '#e7a90f';
    // leaf regions
    for (const lf of leaves(tree)) {
      const [x0, y0, x1, y1] = lf.box;
      const tot = lf.counts[0] + lf.counts[1];
      const p = tot ? lf.counts[1] / tot : 0.5;
      const pur = Math.abs(p - 0.5) * 2;
      const col = p >= 0.5 ? c.series[2] : c.series[3];
      ctx.fillStyle = withAlpha(col, (c.theme === 'dark' ? 0.12 : 0.1) + 0.3 * pur);
      ctx.fillRect(x0 * w, (1 - y1) * h, (x1 - x0) * w, (y1 - y0) * h);
    }
    // split lines
    const walk = (n: TreeNode) => {
      if (n.feature === undefined) return;
      const [x0, y0, x1, y1] = n.box;
      const isNew = n.depth === depth - 1;
      const g = isNew ? grow : 1;
      ctx.strokeStyle = isNew && grow < 1 ? lemon : c.theme === 'dark' ? 'rgba(240,244,255,0.8)' : 'rgba(42,37,64,0.7)';
      ctx.lineWidth = Math.max(1.5, 4 - n.depth * 0.5);
      ctx.beginPath();
      if (n.feature === 0) {
        const px = n.threshold! * w;
        const mid = (1 - (y0 + y1) / 2) * h;
        const half = ((y1 - y0) * h * g) / 2;
        ctx.moveTo(px, mid - half);
        ctx.lineTo(px, mid + half);
      } else {
        const py = (1 - n.threshold!) * h;
        const mid = ((x0 + x1) / 2) * w;
        const half = ((x1 - x0) * w * g) / 2;
        ctx.moveTo(mid - half, py);
        ctx.lineTo(mid + half, py);
      }
      ctx.stroke();
      walk(n.left!);
      walk(n.right!);
    };
    walk(tree);
    if (selNode) {
      const [x0, y0, x1, y1] = selNode.box;
      ctx.strokeStyle = lemon;
      ctx.lineWidth = 4;
      ctx.strokeRect(x0 * w + 2, (1 - y1) * h + 2, (x1 - x0) * w - 4, (y1 - y0) * h - 4);
    }
  };

  return (
    <div ref={box}>
      <div className="vz-pad" style={{ paddingBottom: 0 }}>
        <div style={{ display: 'grid', gridTemplateColumns: wide ? `${mapW}px 1fr` : '1fr', gap: 16, alignItems: 'start' }}>
          <div>
            <div className="vz-panel-title" style={{ marginBottom: 6 }}>
              🗺️ 질문이 지도를 칼질해요 (칸을 누르면 그 규칙이 보여요)
            </div>
            <FieldCanvas
              height={mapH}
              points={TRAIN}
              hollow={showTest ? TEST : undefined}
              under={mapDraw}
              deps={[depth, grow, sel, showTest]}
              onPick={(x, y) => setSel(findLeafPath(tree, x, y))}
              ariaLabel="결정 트리가 축에 나란한 직선으로 지도를 나눈 모습"
              xLabel="크기 →"
              yLabel="↑ 울퉁불퉁한 정도"
            />
          </div>
          <div>
            <div className="vz-panel-title" style={{ marginBottom: 6 }}>
              🌳 스무고개 나무 (질문을 눌러 보세요)
            </div>
            {width > 0 && (
              <svg className="vz-svg" width={diagW} height={diagH} viewBox={`0 0 ${diagW} ${diagH}`} role="img" aria-label="결정 트리 그림" style={{ display: 'block', overflow: 'visible' }}>
                {placed.map((p) =>
                  p.parent ? (
                    <line
                      key={`e${p.path}`}
                      x1={p.parent.x}
                      y1={p.parent.y}
                      x2={p.x}
                      y2={p.y}
                      strokeWidth={onPath(p.path) ? 3.5 : 1.8}
                      opacity={p.node.depth === depth && grow < 1 ? grow : 1}
                      style={{ stroke: onPath(p.path) ? 'var(--lemon)' : c.axis }}
                    />
                  ) : null
                )}
                {placed
                  .filter((p) => p.parent && p.node.depth === 1 && p.room > 40)
                  .map((p) => (
                    <text
                      key={`yn${p.path}`}
                      x={(p.x + p.parent!.x) / 2 + (p.isLeft ? -8 : 8)}
                      y={(p.y + p.parent!.y) / 2}
                      textAnchor={p.isLeft ? 'end' : 'start'}
                      style={{ fontSize: 11.5, fontWeight: 800, fill: c.text }}
                    >
                      {p.isLeft ? '예' : '아니오'}
                    </text>
                  ))}
                {placed.map((p) => {
                  const n = p.node;
                  const tot = n.counts[0] + n.counts[1];
                  const frac = tot ? n.counts[1] / tot : 0.5;
                  const isLeaf = n.feature === undefined;
                  const opacity = n.depth === depth && grow < 1 ? grow : 1;
                  const selected = sel === p.path;
                  const showQ = !isLeaf && p.room >= 82;
                  const label = showQ ? `${q(n)}?` : '';
                  const bw = Math.min(p.room - 6, 8 + label.length * 7.4);
                  if (showQ) {
                    return (
                      <g key={p.path} opacity={opacity} style={{ cursor: 'pointer' }} onClick={() => setSel(p.path)}>
                        <rect
                          x={p.x - bw / 2}
                          y={p.y - 12}
                          width={bw}
                          height={24}
                          rx={9}
                          fill={c.surface}
                          strokeWidth={selected ? 3 : 1.5}
                          style={{ stroke: selected ? 'var(--lemon)' : c.axis }}
                        />
                        <text x={p.x} y={p.y + 4.5} textAnchor="middle" style={{ fontSize: 12, fontWeight: 800, fill: c.ink }}>
                          {label}
                        </text>
                        <rect x={p.x - bw / 2 + 6} y={p.y + 14} width={(bw - 12) * (1 - frac)} height={4} rx={2} fill={c.series[3]} />
                        <rect x={p.x - bw / 2 + 6 + (bw - 12) * (1 - frac)} y={p.y + 14} width={(bw - 12) * frac} height={4} rx={2} fill={c.series[2]} />
                      </g>
                    );
                  }
                  const r = isLeaf ? Math.max(4.5, Math.min(9, p.room / 3.2)) : Math.max(3.5, Math.min(6, p.room / 4));
                  const col = frac >= 0.5 ? c.series[2] : c.series[3];
                  return (
                    <g key={p.path} opacity={opacity} style={{ cursor: 'pointer' }} onClick={() => setSel(p.path)}>
                      <circle cx={p.x} cy={p.y} r={r + 6} fill="transparent" />
                      {isLeaf && frac >= 0.5 ? (
                        <path d={`M${p.x},${p.y - r * 1.15} L${p.x + r * 1.05},${p.y + r * 0.75} L${p.x - r * 1.05},${p.y + r * 0.75} Z`} fill={col} strokeWidth={selected ? 3 : 1.5} style={{ stroke: selected ? 'var(--lemon)' : c.surface }} />
                      ) : (
                        <circle cx={p.x} cy={p.y} r={r} fill={isLeaf ? col : c.surface} strokeWidth={selected ? 3 : 1.5} style={{ stroke: selected ? 'var(--lemon)' : isLeaf ? c.surface : c.axis }} />
                      )}
                      {isLeaf && p.room >= 30 && (
                        <text x={p.x} y={p.y + r + 13} textAnchor="middle" style={{ fontSize: 11, fontWeight: 750, fill: c.text }}>
                          {n.counts[1]}:{n.counts[0]}
                        </text>
                      )}
                    </g>
                  );
                })}
              </svg>
            )}
            <div style={{ fontSize: 12, color: 'var(--text-3)', marginTop: 4 }}>잎 아래 숫자 = ▲악성 : ●양성 개수 · 막대 = 뒤섞인 정도</div>
          </div>
        </div>
        <div style={{ marginTop: 8 }}>
          <ClassLegend extra={showTest ? <span className="vz-legend-item">속 빈 점 = 시험 데이터 (학습에 안 씀)</span> : undefined} />
        </div>
      </div>

      <div className="vz-pad" style={{ display: 'grid', gap: 12 }}>
        {ruleText && (
          <div
            style={{
              padding: '10px 14px',
              borderRadius: 14,
              background: 'var(--surface-2)',
              border: '2px solid var(--lemon)',
              fontSize: 14,
              lineHeight: 1.7
            }}
          >
            <b>📜 규칙:</b> {ruleText.parts.length ? ruleText.parts.join(' 그리고 ') : '(질문 없음 — 전체)'}{' '}
            {ruleText.node.feature === undefined ? (
              <>
                → <b style={{ color: ruleText.node.counts[1] >= ruleText.node.counts[0] ? c.series[2] : c.series[3] }}>{ruleText.node.counts[1] >= ruleText.node.counts[0] ? '▲ 악성' : '● 양성'}</b> (악성 {ruleText.node.counts[1]} : 양성 {ruleText.node.counts[0]}, 불순도 {gini(ruleText.node.counts).toFixed(2)})
              </>
            ) : (
              <>→ 다음 질문: “{q(ruleText.node)}?”</>
            )}
          </div>
        )}
        <Note mood={note.mood} tone={note.tone}>
          {note.text}
        </Note>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 12, alignItems: 'start' }}>
          <div className="vz-stats">
            <Stat label="깊이" value={depth} tone="brand" />
            <Stat label="질문 / 잎" value={`${internal} / ${nLeaves}`} />
            <Stat label="훈련 정확도" value={`${Math.round(acc.train * 100)}%`} />
            <Stat label="시험 정확도" value={`${Math.round(acc.test * 100)}%`} tone={acc.test >= bestTest - 0.001 ? 'good' : depth >= 5 ? 'bad' : undefined} />
          </div>
          <div>
            <div className="vz-panel-title">깊이에 따른 정확도</div>
            <Plot
              xDomain={[0, MAX_DEPTH]}
              yDomain={[0.4, 1]}
              height={140}
              margin={{ top: 8, right: 12, bottom: 34, left: 42 }}
              xTicks={8}
              yTicks={3}
              xLabel="나무 깊이"
              yFormat={(v) => `${Math.round(v * 100)}%`}
              ariaLabel="깊이에 따른 훈련·시험 정확도"
            >
              {({ x, y }) => (
                <g>
                  <line x1={x(depth)} x2={x(depth)} y1={y(0.4)} y2={y(1)} strokeWidth={2} strokeDasharray="4 3" style={{ stroke: 'var(--lemon)' }} />
                  <path d={linePath(ACC.map((a, i) => [x(i), y(a.train)]))} fill="none" stroke={c.series[0]} strokeWidth={3} />
                  <path d={linePath(ACC.map((a, i) => [x(i), y(a.test)]))} fill="none" stroke={c.series[1]} strokeWidth={3} />
                  <circle cx={x(depth)} cy={y(acc.train)} r={5} fill={c.series[0]} />
                  <circle cx={x(depth)} cy={y(acc.test)} r={5} fill={c.series[1]} />
                </g>
              )}
            </Plot>
            <Legend
              items={[
                { color: c.series[0], label: '훈련 (외운 문제)' },
                { color: c.series[1], label: '시험 (처음 보는 문제)' }
              ]}
            />
          </div>
        </div>
      </div>

      <Controls>
        <Slider label="나무 깊이 (질문을 몇 번까지?)" value={depth} min={0} max={MAX_DEPTH} step={1} onChange={(v) => setDepth(Math.round(v))} format={(v) => `${v}단계`} />
        <Button size="sm" variant="primary" onClick={() => (depth >= MAX_DEPTH ? (setDepth(0), setGrowing(true)) : setGrowing((g) => !g))}>
          {growing ? '❚❚ 멈춤' : '🌱 한 층씩 자라기'}
        </Button>
        <Toggle label="시험 데이터 보기" checked={showTest} onChange={setShowTest} />
      </Controls>
    </div>
  );
}
