import { useEffect, useMemo, useState } from 'react';
import Algo from '@/components/mascot/Algo';
import { loadProgress, onProgress, resetProgress, type ProgressState } from '@/lib/progress';
import { useElementSize } from '@/components/viz/core/hooks';

export interface MapChapter {
  n: number;
  title: string;
  href: string;
  question: string;
}
export interface MapPart {
  part: number;
  title: string;
  emoji: string;
  chapters: MapChapter[];
}

const PART_VARS: Record<number, string> = {
  1: '--mint',
  2: '--lemon',
  3: '--lav',
  4: '--sky',
  5: '--peach',
  6: '--pink',
  7: '--brand',
  8: '--lime',
  9: '--amber'
};

/**
 * 학습 지도: 28 chapters as stops on a winding road. Stops light up as chapters
 * are visited / completed (progress lives in localStorage), and 알고 waits at
 * the next chapter to read.
 */
export default function LearningMap({ parts, totalViz, showReset = false }: { parts: MapPart[]; totalViz: number; showReset?: boolean }) {
  const [progress, setProgress] = useState<ProgressState>({ chapters: {}, labs: {}, viz: {} });
  const [hover, setHover] = useState<number | null>(null);
  const [box, { width }] = useElementSize<HTMLDivElement>();

  useEffect(() => {
    const read = () => setProgress(loadProgress());
    read();
    return onProgress(read);
  }, []);

  const stops = useMemo(() => parts.flatMap((p) => p.chapters.map((c) => ({ ...c, part: p.part, emoji: p.emoji, partTitle: p.title }))), [parts]);
  const status = (n: number) => (progress.chapters[n]?.done ? 'done' : progress.chapters[n]?.visited ? 'visited' : 'todo');
  const doneCount = stops.filter((s) => status(s.n) === 'done').length;
  const visitedCount = stops.filter((s) => status(s.n) !== 'todo').length;
  const vizCount = Object.keys(progress.viz).length;
  const next = stops.find((s) => status(s.n) !== 'done') ?? stops[stops.length - 1];
  const focus = stops.find((s) => s.n === hover) ?? next;

  // serpentine layout
  const w = Math.max(width, 300);
  const cols = w >= 1000 ? 7 : w >= 760 ? 6 : w >= 560 ? 4 : 3;
  const padX = w < 560 ? 44 : 64;
  const stepX = (w - padX * 2) / (cols - 1);
  const stepY = w < 560 ? 104 : 118;
  const padY = 64;
  const pts = stops.map((s, i) => {
    const row = Math.floor(i / cols);
    const k = i % cols;
    const col = row % 2 === 0 ? k : cols - 1 - k;
    return { ...s, x: padX + col * stepX, y: padY + row * stepY, row };
  });
  const rows = Math.ceil(stops.length / cols);
  const h = padY * 2 + (rows - 1) * stepY + 20;
  const r = w < 560 ? 19 : 23;

  const seg = (a: (typeof pts)[number], b: (typeof pts)[number]) => {
    if (a.row === b.row) return `L${b.x} ${b.y}`;
    const side = a.x > w / 2 ? 1 : -1;
    const bulge = Math.min(56, padX - 6) * side;
    return `C${a.x + bulge} ${a.y} ${b.x + bulge} ${b.y} ${b.x} ${b.y}`;
  };
  const fullPath = pts.length ? `M${pts[0].x} ${pts[0].y}` + pts.slice(1).map((p, i) => seg(pts[i], p)).join('') : '';
  const lastDoneIdx = (() => {
    let k = -1;
    pts.forEach((p, i) => {
      if (status(p.n) === 'done') k = i;
    });
    return k;
  })();
  const donePath = lastDoneIdx > 0 ? `M${pts[0].x} ${pts[0].y}` + pts.slice(1, lastDoneIdx + 1).map((p, i) => seg(pts[i], p)).join('') : '';
  const nextPt = pts.find((p) => p.n === next.n);
  const ring = 2 * Math.PI * 30;

  return (
    <div className="lmap" ref={box}>
      <div className="lmap-top">
        <div className="lmap-ring" aria-label={`28장 중 ${doneCount}장 완료`}>
          <svg viewBox="0 0 72 72" width="72" height="72">
            <circle cx="36" cy="36" r="30" fill="none" stroke="var(--surface-3)" strokeWidth="8" />
            <circle
              cx="36"
              cy="36"
              r="30"
              fill="none"
              stroke="url(#lmap-grad)"
              strokeWidth="8"
              strokeLinecap="round"
              strokeDasharray={`${(doneCount / stops.length) * ring} ${ring}`}
              transform="rotate(-90 36 36)"
              style={{ transition: 'stroke-dasharray .8s' }}
            />
            <defs>
              <linearGradient id="lmap-grad" x1="0" x2="1" y1="0" y2="1">
                <stop offset="0" stopColor="var(--lav)" />
                <stop offset="1" stopColor="var(--mint)" />
              </linearGradient>
            </defs>
            <text x="36" y="41" textAnchor="middle" style={{ fontSize: 16, fontWeight: 850, fill: 'var(--text)' }}>
              {Math.round((doneCount / stops.length) * 100)}%
            </text>
          </svg>
        </div>
        <div className="lmap-stats">
          <strong>
            {doneCount === 0 ? '여정을 시작해 볼까요?' : doneCount === stops.length ? '28장 완주! 정말 대단해요 🏆' : `${doneCount}장 완료! 잘하고 있어요 👏`}
          </strong>
          <span>
            읽은 장 {visitedCount}/{stops.length} · 완료 {doneCount} · 해 본 시각화 {vizCount}/{totalViz}
          </span>
        </div>
        <div className="lmap-actions">
          <a className="btn btn-primary" href={next.href}>
            {doneCount === 0 && visitedCount === 0 ? '1장부터 시작하기' : `이어서: ${next.n}장`} →
          </a>
          {showReset && (
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              onClick={() => {
                if (window.confirm('학습 기록(읽은 장, 완료 표시, 해 본 시각화)을 모두 지울까요?')) resetProgress();
              }}
            >
              기록 지우기
            </button>
          )}
        </div>
      </div>

      {width > 0 && (
        <svg className="lmap-svg" width={w} height={h} viewBox={`0 0 ${w} ${h}`} role="group" aria-label="28장 학습 지도">
          <path d={fullPath} fill="none" stroke="var(--border-strong)" strokeWidth={w < 560 ? 10 : 12} strokeLinecap="round" strokeLinejoin="round" />
          <path d={fullPath} fill="none" stroke="var(--surface-solid)" strokeWidth={w < 560 ? 4 : 5} strokeDasharray="2 14" strokeLinecap="round" opacity={0.9} />
          {donePath && <path d={donePath} fill="none" stroke="url(#lmap-road)" strokeWidth={w < 560 ? 10 : 12} strokeLinecap="round" strokeLinejoin="round" className="lmap-done-road" />}
          <defs>
            <linearGradient id="lmap-road" x1="0" x2="1">
              <stop offset="0" stopColor="var(--lav)" />
              <stop offset="0.5" stopColor="var(--sky)" />
              <stop offset="1" stopColor="var(--mint)" />
            </linearGradient>
          </defs>
          {pts.map((p, i) => {
            const st = status(p.n);
            const color = `var(${PART_VARS[p.part]})`;
            const firstOfPart = i === 0 || pts[i - 1].part !== p.part;
            return (
              <g key={p.n}>
                {firstOfPart && (
                  <g transform={`translate(${p.x} ${p.y - r - 16})`}>
                    <rect x={-34} y={-13} width={68} height={22} rx={11} fill={color} opacity={0.18} />
                    <text textAnchor="middle" y={3} style={{ fontSize: 11.5, fontWeight: 850, fill: 'var(--text)' }}>
                      {p.emoji} {p.part}부
                    </text>
                  </g>
                )}
                <a
                  href={p.href}
                  aria-label={`${p.n}장 ${p.title}${st === 'done' ? ' (완료)' : st === 'visited' ? ' (읽는 중)' : ''}`}
                  onMouseEnter={() => setHover(p.n)}
                  onMouseLeave={() => setHover(null)}
                  onFocus={() => setHover(p.n)}
                  onBlur={() => setHover(null)}
                  className={`lmap-stop lmap-${st}${hover === p.n ? ' is-hover' : ''}`}
                >
                  {st === 'done' && <circle cx={p.x} cy={p.y} r={r + 7} fill={color} opacity={0.22} className="lmap-halo" />}
                  <circle cx={p.x} cy={p.y} r={r} fill={st === 'done' ? color : 'var(--surface-solid)'} stroke={color} strokeWidth={st === 'todo' ? 2.5 : 4} />
                  {st === 'visited' && <circle cx={p.x} cy={p.y} r={r - 7} fill={color} opacity={0.25} />}
                  <text x={p.x} y={p.y + 5} textAnchor="middle" style={{ fontSize: w < 560 ? 13 : 15, fontWeight: 900, fill: st === 'done' ? '#fff' : 'var(--text)' }}>
                    {st === 'done' ? '✓' : p.n}
                  </text>
                  {st === 'done' && (
                    <text x={p.x} y={p.y + r + 16} textAnchor="middle" style={{ fontSize: 11, fontWeight: 800, fill: 'var(--text-2)' }}>
                      {p.n}장
                    </text>
                  )}
                </a>
              </g>
            );
          })}
          {nextPt && (
            <foreignObject x={nextPt.x + r - 8} y={nextPt.y - r - 34} width={52} height={56} style={{ overflow: 'visible', pointerEvents: 'none' }}>
              <div className="lmap-algo">
                <Algo mood={doneCount ? 'happy' : 'smile'} size={48} />
              </div>
            </foreignObject>
          )}
        </svg>
      )}

      <div className="lmap-card card" data-part={focus.part}>
        <span className="part-badge">
          {focus.emoji} {focus.part}부 · {focus.partTitle}
        </span>
        <h3>
          {focus.n}장. {focus.title}
          <small className={`lmap-tag lmap-tag-${status(focus.n)}`}>{status(focus.n) === 'done' ? '완료 ✓' : status(focus.n) === 'visited' ? '읽는 중' : '아직 안 읽음'}</small>
        </h3>
        <p>{focus.question}</p>
        <a className="btn btn-sm" href={focus.href}>
          읽으러 가기 →
        </a>
      </div>
    </div>
  );
}
