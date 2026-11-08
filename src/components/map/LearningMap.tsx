import { useEffect, useMemo, useState } from 'react';
import Icon from '@/components/ui/Icon';
import { loadProgress, onProgress, resetProgress, type ProgressState } from '@/lib/progress';

export interface MapChapter {
  n: number;
  title: string;
  href: string;
  question: string;
}
export interface MapPart {
  part: number;
  title: string;
  chapters: MapChapter[];
}

type Status = 'todo' | 'visited' | 'done';
const STATUS_LABEL: Record<Status, string> = { todo: '아직 안 읽음', visited: '읽는 중', done: '완료' };

/**
 * 학습 지도: progress over the 28 chapters, part by part. Each chapter is a cell
 * that fills in as it is visited and completed (progress lives in localStorage).
 * The chapter you point at, or the next one to read, is shown below the grid.
 */
export default function LearningMap({ parts, totalViz, showReset = false }: { parts: MapPart[]; totalViz: number; showReset?: boolean }) {
  const [progress, setProgress] = useState<ProgressState>({ chapters: {}, labs: {}, viz: {} });
  const [hover, setHover] = useState<number | null>(null);

  useEffect(() => {
    const read = () => setProgress(loadProgress());
    read();
    return onProgress(read);
  }, []);

  const stops = useMemo(() => parts.flatMap((p) => p.chapters.map((c) => ({ ...c, part: p.part, partTitle: p.title }))), [parts]);
  const status = (n: number): Status => (progress.chapters[n]?.done ? 'done' : progress.chapters[n]?.visited ? 'visited' : 'todo');
  const doneCount = stops.filter((s) => status(s.n) === 'done').length;
  const visitedCount = stops.filter((s) => status(s.n) !== 'todo').length;
  const vizCount = Object.keys(progress.viz).length;
  const next = stops.find((s) => status(s.n) !== 'done') ?? stops[stops.length - 1];
  const focus = stops.find((s) => s.n === hover) ?? next;
  const pct = stops.length ? Math.round((doneCount / stops.length) * 100) : 0;

  return (
    <div className="lmap">
      <div className="lmap-top">
        <dl className="lmap-stats">
          <div>
            <dt>완료한 장</dt>
            <dd>
              {doneCount}
              <span>/{stops.length}</span>
            </dd>
          </div>
          <div>
            <dt>읽은 장</dt>
            <dd>{visitedCount}</dd>
          </div>
          <div>
            <dt>해 본 시각화</dt>
            <dd>
              {vizCount}
              <span>/{totalViz}</span>
            </dd>
          </div>
        </dl>
        <div className="lmap-actions">
          <a className="btn btn-primary" href={next.href}>
            {doneCount === 0 && visitedCount === 0 ? '1장부터 시작하기' : `이어서 읽기: ${next.n}장`}
            <Icon name="arrow-right" size={16} />
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

      <div className="lmap-bar" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct} aria-label="완료한 장의 비율">
        <span style={{ width: `${pct}%` }} />
      </div>

      <ol className="lmap-parts">
        {parts.map((p) => (
          <li key={p.part} className="lmap-part">
            <span className="lmap-part-name">
              <span className="lmap-part-no">{p.part}부</span>
              {p.title}
            </span>
            <span className="lmap-cells">
              {p.chapters.map((c) => {
                const st = status(c.n);
                return (
                  <a
                    key={c.n}
                    href={c.href}
                    className={`lmap-cell lmap-${st}${focus.n === c.n ? ' is-focus' : ''}`}
                    aria-label={`${c.n}장 ${c.title} (${STATUS_LABEL[st]})`}
                    onMouseEnter={() => setHover(c.n)}
                    onMouseLeave={() => setHover(null)}
                    onFocus={() => setHover(c.n)}
                    onBlur={() => setHover(null)}
                  >
                    {st === 'done' ? <Icon name="check" size={14} strokeWidth={2.25} /> : c.n}
                  </a>
                );
              })}
            </span>
          </li>
        ))}
      </ol>

      <div className="lmap-legend" aria-hidden="true">
        <span><i className="lmap-dot lmap-todo" />아직 안 읽음</span>
        <span><i className="lmap-dot lmap-visited" />읽는 중</span>
        <span><i className="lmap-dot lmap-done" />완료</span>
      </div>

      <div className="lmap-card card">
        <span className="lmap-card-meta">
          {focus.part}부 · {focus.partTitle} <span className={`lmap-tag lmap-tag-${status(focus.n)}`}>{STATUS_LABEL[status(focus.n)]}</span>
        </span>
        <h3>
          {focus.n}장. {focus.title}
        </h3>
        {focus.question && <p>{focus.question}</p>}
        <a className="lmap-card-link" href={focus.href}>
          읽으러 가기 <Icon name="arrow-right" size={15} />
        </a>
      </div>
    </div>
  );
}
