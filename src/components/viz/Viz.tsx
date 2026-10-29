import { Component, useCallback, useEffect, useRef, useState, type ComponentType, type ReactNode } from 'react';
import { vizById } from '@/data/viz';
import { markViz } from '@/lib/progress';
import Algo from '@/components/mascot/Algo';

/** Props every visualization component receives from the host. */
export interface VizProps {
  /** Optional per-placement settings from MDX: <Viz id="…" params={{ … }} /> */
  params?: Record<string, unknown>;
  /** true while the frame is in full-screen mode (use it to grow the stage). */
  expanded?: boolean;
}

type Loaded = { default: ComponentType<VizProps> };
const loaders = import.meta.glob<Loaded>(['./part*/*.tsx', './labs/*.tsx']);

type Phase = 'poster' | 'loading' | 'ready' | 'error';

function hasWebGL() {
  try {
    const c = document.createElement('canvas');
    return !!(c.getContext('webgl2') || c.getContext('webgl'));
  } catch {
    return false;
  }
}

/**
 * Host for every visualization: draws the frame (title, mascot hint, expand /
 * reset buttons), lazy-loads the component chunk, and gates heavy 3D scenes
 * behind a "▶ 3D로 실행" poster so pages stay light.
 */
export default function Viz({ id, params }: { id: string; params?: Record<string, unknown> }) {
  const meta = vizById[id];
  const [phase, setPhase] = useState<Phase>('poster');
  const [Comp, setComp] = useState<ComponentType<VizProps> | null>(null);
  const [mountKey, setMountKey] = useState(0);
  const [expanded, setExpanded] = useState(false);
  const [noGL, setNoGL] = useState(false);
  const touched = useRef(false);

  const load = useCallback(async () => {
    if (!meta) return;
    const loader = loaders[`./${meta.component}.tsx`];
    if (!loader) {
      setPhase('error');
      return;
    }
    if (meta.is3D && !hasWebGL()) {
      setNoGL(true);
      setPhase('error');
      return;
    }
    setPhase('loading');
    try {
      const mod = await loader();
      setComp(() => mod.default);
      setPhase('ready');
    } catch {
      setPhase('error');
    }
  }, [meta]);

  useEffect(() => {
    if (meta && !meta.is3D) load();
  }, [meta, load]);

  useEffect(() => {
    if (!expanded) return;
    document.documentElement.classList.add('modal-open');
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setExpanded(false);
    window.addEventListener('keydown', onKey);
    return () => {
      document.documentElement.classList.remove('modal-open');
      window.removeEventListener('keydown', onKey);
    };
  }, [expanded]);

  const onInteract = () => {
    if (touched.current || phase !== 'ready' || !meta) return;
    touched.current = true;
    markViz(meta.id);
  };

  if (!meta) {
    return (
      <div className="vz-frame">
        <div className="vz-error">🛠️ 준비 중인 시각화예요 ({id})</div>
      </div>
    );
  }

  return (
    <figure className={`vz-frame${expanded ? ' is-expanded' : ''}`} id={`viz-${meta.id}`} data-viz={meta.id} aria-label={meta.title}>
      <figcaption className="vz-head">
        <span className="vz-icon" aria-hidden="true">{meta.icon}</span>
        <div className="vz-titles">
          <h3 className="vz-title">
            {meta.title}
            {meta.is3D ? <span className="vz-badge vz-badge-3d">3D</span> : <span className="vz-badge">직접 해보기</span>}
          </h3>
          <p className="vz-tagline">{meta.tagline}</p>
        </div>
        <div className="vz-actions">
          {phase === 'ready' && (
            <button className="vz-action" type="button" onClick={() => setMountKey((k) => k + 1)} title="처음 상태로" aria-label="처음 상태로 되돌리기">
              ↺
            </button>
          )}
          <button
            className="vz-action"
            type="button"
            onClick={() => setExpanded((v) => !v)}
            title={expanded ? '닫기 (Esc)' : '크게 보기'}
            aria-label={expanded ? '크게 보기 닫기' : '크게 보기'}
          >
            {expanded ? '✕' : '⤢'}
          </button>
        </div>
      </figcaption>

      <div className="vz-body vz-container" onPointerDown={onInteract} onKeyDown={onInteract}>
        {phase === 'ready' && Comp ? (
          <ErrorBoundary onRetry={() => setMountKey((k) => k + 1)}>
            <Comp key={mountKey} params={params} expanded={expanded} />
          </ErrorBoundary>
        ) : (
          <Poster meta={meta} phase={phase} noGL={noGL} onPlay={load} />
        )}
      </div>

      {meta.howTo && (
        <div className="vz-foot">
          <Algo mood="wink" size={30} still />
          <span>{meta.howTo}</span>
        </div>
      )}
    </figure>
  );
}

function Poster({
  meta,
  phase,
  noGL,
  onPlay
}: {
  meta: { icon: string; is3D?: boolean; aspect?: number };
  phase: Phase;
  noGL: boolean;
  onPlay: () => void;
}) {
  const aspect = meta.aspect ?? (meta.is3D ? 16 / 10 : 16 / 10);
  return (
    <div className="vz-poster" style={{ aspectRatio: String(aspect) }}>
      <div className="vz-poster-grid" />
      <div className="vz-poster-inner">
        <span className="vz-poster-emoji" aria-hidden="true">{meta.icon}</span>
        {phase === 'error' ? (
          <p className="vz-poster-note">
            {noGL ? '이 기기(브라우저)에서는 3D 그래픽을 켤 수 없어요. 다른 브라우저로 열어 보세요.' : '시각화를 불러오지 못했어요. 새로고침해 볼까요?'}
          </p>
        ) : meta.is3D && phase === 'poster' ? (
          <>
            <button className="vz-play" type="button" onClick={onPlay}>
              <span className="vz-play-ring">▶</span>3D로 실행하기
            </button>
            <span className="vz-poster-note">드래그로 돌리고, 휠·두 손가락으로 확대해요</span>
          </>
        ) : (
          <span className="vz-loading">
            <span className="vz-spinner" aria-hidden="true" />
            불러오는 중…
          </span>
        )}
      </div>
    </div>
  );
}

class ErrorBoundary extends Component<{ children: ReactNode; onRetry: () => void }, { error: boolean }> {
  state = { error: false };
  static getDerivedStateFromError() {
    return { error: true };
  }
  componentDidCatch(err: unknown) {
    console.error('[viz]', err);
  }
  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="vz-error">
        <span style={{ fontSize: 36 }}>🤖💦</span>
        <p>앗, 시각화가 잠깐 멈췄어요.</p>
        <button
          className="vz-btn"
          type="button"
          onClick={() => {
            this.setState({ error: false });
            this.props.onRetry();
          }}
        >
          다시 시도
        </button>
      </div>
    );
  }
}
