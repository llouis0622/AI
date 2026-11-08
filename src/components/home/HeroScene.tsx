import { lazy, Suspense, useEffect, useState } from 'react';

const Canvas3D = lazy(() => import('./HeroCanvas'));

function canRun3D() {
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return false;
  try {
    const c = document.createElement('canvas');
    return !!(c.getContext('webgl2') || c.getContext('webgl'));
  } catch {
    return false;
  }
}

/**
 * Home hero: a ball keeps rolling down a loss landscape, which is "learning" in
 * one picture. The WebGL chunk loads only after hydration; until then (and for
 * reduced-motion / no-WebGL visitors) a static illustration is shown.
 */
export default function HeroScene() {
  const [ok, setOk] = useState(false);
  useEffect(() => setOk(canRun3D()), []);
  return (
    <div className="hero-scene" aria-hidden="true">
      <div className="hero-fallback">
        <svg viewBox="0 0 400 300" className="hero-fallback-svg">
          {[0, 1, 2, 3, 4, 5].map((k) => (
            <ellipse key={k} cx="200" cy="190" rx={40 + k * 28} ry={16 + k * 11} fill="none" stroke="var(--accent)" strokeOpacity={0.7 - k * 0.1} strokeWidth="1.5" />
          ))}
          <path d="M60 70 Q120 120 150 150 T200 190" fill="none" stroke="var(--accent-2)" strokeWidth="2.5" strokeDasharray="2 8" strokeLinecap="round" />
          <circle cx="200" cy="190" r="9" fill="var(--accent-2)" />
        </svg>
      </div>
      {ok && (
        <Suspense fallback={null}>
          <Canvas3D />
        </Suspense>
      )}
    </div>
  );
}
