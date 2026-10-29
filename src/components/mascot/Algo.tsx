export type AlgoMood = 'smile' | 'happy' | 'think' | 'wow' | 'wink' | 'cheer';

interface Props {
  mood?: AlgoMood;
  size?: number;
  /** Draw the little body with arms (hero and big callouts). */
  body?: boolean;
  /** Disable idle float/blink (e.g. tiny inline icons). */
  still?: boolean;
  className?: string;
  title?: string;
}

/**
 * 알고 — the AIology guide robot. Pure SVG, colors come from --algo-* tokens so it
 * turns into a neon outline in dark mode. Safe to render server-side.
 */
export default function Algo({ mood = 'smile', size = 64, body = false, still = false, className = '', title }: Props) {
  const h = body ? 140 : 100;
  const label = title ?? '안내 로봇 알고';
  return (
    <svg
      className={`algo algo-${mood}${still ? ' algo-still' : ''} ${className}`}
      width={size}
      height={(size * h) / 120}
      viewBox={`0 0 120 ${h}`}
      role="img"
      aria-label={label}
    >
      <g className="algo-float">
        {/* antenna */}
        <line x1="60" y1="20" x2="60" y2="8" className="algo-line" strokeWidth="3.5" strokeLinecap="round" />
        <circle cx="60" cy="7" r="5.5" className="algo-bulb" />
        {/* ears */}
        <rect x="10" y="44" width="11" height="22" rx="5.5" className="algo-ear" />
        <rect x="99" y="44" width="11" height="22" rx="5.5" className="algo-ear" />
        {/* head */}
        <rect x="18" y="20" width="84" height="68" rx="28" className="algo-head" strokeWidth="3.5" />
        {/* face screen */}
        <rect x="29" y="32" width="62" height="44" rx="19" className="algo-screen" />
        <circle cx="34" cy="70" r="4.5" className="algo-cheek" />
        <circle cx="86" cy="70" r="4.5" className="algo-cheek" />
        <Face mood={mood} />
        {body && <Body mood={mood} />}
      </g>
    </svg>
  );
}

function Face({ mood }: { mood: AlgoMood }) {
  const eye = 'algo-eye';
  switch (mood) {
    case 'happy':
    case 'cheer':
      return (
        <g>
          <path d="M41 56 q6 -8 12 0" className={`${eye}-stroke`} strokeWidth="4" fill="none" strokeLinecap="round" />
          <path d="M67 56 q6 -8 12 0" className={`${eye}-stroke`} strokeWidth="4" fill="none" strokeLinecap="round" />
          <path d="M51 63 q9 9 18 0 z" className={eye} />
        </g>
      );
    case 'think':
      return (
        <g>
          <g className="algo-eyes">
            <rect x="46" y="44" width="8" height="12" rx="4" className={eye} />
            <rect x="70" y="44" width="8" height="12" rx="4" className={eye} />
          </g>
          <path d="M53 66 h12" className={`${eye}-stroke`} strokeWidth="3.5" strokeLinecap="round" />
        </g>
      );
    case 'wow':
      return (
        <g>
          <g className="algo-eyes">
            <circle cx="47" cy="52" r="6.5" className={eye} />
            <circle cx="73" cy="52" r="6.5" className={eye} />
          </g>
          <ellipse cx="60" cy="66" rx="4.5" ry="5" className={eye} />
        </g>
      );
    case 'wink':
      return (
        <g>
          <rect x="43" y="46" width="8" height="12" rx="4" className={eye} />
          <path d="M67 53 q6 -7 12 0" className={`${eye}-stroke`} strokeWidth="4" fill="none" strokeLinecap="round" />
          <path d="M51 63 q9 8 18 0" className={`${eye}-stroke`} strokeWidth="3.5" fill="none" strokeLinecap="round" />
        </g>
      );
    default:
      return (
        <g>
          <g className="algo-eyes">
            <rect x="43" y="46" width="8" height="12" rx="4" className={eye} />
            <rect x="69" y="46" width="8" height="12" rx="4" className={eye} />
          </g>
          <path d="M51 63 q9 8 18 0" className={`${eye}-stroke`} strokeWidth="3.5" fill="none" strokeLinecap="round" />
        </g>
      );
  }
}

function Body({ mood }: { mood: AlgoMood }) {
  const armsUp = mood === 'cheer';
  return (
    <g>
      <rect x="38" y="92" width="44" height="34" rx="14" className="algo-head" strokeWidth="3.5" />
      <path d="M60 103 l3 4.5 h-6 z M54 108 h12" className="algo-heart" />
      <circle cx="60" cy="109" r="5" className="algo-bulb" />
      {armsUp ? (
        <>
          <path d="M38 102 q-12 -6 -14 -20" className="algo-line" strokeWidth="5" fill="none" strokeLinecap="round" />
          <path d="M82 102 q12 -6 14 -20" className="algo-line" strokeWidth="5" fill="none" strokeLinecap="round" />
        </>
      ) : (
        <>
          <path d="M38 102 q-12 4 -12 16" className="algo-line algo-arm-l" strokeWidth="5" fill="none" strokeLinecap="round" />
          <path d="M82 102 q12 4 12 16" className="algo-line algo-arm-r" strokeWidth="5" fill="none" strokeLinecap="round" />
        </>
      )}
    </g>
  );
}
