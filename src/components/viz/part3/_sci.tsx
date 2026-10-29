/** Scientific-notation helpers that render real superscripts (HTML <sup> / SVG <tspan>). */

export function sciParts(v: number, digits = 1): { mant: string; exp: number | null } {
  if (!Number.isFinite(v)) return { mant: v > 0 ? '∞' : '-∞', exp: null };
  if (v === 0) return { mant: '0', exp: null };
  const a = Math.abs(v);
  if (a >= 1e-3 && a < 1e4) {
    const s = a >= 100 ? v.toFixed(0) : a >= 1 ? v.toFixed(2) : v.toFixed(4).replace(/0+$/, '').replace(/\.$/, '');
    return { mant: s, exp: null };
  }
  const e = Math.floor(Math.log10(a));
  return { mant: (v / 10 ** e).toFixed(digits), exp: e };
}

/** HTML: 8.7×10⁻¹⁹ with a real <sup>. */
export function Sci({ v, digits = 1 }: { v: number; digits?: number }) {
  const { mant, exp } = sciParts(v, digits);
  if (exp === null) return <>{mant}</>;
  return (
    <>
      {mant}×10<sup style={{ fontSize: '0.68em', marginLeft: 1 }}>{exp}</sup>
    </>
  );
}

/** SVG tick label "10^e" drawn with a raised exponent. */
export function PowTen({ e, x, y, anchor = 'end', className = 'vz-tick' }: { e: number; x: number; y: number; anchor?: 'start' | 'middle' | 'end'; className?: string }) {
  if (e === 0) {
    return (
      <text x={x} y={y} textAnchor={anchor} className={className}>
        1
      </text>
    );
  }
  return (
    <text x={x} y={y} textAnchor={anchor} className={className}>
      10
      <tspan dy={-5} style={{ fontSize: '0.78em' }}>
        {e}
      </tspan>
    </text>
  );
}
