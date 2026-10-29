import { useEffect, useState } from 'react';

/**
 * Wide/narrow layout switch with hysteresis, so a page scrollbar appearing or
 * disappearing (≈15px) can't make the layout flip back and forth forever.
 */
export function useWide(width: number, threshold: number, margin = 28) {
  const [wide, setWide] = useState(false);
  useEffect(() => {
    if (!wide && width > threshold + margin) setWide(true);
    else if (wide && width > 0 && width < threshold - margin) setWide(false);
  }, [width, threshold, margin, wide]);
  return wide;
}
