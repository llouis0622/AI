/** Minimal radix-2 FFT + STFT for the audio visualization. */

/** In-place iterative FFT on real/imag arrays of length 2^k. */
export function fft(re: Float64Array, im: Float64Array) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang);
    const wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1;
      let ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const ar = re[i + k];
        const ai = im[i + k];
        const br = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
        const bi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
        re[i + k] = ar + br;
        im[i + k] = ai + bi;
        re[i + k + len / 2] = ar - br;
        im[i + k + len / 2] = ai - bi;
        const ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = ncr;
      }
    }
  }
}

/** Magnitude spectrum (first n/2 bins) of one Hann-windowed frame starting at `start`. */
export function frameSpectrum(signal: Float32Array, start: number, n: number): Float64Array {
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const s = signal[start + i] ?? 0;
    re[i] = s * (0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1)));
  }
  fft(re, im);
  const mag = new Float64Array(n / 2);
  for (let k = 0; k < n / 2; k++) mag[k] = Math.hypot(re[k], im[k]) / (n / 4);
  return mag;
}

/** Short-time Fourier transform: frames × (n/2) magnitudes. */
export function stft(signal: Float32Array, n: number, hop: number): Float64Array[] {
  const frames: Float64Array[] = [];
  for (let start = 0; start + n <= signal.length; start += hop) frames.push(frameSpectrum(signal, start, n));
  return frames;
}

export const hzToMel = (f: number) => 2595 * Math.log10(1 + f / 700);
export const melToHz = (m: number) => 700 * (10 ** (m / 2595) - 1);
