import { useEffect, useMemo, useRef, useState, type PointerEvent as RPointerEvent, type ReactNode } from 'react';
import type { VizProps } from '../Viz';
import { Button, Controls, Note, Segmented, Toggle } from '../core/ui';
import { useCanvas, useElementSize, useRafLoop } from '../core/hooks';
import { mixHex, useVizColors, withAlpha, type VizColors } from '../core/colors';
import { clamp, rng } from '../core/math';
import { frameSpectrum, hzToMel, melToHz, stft } from './_fft';
import { hexToRgb } from './_draw';

/* A sound → its wiggles (waveform) → a prism at one instant (FFT) → the whole score (spectrogram). */

const SR = 8000;
const DUR = 2;
const LEN = SR * DUR;
/** Analysis window lengths (samples at 8 kHz). The book uses 25 ms; longer windows split close notes better. */
const WINDOWS: { n: number; label: string; tip: string }[] = [
  { n: 256, label: '짧게 32ms', tip: '짧은 창: "언제"는 또렷하지만 음높이는 흐릿해요. 화음의 세 줄이 뭉개지죠?' },
  { n: 512, label: '보통 64ms', tip: '중간 창: 시간과 음높이를 적당히 함께 보는 균형점이에요.' },
  { n: 2048, label: '길게 256ms', tip: '긴 창: 음높이는 아주 또렷하지만 "언제"가 흐려져요. 박수가 옆으로 번지죠?' }
];
const HOP = 80;
const FMAX = 4000;
const ROWS = 160;
const ML = 44;
const MR = 10;
const FONT = 'Pretendard Variable, Pretendard, system-ui, sans-serif';

type Key = 'do' | 'sol' | 'chord' | 'whistle' | 'vowel' | 'clap';
const PRESETS: { key: Key; label: string }[] = [
  { key: 'do', label: '🎹 도' },
  { key: 'sol', label: '🎹 솔' },
  { key: 'chord', label: '🎶 도·미·솔 화음' },
  { key: 'whistle', label: '😗 휘파람' },
  { key: 'vowel', label: '🗣️ 아·이·우' },
  { key: 'clap', label: '👏 박수' }
];

/** Vowel formants (F1, F2) in Hz — rough adult values. */
const VOWELS: { name: string; t0: number; t1: number; F: [number, number] }[] = [
  { name: '아', t0: 0.05, t1: 0.6, F: [800, 1200] },
  { name: '이', t0: 0.72, t1: 1.27, F: [300, 2300] },
  { name: '우', t0: 1.39, t1: 1.95, F: [350, 800] }
];

const env = (t: number, a: number, r: number, t0: number, t1: number) => (t < t0 || t > t1 ? 0 : Math.min(1, (t - t0) / a, (t1 - t) / r));

function synth(key: Key): Float32Array<ArrayBuffer> {
  const s = new Float32Array(LEN);
  const TAU = Math.PI * 2;
  if (key === 'do' || key === 'sol') {
    const f = key === 'do' ? 261.63 : 392;
    for (let i = 0; i < LEN; i++) {
      const t = i / SR;
      const e = env(t, 0.01, 0.15, 0.1, 1.9) * Math.exp(-Math.max(0, t - 0.1) / 1.1);
      if (!e) continue;
      let v = 0;
      for (let k = 1; k <= 8 && k * f < FMAX - 100; k++) v += Math.sin(TAU * k * f * t) / k ** 1.1;
      s[i] = v * e;
    }
  } else if (key === 'chord') {
    for (let i = 0; i < LEN; i++) {
      const t = i / SR;
      const e = env(t, 0.03, 0.12, 0.1, 1.9);
      if (!e) continue;
      let v = 0;
      for (const f of [261.63, 329.63, 392]) for (let k = 1; k <= 4; k++) v += Math.sin(TAU * k * f * t + k) / k ** 1.4;
      s[i] = v * e;
    }
  } else if (key === 'whistle') {
    for (let i = 0; i < LEN; i++) {
      const t = i / SR;
      // frequency 1200 - 600·cos(πt): 600 Hz → 1800 Hz (at 1 s) → 600 Hz
      const ph = TAU * (1200 * t - (600 / Math.PI) * Math.sin(Math.PI * t));
      s[i] = (Math.sin(ph) + 0.08 * Math.sin(2 * ph)) * env(t, 0.06, 0.1, 0.05, 1.95);
    }
  } else if (key === 'vowel') {
    const f0 = 140;
    const K = Math.floor((FMAX - 100) / f0);
    for (const vw of VOWELS) {
      const amps: number[] = [];
      for (let k = 1; k <= K; k++) {
        const f = k * f0;
        const gain = 1 / (1 + ((f - vw.F[0]) / 90) ** 2) + 0.7 / (1 + ((f - vw.F[1]) / 120) ** 2) + 0.15 / (1 + ((f - 2700) / 160) ** 2);
        amps.push((gain + 0.02) / k ** 0.6);
      }
      for (let i = Math.floor(vw.t0 * SR); i < Math.ceil(vw.t1 * SR) && i < LEN; i++) {
        const t = i / SR;
        const e = env(t, 0.04, 0.06, vw.t0, vw.t1);
        let v = 0;
        for (let k = 1; k <= K; k++) v += amps[k - 1] * Math.sin(TAU * k * f0 * t + k * 0.7);
        s[i] = v * e;
      }
    }
  } else {
    const r = rng(7);
    for (let i = 0; i < LEN; i++) {
      const t = i / SR;
      let v = 0;
      for (const b of [0.3, 0.9, 1.5]) if (t >= b && t < b + 0.22) v += r.normal() * Math.exp(-(t - b) / 0.03);
      s[i] = v;
    }
  }
  let peak = 1e-9;
  for (const v of s) peak = Math.max(peak, Math.abs(v));
  for (let i = 0; i < LEN; i++) s[i] = (s[i] / peak) * 0.8;
  return s;
}

const NOTE_KO = ['도', '도#', '레', '레#', '미', '파', '파#', '솔', '솔#', '라', '라#', '시'];
const NOTE_EN = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
function noteName(f: number) {
  const midi = Math.round(69 + 12 * Math.log2(f / 440));
  const n = ((midi % 12) + 12) % 12;
  return `${NOTE_KO[n]}(${NOTE_EN[n]}${Math.floor(midi / 12) - 1})`;
}

/** Row (0 = bottom) → frequency, on a linear or mel axis. */
const rowFreq = (r: number, mel: boolean) => (mel ? melToHz((r / ROWS) * hzToMel(FMAX)) : (r / ROWS) * FMAX);
const freqFrac = (f: number, mel: boolean) => (mel ? hzToMel(f) / hzToMel(FMAX) : f / FMAX);

function makeLut(c: VizColors): [number, number, number][] {
  const stops = [c.bg, mixHex(c.bg, c.series[0], 0.45), c.series[0], c.series[5], c.series[4]].map(hexToRgb);
  const lut: [number, number, number][] = [];
  for (let i = 0; i < 256; i++) {
    const x = (i / 255) * (stops.length - 1);
    const k = Math.min(stops.length - 2, Math.floor(x));
    const t = x - k;
    lut.push([0, 1, 2].map((j) => Math.round(stops[k][j] + (stops[k + 1][j] - stops[k][j]) * t)) as [number, number, number]);
  }
  return lut;
}

/** Sample a magnitude spectrum (bins of SR/N Hz) at frequency f with linear interpolation. */
function at(spec: ArrayLike<number>, f: number, binHz: number) {
  const b = f / binHz;
  const i = Math.floor(b);
  if (i >= spec.length - 1) return spec[spec.length - 1];
  return spec[i] + (spec[i + 1] - spec[i]) * (b - i);
}

const LIVE_COLS = 260;

interface Live {
  ac: AudioContext;
  stream: MediaStream;
  an: AnalyserNode;
  freq: Float32Array<ArrayBuffer>;
  time: Float32Array<ArrayBuffer>;
}

export default function WaveToSpectrogram({ expanded }: VizProps) {
  const c = useVizColors();
  const [key, setKey] = useState<Key>('do');
  const [t, setT] = useState(0.5);
  const [playing, setPlaying] = useState(false);
  const [db, setDb] = useState(true);
  const [mel, setMel] = useState(false);
  const [nfft, setNfft] = useState(512);
  const [live, setLive] = useState(false);
  const [liveTick, setLiveTick] = useState(0);
  const [micMsg, setMicMsg] = useState<string | null>(null);
  const [box, { width }] = useElementSize<HTMLDivElement>();
  const narrow = width > 0 && width < 600;

  const signal = useMemo(() => synth(key), [key]);
  const frames = useMemo(() => stft(signal, nfft, HOP), [signal, nfft]);
  const lut = useMemo(() => makeLut(c), [c]);
  const maxMag = useMemo(() => {
    let m = 1e-9;
    for (const fr of frames) for (const v of fr) m = Math.max(m, v);
    return m;
  }, [frames]);
  const level = (mag: number) => (db ? clamp((20 * Math.log10(mag / maxMag + 1e-9) + 55) / 55, 0, 1) : clamp(mag / maxMag, 0, 1));

  /* ---------- spectrogram image (frames × ROWS) ---------- */
  const specImg = useMemo(() => {
    const cv = document.createElement('canvas');
    cv.width = frames.length;
    cv.height = ROWS;
    const ctx = cv.getContext('2d')!;
    const img = ctx.createImageData(frames.length, ROWS);
    const binHz = SR / nfft;
    for (let x = 0; x < frames.length; x++)
      for (let r = 0; r < ROWS; r++) {
        const v = level(at(frames[x], rowFreq(r + 0.5, mel), binHz));
        const [R, G, B] = lut[Math.round(v * 255)];
        const k = ((ROWS - 1 - r) * frames.length + x) * 4;
        img.data[k] = R;
        img.data[k + 1] = G;
        img.data[k + 2] = B;
        img.data[k + 3] = 255;
      }
    ctx.putImageData(img, 0, 0);
    return cv;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frames, mel, db, lut, maxMag]);

  /* ---------- audio ---------- */
  const audio = useRef<{ ac: AudioContext | null; src: AudioBufferSourceNode | null; at: number; offset: number; wall: number }>({ ac: null, src: null, at: 0, offset: 0, wall: 0 });
  const liveRef = useRef<Live | null>(null);
  const liveCv = useRef<HTMLCanvasElement | null>(null);

  const stopAudio = () => {
    const a = audio.current;
    if (a.src) {
      a.src.onended = null;
      try {
        a.src.stop();
      } catch {
        /* already stopped */
      }
      a.src = null;
    }
  };
  const play = (from: number) => {
    stopAudio();
    const a = audio.current;
    const off = from >= DUR - 0.02 ? 0 : from;
    a.offset = off;
    a.wall = performance.now();
    try {
      const AC = window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (AC) {
        a.ac ??= new AC();
        void a.ac.resume();
        const buf = a.ac.createBuffer(1, LEN, SR);
        buf.copyToChannel(signal, 0);
        const src = a.ac.createBufferSource();
        src.buffer = buf;
        const gain = a.ac.createGain();
        gain.gain.value = 0.55;
        src.connect(gain).connect(a.ac.destination);
        src.start(0, off);
        a.src = src;
        a.at = a.ac.currentTime;
      }
    } catch {
      a.src = null;
    }
    setT(off);
    setPlaying(true);
  };
  const pause = () => {
    stopAudio();
    setPlaying(false);
  };

  useRafLoop(() => {
    const a = audio.current;
    const now = a.src && a.ac ? a.offset + (a.ac.currentTime - a.at) : a.offset + (performance.now() - a.wall) / 1000;
    if (now >= DUR) {
      stopAudio();
      setPlaying(false);
      setT(DUR);
    } else setT(now);
  }, playing && !live);

  useEffect(() => {
    pause();
    setT(0.5);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  /* ---------- microphone (optional) ---------- */
  const stopMic = () => {
    const L = liveRef.current;
    if (L) {
      L.stream.getTracks().forEach((tr) => tr.stop());
      void L.ac.close();
    }
    liveRef.current = null;
    setLive(false);
  };
  const startMic = async () => {
    setMicMsg(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setMicMsg('이 브라우저(또는 http 주소)에서는 마이크를 쓸 수 없어요. 대신 위의 소리 예시로 충분히 실험할 수 있어요!');
      return;
    }
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
    } catch {
      setMicMsg('마이크 사용이 허락되지 않았어요. 주소창 옆의 마이크 권한을 확인하거나, 위의 소리 예시를 써 보세요.');
      return;
    }
    pause();
    const ac = new AudioContext();
    const an = ac.createAnalyser();
    an.fftSize = 2048;
    an.smoothingTimeConstant = 0.15;
    ac.createMediaStreamSource(stream).connect(an);
    liveRef.current = { ac, stream, an, freq: new Float32Array(an.frequencyBinCount), time: new Float32Array(an.fftSize) };
    const cv = document.createElement('canvas');
    cv.width = LIVE_COLS;
    cv.height = ROWS;
    const ctx = cv.getContext('2d')!;
    ctx.fillStyle = `rgb(${lut[0].join(',')})`;
    ctx.fillRect(0, 0, LIVE_COLS, ROWS);
    liveCv.current = cv;
    setLive(true);
  };
  useEffect(
    () => () => {
      stopAudio();
      const L = liveRef.current;
      if (L) {
        L.stream.getTracks().forEach((tr) => tr.stop());
        void L.ac.close();
      }
      void audio.current.ac?.close();
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    []
  );
  const liveLevel = (dB: number) => clamp((dB + 100) / 70, 0, 1);
  useRafLoop(() => {
    const L = liveRef.current;
    const cv = liveCv.current;
    if (!L || !cv) return;
    L.an.getFloatFrequencyData(L.freq);
    L.an.getFloatTimeDomainData(L.time);
    const ctx = cv.getContext('2d')!;
    ctx.drawImage(cv, -2, 0);
    const binHz = L.ac.sampleRate / L.an.fftSize;
    for (let r = 0; r < ROWS; r++) {
      const v = liveLevel(at(L.freq, rowFreq(r + 0.5, mel), binHz));
      const [R, G, B] = lut[Math.round(v * 255)];
      ctx.fillStyle = `rgb(${R},${G},${B})`;
      ctx.fillRect(LIVE_COLS - 2, ROWS - 1 - r, 2, 1);
    }
    setLiveTick((n) => n + 1);
  }, live);

  /* ---------- derived: spectrum at the playhead ---------- */
  const startSample = clamp(Math.round(t * SR) - nfft / 2, 0, LEN - nfft);
  const spectrum = useMemo(() => frameSpectrum(signal, startSample, nfft), [signal, startSample, nfft]);
  const binHz = SR / nfft;
  const peaks = useMemo(() => {
    if (live) return [];
    const mags = Array.from(spectrum);
    const m = Math.max(1e-9, ...mags);
    const out: { f: number; v: number }[] = [];
    for (let k = 2; k < mags.length - 2; k++) {
      const v = mags[k];
      if (v < m * 0.12 || v < mags[k - 1] || v < mags[k + 1] || v < mags[k - 2] || v < mags[k + 2]) continue;
      // parabolic interpolation on log magnitude for a sharper frequency estimate
      const a = Math.log(mags[k - 1] + 1e-12);
      const b = Math.log(v + 1e-12);
      const g = Math.log(mags[k + 1] + 1e-12);
      const d = (0.5 * (a - g)) / (a - 2 * b + g);
      out.push({ f: (k + (Number.isFinite(d) ? d : 0)) * binHz, v });
    }
    return out.sort((p, q) => q.v - p.v).slice(0, key === 'chord' ? 3 : 2);
  }, [spectrum, live, binHz, key]);
  const quiet = Math.max(...Array.from(spectrum)) < maxMag * 0.01;

  /* ---------- canvases ---------- */
  const H_WAVE = 70;
  const H_PRISM = narrow ? 120 : 140;
  const H_SPEC = expanded ? 300 : narrow ? 180 : 210;
  const tx = (time: number, w: number) => ML + (time / DUR) * (w - ML - MR);

  const envelope = useMemo(() => {
    const B = 600;
    const mn = new Float32Array(B);
    const mx = new Float32Array(B);
    for (let b = 0; b < B; b++) {
      let lo = 0;
      let hi = 0;
      for (let i = Math.floor((b * LEN) / B); i < Math.floor(((b + 1) * LEN) / B); i++) {
        lo = Math.min(lo, signal[i]);
        hi = Math.max(hi, signal[i]);
      }
      mn[b] = lo;
      mx[b] = hi;
    }
    return { mn, mx, B };
  }, [signal]);

  const label = (ctx: CanvasRenderingContext2D, s: string, x: number, y: number, align: CanvasTextAlign = 'left', color = c.text, weight = 700) => {
    ctx.font = `${weight} 10.5px ${FONT}`;
    ctx.textAlign = align;
    ctx.fillStyle = color;
    ctx.fillText(s, x, y);
  };

  const [waveCv] = useCanvas(
    (ctx, w, h) => {
      const mid = h / 2;
      ctx.fillStyle = withAlpha(c.series[3], 0.06);
      ctx.fillRect(ML, 0, w - ML - MR, h);
      ctx.strokeStyle = c.grid;
      ctx.beginPath();
      ctx.moveTo(ML, mid);
      ctx.lineTo(w - MR, mid);
      ctx.stroke();
      ctx.strokeStyle = c.series[3];
      ctx.lineWidth = 1;
      ctx.beginPath();
      const pw = w - ML - MR;
      for (let px = 0; px < pw; px++) {
        const b = Math.floor((px / pw) * envelope.B);
        ctx.moveTo(ML + px + 0.5, mid - envelope.mx[b] * mid * 0.95);
        ctx.lineTo(ML + px + 0.5, mid - envelope.mn[b] * mid * 0.95 + 0.5);
      }
      ctx.stroke();
      // 30 ms zoom window marker
      const zx0 = tx(clamp(t - 0.015, 0, DUR), w);
      const zx1 = tx(clamp(t + 0.015, 0, DUR), w);
      ctx.fillStyle = withAlpha(c.series[2], 0.25);
      ctx.fillRect(zx0, 0, Math.max(2, zx1 - zx0), h);
      ctx.strokeStyle = c.series[2];
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(tx(t, w), 0);
      ctx.lineTo(tx(t, w), h);
      ctx.stroke();
      label(ctx, '떨림', ML - 6, mid + 4, 'right');
    },
    [envelope, t, c, narrow, live]
  );

  const [zoomCv] = useCanvas(
    (ctx, w, h) => {
      const mid = h / 2;
      ctx.fillStyle = withAlpha(c.series[2], 0.07);
      ctx.fillRect(0, 0, w, h);
      ctx.strokeStyle = c.grid;
      ctx.beginPath();
      ctx.moveTo(0, mid);
      ctx.lineTo(w, mid);
      ctx.stroke();
      ctx.strokeStyle = c.series[2];
      ctx.lineWidth = 2;
      ctx.beginPath();
      if (live && liveRef.current) {
        const td = liveRef.current.time;
        let pk = 0.05;
        for (const v of td) pk = Math.max(pk, Math.abs(v));
        for (let i = 0; i < td.length; i++) {
          const x = (i / (td.length - 1)) * w;
          const y = mid - (td[i] / pk) * mid * 0.9;
          if (i) ctx.lineTo(x, y);
          else ctx.moveTo(x, y);
        }
      } else {
        const n = Math.round(0.03 * SR);
        const s0 = Math.round(t * SR) - n / 2;
        for (let i = 0; i <= n; i++) {
          const v = signal[s0 + i] ?? 0;
          const x = (i / n) * w;
          const y = mid - v * mid * 1.1;
          if (i) ctx.lineTo(x, y);
          else ctx.moveTo(x, y);
        }
      }
      ctx.stroke();
      label(ctx, live ? '마이크 실시간' : '30ms 확대', 6, 13, 'left', c.text, 800);
    },
    [t, signal, c, live, liveTick]
  );

  const [prismCv] = useCanvas(
    (ctx, w, h) => {
      const MB = 20;
      const pw = w - ML - MR;
      const ph = h - MB - 6;
      const n = 256;
      const L = liveRef.current;
      const liveBin = L ? L.ac.sampleRate / L.an.fftSize : 1;
      for (let k = 0; k < n; k++) {
        const f = ((k + 0.5) / n) * FMAX;
        const v = live && L ? liveLevel(at(L.freq, f, liveBin)) : level(at(spectrum, f, binHz));
        const bh = v * ph;
        ctx.fillStyle = `hsl(${(k / n) * 280}, 85%, ${c.theme === 'dark' ? 60 : 55}%)`;
        ctx.fillRect(ML + (k / n) * pw, 6 + ph - bh, pw / n + 0.6, bh);
      }
      ctx.strokeStyle = c.axis;
      ctx.beginPath();
      ctx.moveTo(ML, 6 + ph + 0.5);
      ctx.lineTo(w - MR, 6 + ph + 0.5);
      ctx.stroke();
      for (const f of [0, 1000, 2000, 3000, 4000]) label(ctx, f === 0 ? '0' : f === FMAX ? '4kHz' : `${f / 1000}k`, ML + (f / FMAX) * pw, h - 5, f === FMAX ? 'right' : 'center', c.text);
      label(ctx, '세기', ML - 6, 16, 'right');
      for (const pk of peaks) {
        const x = ML + (pk.f / FMAX) * pw;
        const txt = pk.f >= 120 && pk.f <= 2100 && key !== 'vowel' && key !== 'clap' ? `${Math.round(pk.f)}Hz ≈ ${noteName(pk.f)}` : `${Math.round(pk.f)}Hz`;
        ctx.font = `800 11px ${FONT}`;
        const tw = ctx.measureText(txt).width + 10;
        const lx = clamp(x - tw / 2, ML, w - MR - tw);
        const ly = 8 + peaks.indexOf(pk) * 19;
        ctx.fillStyle = withAlpha(c.surface, 0.92);
        ctx.strokeStyle = c.ink;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.roundRect(lx, ly, tw, 17, 6);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = c.ink;
        ctx.textAlign = 'left';
        ctx.fillText(txt, lx + 5, ly + 12.5);
      }
    },
    [spectrum, peaks, db, c, live, liveTick, key]
  );

  const [specCv] = useCanvas(
    (ctx, w, h) => {
      const MB = 20;
      const pw = w - ML - MR;
      const ph = h - MB - 4;
      ctx.imageSmoothingEnabled = true;
      if (live) {
        if (liveCv.current) ctx.drawImage(liveCv.current, ML, 4, pw, ph);
      } else {
        // each column is one window, centred N/2 samples after its start
        ctx.fillStyle = `rgb(${lut[0].join(',')})`;
        ctx.fillRect(ML, 4, pw, ph);
        const t0 = (nfft / 2 - HOP / 2) / SR;
        const t1 = ((frames.length - 1) * HOP + nfft / 2 + HOP / 2) / SR;
        ctx.drawImage(specImg, tx(t0, w), 4, tx(t1, w) - tx(t0, w), ph);
      }
      ctx.strokeStyle = c.border;
      ctx.strokeRect(ML + 0.5, 4.5, pw - 1, ph - 1);
      for (const f of mel ? [0, 250, 500, 1000, 2000, 4000] : [0, 1000, 2000, 3000, 4000]) {
        const y = 4 + ph - freqFrac(f, mel) * ph;
        label(ctx, f >= 1000 ? `${f / 1000}k` : String(f), ML - 6, clamp(y + 4, 12, h - MB), 'right');
        ctx.strokeStyle = withAlpha(c.surface, 0.18);
        ctx.beginPath();
        ctx.moveTo(ML, y);
        ctx.lineTo(w - MR, y);
        ctx.stroke();
      }
      if (live) {
        label(ctx, '← 조금 전', ML + 4, h - 5, 'left');
        label(ctx, '지금', w - MR, h - 5, 'right', c.text, 900);
      } else {
        for (const s of [0, 0.5, 1, 1.5, 2]) label(ctx, `${s}초`, clamp(tx(s, w), ML + 10, w - MR - 10), h - 5, 'center');
        // vowel labels
        if (key === 'vowel') for (const vw of VOWELS) label(ctx, vw.name, tx((vw.t0 + vw.t1) / 2, w), 20, 'center', '#ffffff', 900);
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(tx(t, w), 4);
        ctx.lineTo(tx(t, w), 4 + ph);
        ctx.stroke();
        ctx.strokeStyle = c.series[2];
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(tx(t, w), 4);
        ctx.lineTo(tx(t, w), 4 + ph);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    },
    [specImg, t, c, mel, live, liveTick, key, nfft, frames, lut]
  );

  /* ---------- scrubbing ---------- */
  const scrub = useRef<{ was: boolean } | null>(null);
  const timeAt = (e: RPointerEvent<HTMLCanvasElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    return clamp(((e.clientX - r.left - ML) / (r.width - ML - MR)) * DUR, 0, DUR);
  };
  const scrubHandlers = {
    onPointerDown: (e: RPointerEvent<HTMLCanvasElement>) => {
      if (live) return;
      e.currentTarget.setPointerCapture(e.pointerId);
      scrub.current = { was: playing };
      if (playing) pause();
      setT(timeAt(e));
    },
    onPointerMove: (e: RPointerEvent<HTMLCanvasElement>) => {
      if (scrub.current) setT(timeAt(e));
    },
    onPointerUp: (e: RPointerEvent<HTMLCanvasElement>) => {
      if (!scrub.current) return;
      const was = scrub.current.was;
      scrub.current = null;
      if (was) play(timeAt(e));
    }
  };

  const notes: Record<Key, { text: ReactNode; mood: 'smile' | 'happy' | 'think' | 'wow' | 'cheer' }> = {
    do: { text: <>"도" 한 음인데 줄이 여러 개예요! 맨 아래 262Hz(기본음) 위로 2배·3배… 배음이 쌓여 있어요. 배음의 세기 비율이 악기마다 다른 "음색"이에요.</>, mood: 'wow' },
    sol: { text: <>"솔"(392Hz)은 "도"보다 높아서 줄 전체가 위로 올라가요. 높은 음 = 빠른 떨림 = 큰 주파수!</>, mood: 'happy' },
    chord: { text: <>도·미·솔을 동시에! 파형만 보면 복잡한 꼬불꼬불인데, 프리즘과 스펙트로그램에선 세 음이 깔끔하게 나뉘어요. 이게 푸리에 변환의 힘이에요.</>, mood: 'cheer' },
    whistle: { text: <>음이 올라갔다 내려오면 줄이 언덕 모양이 돼요. 시간에 따라 바뀌는 주파수를 한 장에 — 그래서 "소리의 악보"예요.</>, mood: 'happy' },
    vowel: { text: <>같은 높이(140Hz)로 말해도 아·이·우는 밝은 띠(포먼트)의 위치가 달라요. 음성 인식 모델은 바로 이 무늬를 "읽어요".</>, mood: 'think' },
    clap: { text: <>짝! 소리는 모든 주파수가 한꺼번에 터져서 세로 줄이 생겨요. 줄무늬 방향만 봐도 소리 종류가 보이죠? 이 그림을 CNN에 그대로 넣을 수 있어요.</>, mood: 'wow' }
  };
  const note = live ? { text: <>🎤 말하거나 노래하거나 휘파람을 불어 보세요! 오른쪽 끝이 "지금"이고, 왼쪽으로 흘러가요.</>, mood: 'cheer' as const } : notes[key];

  const panelTitle = (s: string) => (
    <div className="vz-panel-title" style={{ marginBottom: 4 }}>
      {s}
    </div>
  );

  return (
    <div ref={box}>
      <div className="vz-pad" style={{ display: 'grid', gap: 12 }}>
        {!live && (
          <div style={{ display: 'grid', gridTemplateColumns: narrow ? '1fr' : 'minmax(0, 1fr) 34%', gap: 10 }}>
            <div>
              {panelTitle('① 파형 — 공기가 떨리는 모양 (2초)')}
              <canvas ref={waveCv} className="vz-canvas" style={{ height: H_WAVE, cursor: 'ew-resize' }} role="img" aria-label="소리의 파형. 눌러서 재생 위치를 옮길 수 있어요" {...scrubHandlers} onPointerCancel={() => (scrub.current = null)} />
            </div>
            <div>
              {panelTitle('🔍 주황 칸 확대')}
              <canvas ref={zoomCv} className="vz-canvas" style={{ height: H_WAVE, borderRadius: 10, touchAction: 'pan-y' }} role="img" aria-label="30밀리초 확대 파형" />
            </div>
          </div>
        )}
        {live && (
          <div>
            {panelTitle('① 파형 — 마이크로 들어온 떨림')}
            <canvas ref={zoomCv} className="vz-canvas" style={{ height: H_WAVE, borderRadius: 10, touchAction: 'pan-y' }} role="img" aria-label="마이크 실시간 파형" />
          </div>
        )}
        <div>
          {panelTitle(live ? '② 프리즘 — 지금 들리는 소리의 주파수' : `② 프리즘 — ${t.toFixed(2)}초 순간의 주파수 (푸리에 변환)`)}
          <canvas ref={prismCv} className="vz-canvas" style={{ height: H_PRISM, touchAction: 'pan-y' }} role="img" aria-label="재생 위치에서의 주파수 막대그래프" />
          {!live && quiet && <div style={{ fontSize: 12, color: 'var(--text-3)' }}>지금 위치는 조용해요 — 파형이나 스펙트로그램을 눌러 소리가 있는 곳으로 옮겨 보세요.</div>}
        </div>
        <div>
          {panelTitle('③ 스펙트로그램 — 프리즘을 시간 순서로 이어 붙인 "소리의 악보"')}
          <canvas ref={specCv} className="vz-canvas" style={{ height: H_SPEC, cursor: live ? 'default' : 'ew-resize' }} role="img" aria-label="스펙트로그램: 가로는 시간, 세로는 주파수, 밝기는 세기" {...scrubHandlers} onPointerCancel={() => (scrub.current = null)} />
          {!live && (
            <div style={{ fontSize: 12, color: 'var(--text-3)', marginTop: 4 }}>
              {(nfft / 8).toLocaleString()}ms 창을 10ms씩 밀며 만든 그림 · {WINDOWS.find((wd) => wd.n === nfft)!.tip}
            </div>
          )}
        </div>
        <div className="vz-playbar">
          {!live && (
            <Button variant="primary" onClick={() => (playing ? pause() : play(t))}>
              {playing ? '❚❚ 멈춤' : '▶ 소리 듣기'}
            </Button>
          )}
          <Button variant={live ? 'danger' : 'ghost'} onClick={() => (live ? stopMic() : void startMic())}>
            {live ? '■ 마이크 끄기' : '🎤 내 목소리로 해보기'}
          </Button>
          <span style={{ fontSize: 12, color: 'var(--text-3)' }}>{live ? '소리는 이 기기 밖으로 나가지 않아요.' : '파형이나 스펙트로그램을 눌러 시간을 옮길 수 있어요.'}</span>
        </div>
        {micMsg && (
          <Note mood="think" tone="warn">
            {micMsg}
          </Note>
        )}
        <Note mood={note.mood}>{note.text}</Note>
      </div>
      <Controls>
        {!live && <Segmented label="소리 고르기" options={PRESETS.map((p) => ({ value: p.key, label: p.label }))} value={key} onChange={setKey} />}
        {!live && <Segmented label="창 길이" options={WINDOWS.map((wd) => ({ value: wd.n, label: wd.label }))} value={nfft} onChange={setNfft} />}
        <Toggle label="세기를 dB(로그)로" checked={db} onChange={setDb} />
        <Toggle label="멜 눈금 (사람 귀처럼)" checked={mel} onChange={setMel} />
      </Controls>
    </div>
  );
}
