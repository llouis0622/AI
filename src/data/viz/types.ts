export interface VizMeta {
  /** Stable id used in MDX: <Viz id="gradient-descent-3d" client:visible /> */
  id: string;
  title: string;
  /** One line, middle-school level: what you can do with it. */
  tagline: string;
  /** Emoji shown in the frame header and gallery card. */
  icon: string;
  /** Component path relative to src/components/viz/, without extension. */
  component: string;
  /** Chapter that hosts it (book). */
  chapter?: number;
  /** Code lab that hosts it. */
  lab?: number;
  /** Heavy WebGL scene: shown behind a "▶ 3D로 실행" button until clicked. */
  is3D?: boolean;
  /** Mascot hint shown under the visualization. */
  howTo?: string;
  /** Placeholder aspect ratio (width / height) before the component loads. */
  aspect?: number;
  /** Show on the home page. */
  featured?: boolean;
}
