import type { ReactNode } from 'react';
import type { VizProps } from '../Viz';
import { CodeStory, type StoryStep } from '../core/story';
import type { VizColors } from '../core/colors';
import { VB, FS, seg, alpha, rich, dd, T, Arrow, Pill } from './_LabKitB';

type Role = 'user' | 'assistant' | 'result' | 'error' | 'final';
interface Msg {
  role: Role;
  badge: string;
  lines: string[];
  mono?: boolean;
}

/* One real-looking run of the lab's agent: list → (wrong path) → parallel reads → calculator → answer. */
const MSGS: Msg[] = [
  { role: 'user', badge: 'user', lines: ['data/ CSV 매출 합계 계산해줘'] },
  { role: 'assistant', badge: 'assistant · tool_use', lines: ['list_files(directory="data")'], mono: true },
  { role: 'result', badge: 'user · tool_result', lines: ['data/jan.csv, data/feb.csv'], mono: true },
  { role: 'assistant', badge: 'assistant · tool_use', lines: ['read_file(path="jan.csv")'], mono: true },
  { role: 'error', badge: 'user · tool_result', lines: ['오류: jan.csv 파일이 없습니다.'] },
  { role: 'assistant', badge: 'assistant · tool_use ×2', lines: ['read_file("data/jan.csv")', 'read_file("data/feb.csv")'], mono: true },
  { role: 'result', badge: 'user · tool_result ×2 (한 메시지)', lines: ['jan.csv → 합계 1540', 'feb.csv → 합계 2210'] },
  { role: 'assistant', badge: 'assistant · tool_use', lines: ['calculator("1540 + 2210")'], mono: true },
  { role: 'result', badge: 'user · tool_result', lines: ['3750'], mono: true },
  { role: 'final', badge: 'assistant · end_turn', lines: ['1·2월 매출 합계는 3,750입니다.'] }
];

const roleColor = (c: VizColors, r: Role) =>
  r === 'user' ? c.series[3] : r === 'assistant' ? c.series[0] : r === 'error' ? c.neg : r === 'final' ? c.pos : c.series[1];

const cardH = (m: Msg) => 26 + m.lines.length * 17;

/** Message history column: shows the last few messages, newest ones animate in. */
function History({ c, t, count, fresh }: { c: VizColors; t: number; count: number; fresh: number }) {
  const all = MSGS.slice(0, count);
  // keep what fits in the column (newest at the bottom)
  const maxH = 262;
  const shown: Msg[] = [];
  let h = 0;
  for (let i = all.length - 1; i >= 0; i--) {
    const ch = cardH(all[i]) + 8;
    if (h + ch > maxH) break;
    shown.unshift(all[i]);
    h += ch;
  }
  const hidden = all.length - shown.length;
  let y = hidden > 0 ? 34 : 20;
  const nodes: ReactNode[] = [];
  if (hidden > 0) {
    nodes.push(
      <T key="more" x={286} y={22} anchor="middle" size={FS.m} weight={750} color="var(--text-3)">
        ⋯ 이전 메시지 {hidden}개
      </T>
    );
  }
  shown.forEach((m, k) => {
    const idx = hidden + k;
    const isFresh = idx >= count - fresh;
    const a = isFresh ? seg(t, 0.25 + (idx - (count - fresh)) * 0.25, 0.3) : 1;
    const col = roleColor(c, m.role);
    const hh = cardH(m);
    nodes.push(
      <g key={idx} opacity={a} transform={`translate(${(1 - a) * 18} 0)`}>
        <rect x={164} y={y} width={246} height={hh} rx={10} fill={alpha(col, isFresh ? 0.2 : 0.1)} stroke={col} strokeWidth={isFresh ? 2 : 1.2} />
        <T x={174} y={y + 16} size={11.5} weight={850} color={col}>
          {m.badge}
        </T>
        {m.lines.map((ln, j) => (
          <T key={j} x={174} y={y + 33 + j * 17} size={FS.m} weight={650} mono={m.mono}>
            {ln}
          </T>
        ))}
      </g>
    );
    y += hh + 8;
  });
  return <g>{nodes}</g>;
}

const NODES = [
  { label: '🤖 모델', angle: -90 },
  { label: '🛠️ 도구', angle: 30 },
  { label: '📨 결과', angle: 150 }
];
const LC = { x: 80, y: 118, r: 52 };
const nodePos = (deg: number): [number, number] => [LC.x + LC.r * Math.cos((deg * Math.PI) / 180), LC.y + LC.r * Math.sin((deg * Math.PI) / 180)];

/** The while-loop: 모델 → 도구 → 결과 → 모델 …, with a turn gauge and the stop_reason. */
function Loop({ c, t, active, turn, stop, exit = false }: { c: VizColors; t: number; active: number[]; turn: number; stop: string; exit?: boolean }) {
  const arcs = [0, 1, 2].map((i) => {
    const a0 = NODES[i].angle + 26;
    const a1 = NODES[(i + 1) % 3].angle - 26 + (i === 2 ? 360 : 0);
    const [x0, y0] = nodePos(a0);
    const [x1, y1] = nodePos(a1);
    return { d: `M${x0} ${y0} A${LC.r} ${LC.r} 0 0 1 ${x1} ${y1}`, x1, y1, a1 };
  });
  return (
    <g>
      {arcs.map((ar, i) => (
        <g key={i}>
          <path d={ar.d} fill="none" stroke="var(--border-strong)" strokeWidth={2.5} />
          <circle cx={ar.x1} cy={ar.y1} r={3.5} fill="var(--border-strong)" />
        </g>
      ))}
      {NODES.map((n, i) => {
        const [x, y] = nodePos(n.angle);
        const on = active.includes(i);
        const col = [c.series[0], c.series[2], c.series[1]][i];
        return (
          <g key={i}>
            {on && <circle cx={x} cy={y} r={27 + 3 * Math.sin(t * Math.PI)} fill={alpha(col, 0.22)} />}
            <circle cx={x} cy={y} r={22} fill="var(--surface-solid)" stroke={col} strokeWidth={on ? 3 : 1.5} />
            <T x={x} y={y + 5} anchor="middle" size={FS.m} weight={850}>
              {n.label.split(' ')[0]}
            </T>
            <T x={x} y={i === 0 ? y - 30 : y + 40} anchor="middle" size={FS.m} weight={800} color={on ? 'var(--text)' : 'var(--text-3)'}>
              {n.label.split(' ')[1]}
            </T>
          </g>
        );
      })}
      <T x={LC.x} y={216} anchor="middle" size={FS.s} weight={850}>
        turn {turn} / 15
      </T>
      {Array.from({ length: 15 }, (_, k) => (
        <rect key={k} x={14 + k * 9.6} y={226} width={7.6} height={10} rx={2.5} fill={k <= turn ? c.series[0] : 'var(--surface-3)'} />
      ))}
      <Pill x={LC.x} y={262} label={`stop_reason: ${stop}`} color={stop === 'end_turn' ? c.pos : c.series[2]} t={seg(t, 0.1, 0.3)} size={11.5} mono />
      {exit && <Arrow x1={LC.x + 24} y1={LC.y - LC.r} x2={152} y2={40} color={c.pos} t={seg(t, 0.4, 0.3)} label="return" labelDy={-6} />}
    </g>
  );
}

const steps: StoryStep[] = [
  {
    title: '할 일을 받고 첫 호출',
    text: rich('대화 이력 `messages`는 사용자의 할 일 하나로 시작해요. 그리고 **최대 15번 도는 for 루프** 안에서 모델을 부르죠. 이 15번 제한이 폭주를 막는 첫 번째 안전장치예요.'),
    code: dd(String.raw`def run_agent(task: str) -> str:
    client = Anthropic()
    messages = [{"role": "user", "content": task}]

    for turn in range(MAX_TURNS):
        resp = client.messages.create(
            model=MODEL, max_tokens=4096,
            system=SYSTEM, tools=TOOLS, messages=messages)`),
    highlight: [3, 5, 6],
    render: (t, c) => (
      <g>
        <Loop c={c} t={t} active={[0]} turn={0} stop="…" />
        <History c={c} t={t} count={1} fresh={1} />
      </g>
    )
  },
  {
    title: '모델: "list_files를 쓸게요"',
    text: rich('모델은 답 대신 **도구 요청(tool_use)** 을 돌려줘요. 어디에 파일이 있는지 모르니 목록부터 보겠다는 거죠. 응답은 그대로 이력에 붙이고, `stop_reason`이 `tool_use`면 루프를 계속 돌아요.'),
    code: dd(String.raw`        # 모델의 응답(도구 요청 포함)을 대화 이력에 그대로 추가
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text")`),
    highlight: [2, 4],
    render: (t, c) => (
      <g>
        <Loop c={c} t={t} active={[0]} turn={0} stop="tool_use" />
        <History c={c} t={t} count={2} fresh={1} />
      </g>
    )
  },
  {
    title: '도구 실행 → 결과를 이력에',
    text: rich('도구는 그냥 **파이썬 함수**예요. 이름으로 함수를 찾아 실행하고, 결과를 요청 번호(`tool_use_id`)와 짝지어 **user 메시지**로 돌려줘요. 그다음 turn에서 모델이 이 결과를 읽어요.'),
    code: dd(String.raw`        results = []
        for block in resp.content:
            if block.type == "tool_use":
                fn = TOOL_FUNCTIONS[block.name]
                print(f"  [{turn}] {block.name}({json.dumps(block.input, ensure_ascii=False)})")
                try:
                    output = fn(**block.input)
                    results.append({"type": "tool_result",
                                    "tool_use_id": block.id, "content": output})`),
    highlight: [4, 7, 8, 9],
    render: (t, c) => (
      <g>
        <Loop c={c} t={t} active={[1, 2]} turn={0} stop="tool_use" />
        <History c={c} t={t} count={3} fresh={1} />
      </g>
    )
  },
  {
    title: '실수해도 괜찮아요 — 오류도 결과로',
    text: rich('모델이 경로를 `jan.csv`로 잘못 적었어요. `read_file`은 멈추지 않고 **"파일이 없습니다"라는 결과**를 돌려줘요. 다음 turn에 모델이 그걸 읽고 `data/`를 붙여 스스로 고쳐요. 작업 폴더 밖은 아예 못 읽게 막는 **샌드박스**도 여기 있어요.'),
    code: dd(String.raw`def read_file(path: str) -> str:
    """샌드박스: 작업 디렉토리 바깥은 읽지 못하게 막는다."""
    p = Path(path).resolve()
    if not p.is_relative_to(Path.cwd().resolve()):
        return "오류: 작업 디렉토리 밖의 파일은 읽을 수 없습니다."
    if not p.exists():
        return f"오류: {path} 파일이 없습니다."`),
    highlight: [4, 5, 6, 7],
    render: (t, c) => (
      <g>
        <Loop c={c} t={t} active={[0, 1, 2]} turn={1} stop="tool_use" />
        <History c={c} t={t} count={5} fresh={2} />
      </g>
    )
  },
  {
    title: '도구 두 개를 한 번에 (병렬 요청)',
    text: rich('이번엔 모델이 한 응답에 `read_file`을 **두 개** 넣었어요. 둘 다 실행해서 결과를 **하나의 user 메시지**에 모아 돌려줘야 해요. 나눠 보내면 모델이 병렬 요청을 꺼리게 돼요.'),
    code: dd(String.raw`        # 요청된 도구를 전부 실행 — 결과는 '하나의' user 메시지에 모아 반환한다
        results = []
        for block in resp.content:
            if block.type == "tool_use":
                fn = TOOL_FUNCTIONS[block.name]
                print(f"  [{turn}] {block.name}({json.dumps(block.input, ensure_ascii=False)})")
                try:
                    output = fn(**block.input)
                    results.append({"type": "tool_result",
                                    "tool_use_id": block.id, "content": output})
                except Exception as e:               # 실패도 모델에게 알린다 — 스스로 복구하게
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": f"도구 실행 오류: {e}", "is_error": True})
        messages.append({"role": "user", "content": results})`),
    highlight: [1, 3, 14],
    render: (t, c) => (
      <g>
        <Loop c={c} t={t} active={[1, 2]} turn={2} stop="tool_use" />
        <History c={c} t={t} count={7} fresh={2} />
      </g>
    )
  },
  {
    title: '계산은 계산기 도구로 — 안전하게',
    text: rich('합계는 `calculator`에게 맡겨요. 파이썬 `eval`을 그대로 쓰면 아무 코드나 실행되는 구멍이 생기니, **허용된 연산(+, -, ×, ÷, 거듭제곱)만** 계산하는 화이트리스트로 만들었어요. 안전장치 세 번째 겹이에요.'),
    code: dd(String.raw`def calculator(expression: str) -> str:
    """안전한 산술 평가. eval을 그대로 쓰면 안 된다 — 코드 실행 구멍이 된다."""
    import ast, operator as op
    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
           ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg}`),
    highlight: [4, 5],
    render: (t, c) => (
      <g>
        <Loop c={c} t={t} active={[1, 2]} turn={3} stop="tool_use" />
        <History c={c} t={t} count={9} fresh={2} />
      </g>
    )
  },
  {
    title: 'end_turn — 루프 탈출, 최종 답변',
    text: rich('필요한 정보를 다 모은 모델이 이번엔 도구 없이 글로 답해요. `stop_reason`이 `end_turn`이 되면 **루프를 빠져나와 답을 돌려줘요**. 끝까지 못 끝내면 15번째에 멈추고 "최대 턴 수에 도달"을 알려요.'),
    code: dd(String.raw`        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text")`),
    highlight: [1, 2],
    render: (t, c) => (
      <g>
        <Loop c={c} t={t} active={[0]} turn={4} stop="end_turn" exit />
        <History c={c} t={t} count={10} fresh={1} />
      </g>
    )
  }
];

export default function Lab13AgentTrace(_: VizProps) {
  return <CodeStory steps={steps} viewBox={VB} stepSeconds={2} />;
}
