'use client';

import { useEffect, useState } from 'react';
import { Pause, Play, RotateCcw, SkipBack, SkipForward } from 'lucide-react';

const kPlayIntervalMs = 1400;

/** Shared transport controls (reset / step / play-pause) used by all three diagrams below. */
function PlaybackControls({
  step,
  lastStep,
  playing,
  onStep,
  onPlaying,
  note,
}: {
  step: number;
  lastStep: number;
  playing: boolean;
  onStep: (step: number) => void;
  onPlaying: (playing: boolean) => void;
  note: string;
}) {
  return (
    <div className="mt-4 flex flex-wrap items-center gap-2">
      <button
        type="button"
        onClick={() => {
          onPlaying(false);
          onStep(0);
        }}
        aria-label="重設動畫"
        className="rounded-md border border-border p-2 text-muted-foreground transition hover:border-primary hover:text-foreground"
      >
        <RotateCcw className="h-4 w-4" aria-hidden />
      </button>
      <button
        type="button"
        onClick={() => {
          onPlaying(false);
          onStep(Math.max(0, step - 1));
        }}
        aria-label="上一步"
        className="rounded-md border border-border p-2 text-muted-foreground transition hover:border-primary hover:text-foreground"
      >
        <SkipBack className="h-4 w-4" aria-hidden />
      </button>
      <button
        type="button"
        onClick={() => {
          if (playing) {
            onPlaying(false);
            return;
          }
          if (step >= lastStep) onStep(0);
          onPlaying(true);
        }}
        aria-label={playing ? '暫停' : '播放'}
        className="rounded-md border border-primary bg-primary/10 p-2 text-primary transition hover:bg-primary/20"
      >
        {playing ? <Pause className="h-4 w-4" aria-hidden /> : <Play className="h-4 w-4" aria-hidden />}
      </button>
      <button
        type="button"
        onClick={() => {
          onPlaying(false);
          onStep(Math.min(lastStep, step + 1));
        }}
        aria-label="下一步"
        className="rounded-md border border-border p-2 text-muted-foreground transition hover:border-primary hover:text-foreground"
      >
        <SkipForward className="h-4 w-4" aria-hidden />
      </button>
      <p className="ml-2 text-sm text-muted-foreground" role="status">
        <span className="font-mono text-foreground">
          {step}/{lastStep}
        </span>{' '}
        — {note}
      </p>
    </div>
  );
}

/** Advances `step` by one every `kPlayIntervalMs` while `playing`, stopping at `lastStep`. */
function usePlayback(lastStep: number) {
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => {
      setStep((s) => {
        if (s >= lastStep) {
          setPlaying(false);
          return s;
        }
        return s + 1;
      });
    }, kPlayIntervalMs);
    return () => window.clearInterval(timer);
  }, [playing, lastStep]);

  return { step, setStep, playing, setPlaying };
}

const kStatusStyle: Record<'idle' | 'active' | 'blocked', string> = {
  idle: 'border-border bg-background text-muted-foreground',
  active: 'border-primary bg-primary/10 text-primary',
  blocked: 'border-[#ff7b72] bg-[#ff7b72]/10 text-[#ff7b72]',
};

const kStatusLabel: Record<'idle' | 'active' | 'blocked', string> = {
  idle: 'idle',
  active: '執行中',
  blocked: 'blocked',
};

interface PcStep {
  queue: number[];
  producer: 'idle' | 'active' | 'blocked';
  consumer: 'idle' | 'active' | 'blocked';
  note: string;
}

const kCapacity = 3;

// Deterministic walk through a capacity-3 bounded buffer that hits both
// blocking conditions: producer blocks when full, consumer blocks when empty.
const kPcSteps: PcStep[] = [
  { queue: [], producer: 'idle', consumer: 'idle', note: '初始狀態: buffer 空, empty_slots=3, full_slots=0' },
  { queue: [1], producer: 'active', consumer: 'idle', note: 'Producer: sem_wait(empty) 3→2, 寫入, sem_post(full) 0→1' },
  { queue: [1, 2], producer: 'active', consumer: 'idle', note: 'Producer 繼續生產: empty 2→1, full 1→2' },
  { queue: [1, 2, 3], producer: 'active', consumer: 'idle', note: 'Producer 再生產一項: empty 1→0, full 2→3 — buffer 滿了' },
  { queue: [1, 2, 3], producer: 'blocked', consumer: 'idle', note: 'empty_slots=0: Producer 的 sem_wait(empty) 被擋住, 必須等 Consumer 先消費' },
  { queue: [2, 3], producer: 'blocked', consumer: 'active', note: 'Consumer: sem_wait(full) 3→2, 讀走 1, sem_post(empty) 0→1 — Producer 被喚醒' },
  { queue: [2, 3, 4], producer: 'active', consumer: 'idle', note: 'Producer 解除阻塞, 生產下一項: empty 1→0, full 2→3' },
  { queue: [3, 4], producer: 'idle', consumer: 'active', note: 'Consumer 讀走 2: empty 0→1, full 3→2' },
  { queue: [4], producer: 'idle', consumer: 'active', note: 'Consumer 讀走 3: empty 1→2, full 2→1' },
  { queue: [], producer: 'idle', consumer: 'active', note: 'Consumer 讀走 4: empty 2→3, full 1→0 — buffer 空了' },
  { queue: [], producer: 'idle', consumer: 'blocked', note: 'full_slots=0: Consumer 的 sem_wait(full) 被擋住, 必須等 Producer 生產' },
  { queue: [5], producer: 'active', consumer: 'blocked', note: 'Producer 生產下一項: empty 3→2, full 0→1 — Consumer 被喚醒' },
  { queue: [], producer: 'idle', consumer: 'active', note: 'Consumer 讀走 5: empty 2→3, full 1→0' },
];

/**
 * ProducerConsumerDiagram walks through a bounded-buffer producer/consumer
 * pipeline (the classic semaphore pattern), showing both blocking edges:
 * a full buffer stalls the producer, an empty buffer stalls the consumer.
 */
export function ProducerConsumerDiagram() {
  const lastStep = kPcSteps.length - 1;
  const { step, setStep, playing, setPlaying } = usePlayback(lastStep);
  const current = kPcSteps[step];
  const emptySlots = kCapacity - current.queue.length;
  const fullSlots = current.queue.length;

  return (
    <div className="my-6 rounded-lg border border-border bg-card/40 p-5">
      <p className="mb-1 text-base font-semibold text-foreground">Producer / Consumer (Bounded Buffer)</p>
      <p className="mb-4 text-sm text-muted-foreground">
        一個容量 {kCapacity} 的環狀 buffer, 由 <code className="text-primary">empty_slots</code> /{' '}
        <code className="text-primary">full_slots</code> 兩個 semaphore 控制何時該讓 producer 或 consumer 停下來等待。
      </p>

      <div className="flex flex-col items-center gap-3 sm:flex-row sm:justify-center">
        <div
          className={`flex w-28 shrink-0 flex-col items-center rounded-lg border-2 p-3 transition-colors ${kStatusStyle[current.producer]}`}
          role="img"
          aria-label={`Producer: ${kStatusLabel[current.producer]}`}
        >
          <span className="text-sm font-semibold">Producer</span>
          <span className="mt-1 text-xs">{kStatusLabel[current.producer]}</span>
        </div>

        <span className="text-2xl text-muted-foreground sm:rotate-0" aria-hidden>
          →
        </span>

        <div className="flex shrink-0 gap-1" role="img" aria-label={`buffer 中有 ${fullSlots}/${kCapacity} 筆資料`}>
          {Array.from({ length: kCapacity }, (_, i) => {
            const value = current.queue[i];
            const filled = value !== undefined;
            return (
              <div
                key={i}
                className={`flex h-12 w-12 items-center justify-center rounded border font-mono text-sm transition-colors ${
                  filled ? 'border-[#ffa657] bg-[#ffa657]/15 text-[#ffa657]' : 'border-dashed border-border/60 text-transparent'
                }`}
              >
                {filled ? value : '·'}
              </div>
            );
          })}
        </div>

        <span className="text-2xl text-muted-foreground" aria-hidden>
          →
        </span>

        <div
          className={`flex w-28 shrink-0 flex-col items-center rounded-lg border-2 p-3 transition-colors ${kStatusStyle[current.consumer]}`}
          role="img"
          aria-label={`Consumer: ${kStatusLabel[current.consumer]}`}
        >
          <span className="text-sm font-semibold">Consumer</span>
          <span className="mt-1 text-xs">{kStatusLabel[current.consumer]}</span>
        </div>
      </div>

      <dl className="mt-4 grid grid-cols-2 gap-3 text-sm">
        <div className="rounded-md bg-background p-3 text-center">
          <dt className="text-xs text-muted-foreground">empty_slots</dt>
          <dd className="font-mono text-lg text-foreground">{emptySlots}</dd>
        </div>
        <div className="rounded-md bg-background p-3 text-center">
          <dt className="text-xs text-muted-foreground">full_slots</dt>
          <dd className="font-mono text-lg text-foreground">{fullSlots}</dd>
        </div>
      </dl>

      <PlaybackControls step={step} lastStep={lastStep} playing={playing} onStep={setStep} onPlaying={setPlaying} note={current.note} />
    </div>
  );
}

type RwKind = 'reading' | 'writing' | 'waiting' | 'idle';

const kRwKindStyle: Record<RwKind, { bg: string; text: string; dashed?: boolean }> = {
  reading: { bg: '#58a6ff', text: '#000' },
  writing: { bg: '#ffa657', text: '#000' },
  waiting: { bg: 'transparent', text: 'hsl(var(--muted-foreground))', dashed: true },
  idle: { bg: 'hsl(var(--muted) / 0.4)', text: 'hsl(var(--muted-foreground))' },
};

// Occupancy of each thread at every time slot t=0..7. Writer-preference: a
// reader that arrives while a writer is already waiting queues up behind it,
// which is why R3 waits instead of joining R1/R2 at t3.
const kRwThreads: { label: string; kinds: RwKind[] }[] = [
  { label: 'Reader A', kinds: ['reading', 'reading', 'reading', 'reading', 'idle', 'idle', 'idle', 'idle'] },
  { label: 'Reader B', kinds: ['idle', 'reading', 'reading', 'reading', 'reading', 'idle', 'idle', 'idle'] },
  { label: 'Writer', kinds: ['idle', 'idle', 'waiting', 'waiting', 'waiting', 'writing', 'idle', 'idle'] },
  { label: 'Reader C', kinds: ['idle', 'idle', 'idle', 'waiting', 'waiting', 'waiting', 'reading', 'reading'] },
];

const kRwNotes = [
  'Reader A 開始讀取: 共享資料允許多個 reader 同時讀。',
  'Reader B 也開始讀取: 兩個 reader 並存, 互不阻擋。',
  'Writer 想寫入, 但仍有 reader 在讀 → writer 必須等待 (exclusive access)。',
  'Reader C 這時才想讀 — 但 writer 已經在排隊了; 為了避免 writer 餓死, C 排在 writer 後面而不是直接插隊。',
  'Reader A 讀完離開, Reader B 仍在讀, Writer/C 繼續等待。',
  'Reader B 也讀完了, 所有 reader 都離開 → Writer 取得獨佔存取, 開始寫入。',
  'Writer 寫完釋放 lock → 排在後面的 Reader C 終於能開始讀取。',
  'Reader C 讀完, 系統回到 idle。',
];

/**
 * ReadersWritersDiagram shows the readers-writers problem: any number of
 * readers may hold the resource concurrently, a writer needs it exclusively,
 * and a writer-preference queueing rule keeps writers from starving.
 */
export function ReadersWritersDiagram() {
  const numSlots = kRwThreads[0].kinds.length;
  const lastStep = numSlots;
  const { step, setStep, playing, setPlaying } = usePlayback(lastStep);
  const note = kRwNotes[Math.min(step, numSlots - 1)];

  const activeReaders = (t: number) => kRwThreads.filter((th, i) => i !== 2 && th.kinds[t] === 'reading').length;
  const writerHolds = (t: number) => kRwThreads[2].kinds[t] === 'writing';

  return (
    <div className="my-6 rounded-lg border border-border bg-card/40 p-5">
      <p className="mb-1 text-base font-semibold text-foreground">Readers / Writers</p>
      <p className="mb-4 text-sm text-muted-foreground">
        多個 reader 可以同時持有共享資料, 但 writer 需要獨佔; 這裡採用 <strong>writer-preference</strong>:
        新來的 reader 若發現已有 writer 在排隊, 就排在它後面, 避免 writer 被無限插隊的 reader 餓死。
      </p>

      <div className="mt-5 overflow-x-auto">
        <div className="min-w-[560px]">
          {kRwThreads.map((th) => (
            <div key={th.label} className="mb-2 flex items-center gap-2">
              <span className="w-20 shrink-0 font-mono text-xs text-muted-foreground">{th.label}</span>
              <div className="grid flex-1 gap-1" style={{ gridTemplateColumns: `repeat(${numSlots}, minmax(0, 1fr))` }}>
                {th.kinds.map((kind, t) => {
                  const revealed = t < step;
                  const style = kRwKindStyle[kind];
                  return (
                    <div
                      key={t}
                      className={`flex h-8 items-center justify-center rounded border text-[10px] transition-opacity duration-300 ${
                        style.dashed ? 'border-dashed border-border' : 'border-transparent'
                      }`}
                      style={{
                        backgroundColor: revealed ? style.bg : 'hsl(var(--muted) / 0.3)',
                        color: revealed ? style.text : 'transparent',
                        opacity: revealed ? 1 : 0.6,
                      }}
                      title={revealed ? kind : '尚未執行'}
                    >
                      {revealed && kind !== 'idle' ? kind : ''}
                    </div>
                  );
                })}
              </div>
            </div>
          ))}

          <div className="flex items-center gap-2 border-t border-border pt-2">
            <span className="w-20 shrink-0 font-mono text-xs text-primary">active readers</span>
            <div className="grid flex-1 gap-1" style={{ gridTemplateColumns: `repeat(${numSlots}, minmax(0, 1fr))` }}>
              {Array.from({ length: numSlots }, (_, t) => (
                <div
                  key={t}
                  className={`flex h-6 items-center justify-center rounded font-mono text-xs ${
                    t < step ? (writerHolds(t) ? 'bg-[#ff7b72]/20 text-[#ff7b72]' : 'bg-primary/15 text-primary') : 'bg-background text-transparent'
                  }`}
                >
                  {t < step ? (writerHolds(t) ? 'W' : activeReaders(t)) : '·'}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <PlaybackControls step={step} lastStep={lastStep} playing={playing} onStep={setStep} onPlaying={setPlaying} note={note} />
    </div>
  );
}

type PhilState = 'thinking' | 'hungry' | 'eating' | 'blocked';

interface DpStep {
  forks: (number | null)[];
  phil: PhilState[];
  note: string;
}

type DpMode = 'naive' | 'fixed';

const kNumPhilosophers = 5;
const kPhilColors = ['#58a6ff', '#a371f7', '#ffa657', '#39d353', '#ff7b72'];

const kPhilStateRing: Record<PhilState, string> = {
  thinking: 'hsl(var(--border))',
  hungry: '#ffa657',
  eating: '#39d353',
  blocked: '#ff7b72',
};

// fork[i] sits between philosopher i and philosopher (i+1)%5.
// Philosopher i's "left" fork is fork[(i-1+5)%5]; "right" fork is fork[i].
const kDpScripts: Record<DpMode, DpStep[]> = {
  naive: [
    {
      forks: [null, null, null, null, null],
      phil: ['thinking', 'thinking', 'thinking', 'thinking', 'thinking'],
      note: '五位哲學家都在思考, 五支叉子都放在桌上。',
    },
    {
      forks: [1, 2, 3, 4, 0],
      phil: ['blocked', 'blocked', 'blocked', 'blocked', 'blocked'],
      note:
        '每個人都先抓「左手」那支叉子, 接著想抓右手叉子 — 但右手那支正好被順時鐘鄰居拿走了。 每個人都握著一支、等著另一支, 形成環狀等待 (circular wait): 沒有人能吃, 永遠卡住。',
    },
  ],
  fixed: [
    {
      forks: [null, null, null, null, null],
      phil: ['thinking', 'thinking', 'thinking', 'thinking', 'thinking'],
      note: '五位哲學家都在思考, 五支叉子都放在桌上。',
    },
    {
      forks: [1, 2, 3, null, 0],
      phil: ['blocked', 'blocked', 'blocked', 'hungry', 'blocked'],
      note:
        'P4 刻意打破對稱, 改成「先抓右手」的 fork4 — 這次它和 P0 搶 fork4, 但輸了。 於是 P4 的左手 fork3 從頭到尾沒被拿走, 缺口就留在這裡。',
    },
    {
      forks: [1, 2, 3, 3, 0],
      phil: ['blocked', 'blocked', 'blocked', 'eating', 'blocked'],
      note: 'P3 兩支叉子都到手, 開始吃 — 因為缺口 fork3 沒被搶走, 保證至少有一位哲學家能吃到, 迴圈永遠無法完整閉合。',
    },
    {
      forks: [1, 2, null, null, 0],
      phil: ['blocked', 'blocked', 'hungry', 'thinking', 'blocked'],
      note: 'P3 吃完放下兩支叉子: 空出的 fork2 讓 P2 接下來能拿到, 骨牌效應會依序釋放下去, 不會有人永遠等待。',
    },
  ],
};

function polarPoint(cx: number, cy: number, r: number, angleDeg: number) {
  const rad = (angleDeg * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}

/**
 * DiningPhilosophersDiagram renders the classic five-philosophers table as an
 * SVG figure and steps through the naive "everyone grabs left first" scheme
 * (which deadlocks) versus the asymmetric fix (one philosopher grabs right
 * first, which breaks the circular wait).
 */
export function DiningPhilosophersDiagram() {
  const [mode, setMode] = useState<DpMode>('naive');
  const script = kDpScripts[mode];
  const lastStep = script.length - 1;
  const { step, setStep, playing, setPlaying } = usePlayback(lastStep);
  const current = script[Math.min(step, lastStep)];

  const selectMode = (m: DpMode) => {
    setMode(m);
    setStep(0);
    setPlaying(false);
  };

  const cx = 130;
  const cy = 120;
  const philR = 90;
  const forkR = 58;

  const philPoints = Array.from({ length: kNumPhilosophers }, (_, i) => polarPoint(cx, cy, philR, -90 + i * (360 / kNumPhilosophers)));
  const forkPoints = Array.from({ length: kNumPhilosophers }, (_, i) =>
    polarPoint(cx, cy, forkR, -90 + i * (360 / kNumPhilosophers) + 360 / kNumPhilosophers / 2),
  );

  return (
    <div className="my-6 rounded-lg border border-border bg-card/40 p-5">
      <p className="mb-1 text-base font-semibold text-foreground">Dining Philosophers</p>
      <p className="mb-4 text-sm text-muted-foreground">
        五位哲學家共用五支叉子, 吃飯前要同時拿到左右兩支。 若大家都用同樣的順序 (先左後右) 去拿, 就可能全部卡死;
        只要打破其中一人的順序, 就能保證不會 deadlock。
      </p>

      <div role="radiogroup" aria-label="dining philosophers 策略" className="flex overflow-hidden rounded-md border border-border">
        {(
          [
            ['naive', '全部先左後右 (會 deadlock)'],
            ['fixed', '一人反向拿 (不會 deadlock)'],
          ] as [DpMode, string][]
        ).map(([m, label]) => (
          <button
            key={m}
            type="button"
            role="radio"
            aria-checked={mode === m}
            onClick={() => selectMode(m)}
            className={`px-3 py-1.5 text-sm font-medium transition ${
              mode === m ? 'bg-primary/15 text-primary' : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="mt-4 flex justify-center">
        <svg viewBox="0 0 260 240" className="h-64 w-64" role="img" aria-label="五位哲學家與五支叉子的座位圖">
          {/* fork possession lines */}
          {forkPoints.map((f, i) => {
            const owner = current.forks[i];
            if (owner === null) return null;
            const p = philPoints[owner];
            return <line key={`edge-${i}`} x1={f.x} y1={f.y} x2={p.x} y2={p.y} stroke={kPhilColors[owner]} strokeWidth={2} opacity={0.6} />;
          })}

          {/* forks */}
          {forkPoints.map((f, i) => {
            const owner = current.forks[i];
            const held = owner !== null;
            return (
              <g key={`fork-${i}`}>
                <circle cx={f.x} cy={f.y} r={7} fill={held ? kPhilColors[owner as number] : 'hsl(var(--muted))'} stroke="hsl(var(--card))" strokeWidth={1.5} />
                <title>{held ? `fork${i}: 被 P${owner} 拿著` : `fork${i}: 空閒`}</title>
              </g>
            );
          })}

          {/* philosophers */}
          {philPoints.map((p, i) => (
            <g key={`phil-${i}`}>
              <circle
                cx={p.x}
                cy={p.y}
                r={20}
                fill={kPhilColors[i]}
                fillOpacity={current.phil[i] === 'thinking' ? 0.25 : 0.85}
                stroke={kPhilStateRing[current.phil[i]]}
                strokeWidth={3}
                strokeDasharray={current.phil[i] === 'blocked' ? '3 3' : undefined}
              />
              <text x={p.x} y={p.y + 4} textAnchor="middle" fontSize={11} fontFamily="monospace" fill="hsl(var(--card-foreground))">
                P{i}
              </text>
              <title>
                P{i}: {current.phil[i]}
              </title>
            </g>
          ))}
        </svg>
      </div>

      <div className="flex flex-wrap justify-center gap-x-4 gap-y-1 text-xs text-muted-foreground" aria-hidden>
        {(['thinking', 'hungry', 'eating', 'blocked'] as PhilState[]).map((s) => (
          <span key={s} className="flex items-center gap-1">
            <span className="inline-block h-2 w-2 rounded-full" style={{ backgroundColor: kPhilStateRing[s] }} />
            {s}
          </span>
        ))}
      </div>

      <PlaybackControls step={step} lastStep={lastStep} playing={playing} onStep={setStep} onPlaying={setPlaying} note={current.note} />
    </div>
  );
}
