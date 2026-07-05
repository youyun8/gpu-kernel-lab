'use client';

import { useEffect, useMemo, useState } from 'react';
import { Pause, Play, RotateCcw, SkipBack, SkipForward } from 'lucide-react';

type Mode = 'race' | 'atomic' | 'mutex';
type EventKind = 'read' | 'compute' | 'write' | 'atomic' | 'lock' | 'unlock' | 'wait';

interface TimelineEvent {
  t: number; // time slot
  thread: 0 | 1;
  kind: EventKind;
  label: string;
  xAfter: number; // value of shared x after this slot
}

const kModeLabels: Record<Mode, string> = {
  race: '無同步 (data race)',
  atomic: 'atomicAdd',
  mutex: 'mutex / lock',
};

const kKindStyles: Record<EventKind, { bg: string; text: string }> = {
  read: { bg: '#58a6ff', text: '#000' },
  compute: { bg: 'hsl(var(--muted))', text: 'hsl(var(--foreground))' },
  write: { bg: '#ffa657', text: '#000' },
  atomic: { bg: '#a371f7', text: '#000' },
  lock: { bg: '#39d353', text: '#000' },
  unlock: { bg: '#39d353', text: '#000' },
  wait: { bg: 'transparent', text: 'hsl(var(--muted-foreground))' },
};

// Both threads run `x = x + 1` on shared x starting at 0; expected result 2.
const kScripts: Record<Mode, TimelineEvent[]> = {
  race: [
    { t: 0, thread: 0, kind: 'read', label: 'read x → r0 = 0', xAfter: 0 },
    { t: 1, thread: 1, kind: 'read', label: 'read x → r1 = 0', xAfter: 0 },
    { t: 2, thread: 0, kind: 'compute', label: 'r0 + 1 = 1', xAfter: 0 },
    { t: 3, thread: 1, kind: 'compute', label: 'r1 + 1 = 1', xAfter: 0 },
    { t: 4, thread: 0, kind: 'write', label: 'write x = 1', xAfter: 1 },
    { t: 5, thread: 1, kind: 'write', label: 'write x = 1', xAfter: 1 },
  ],
  atomic: [
    { t: 0, thread: 0, kind: 'atomic', label: 'atomicAdd(&x, 1) → 1', xAfter: 1 },
    { t: 1, thread: 1, kind: 'wait', label: '(硬體排隊)', xAfter: 1 },
    { t: 2, thread: 1, kind: 'atomic', label: 'atomicAdd(&x, 1) → 2', xAfter: 2 },
  ],
  mutex: [
    { t: 0, thread: 0, kind: 'lock', label: 'lock ✓', xAfter: 0 },
    { t: 1, thread: 1, kind: 'wait', label: 'lock… blocked', xAfter: 0 },
    { t: 2, thread: 0, kind: 'read', label: 'read x → 0', xAfter: 0 },
    { t: 3, thread: 0, kind: 'write', label: 'write x = 1', xAfter: 1 },
    { t: 4, thread: 0, kind: 'unlock', label: 'unlock', xAfter: 1 },
    { t: 5, thread: 1, kind: 'lock', label: 'lock ✓', xAfter: 1 },
    { t: 6, thread: 1, kind: 'read', label: 'read x → 1', xAfter: 1 },
    { t: 7, thread: 1, kind: 'write', label: 'write x = 2', xAfter: 2 },
    { t: 8, thread: 1, kind: 'unlock', label: 'unlock', xAfter: 2 },
  ],
};

const kPlayIntervalMs = 900;

/**
 * DataRaceTimeline steps through two threads doing `x = x + 1` on the same
 * address, showing how an unsynchronized interleaving loses an update and how
 * atomics or a lock restore correctness.
 */
export function DataRaceTimeline() {
  const [mode, setMode] = useState<Mode>('race');
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);

  const script = kScripts[mode];
  const num_slots = script[script.length - 1].t + 1;
  const last_step = num_slots;

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => {
      setStep((s) => {
        if (s >= last_step) {
          setPlaying(false);
          return s;
        }
        return s + 1;
      });
    }, kPlayIntervalMs);
    return () => window.clearInterval(timer);
  }, [playing, last_step]);

  const selectMode = (m: Mode) => {
    setMode(m);
    setStep(0);
    setPlaying(false);
  };

  // x value at each revealed slot, for the shared-variable trace row.
  const x_trace = useMemo(() => {
    const trace: number[] = [];
    let x = 0;
    for (let t = 0; t < num_slots; ++t) {
      for (const ev of script) {
        if (ev.t === t) x = ev.xAfter;
      }
      trace.push(x);
    }
    return trace;
  }, [script, num_slots]);

  const finished = step >= last_step;
  const final_x = x_trace[num_slots - 1];
  const correct = final_x === 2;

  const eventAt = (thread: 0 | 1, t: number) => script.find((ev) => ev.thread === thread && ev.t === t);

  return (
    <div className="my-6 rounded-lg border border-border bg-card/40 p-5">
      <p className="mb-1 text-base font-semibold text-foreground">Data Race Timeline</p>
      <p className="mb-4 text-sm text-muted-foreground">
        兩個 threads 同時執行 <code className="text-primary">x = x + 1</code> (x 初始為 0), 正確結果應為 2。
      </p>

      <div role="radiogroup" aria-label="同步方式" className="flex flex-wrap overflow-hidden rounded-md border border-border">
        {(Object.keys(kModeLabels) as Mode[]).map((m) => (
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
            {kModeLabels[m]}
          </button>
        ))}
      </div>

      <div className="mt-5 overflow-x-auto">
        <div className="min-w-[520px]">
          {/* Thread rows */}
          {([0, 1] as const).map((thread) => (
            <div key={thread} className="mb-2 flex items-center gap-2">
              <span className="w-16 shrink-0 font-mono text-xs text-muted-foreground">Thread {thread}</span>
              <div className="grid flex-1 gap-1" style={{ gridTemplateColumns: `repeat(${num_slots}, minmax(0, 1fr))` }}>
                {Array.from({ length: num_slots }, (_, t) => {
                  const ev = eventAt(thread, t);
                  const revealed = t < step;
                  if (!ev) return <div key={t} className="h-9 rounded border border-dashed border-border/40" />;
                  const style = kKindStyles[ev.kind];
                  return (
                    <div
                      key={t}
                      className={`flex h-9 items-center justify-center rounded border px-1 text-center text-[10px] leading-tight transition-opacity duration-300 ${
                        ev.kind === 'wait' ? 'border-dashed border-border' : 'border-transparent'
                      }`}
                      style={{
                        backgroundColor: revealed ? style.bg : 'hsl(var(--muted) / 0.3)',
                        color: revealed ? style.text : 'transparent',
                        opacity: revealed ? 1 : 0.6,
                      }}
                      title={revealed ? ev.label : '尚未執行'}
                    >
                      {revealed ? ev.label : ''}
                    </div>
                  );
                })}
              </div>
            </div>
          ))}

          {/* Shared x trace */}
          <div className="flex items-center gap-2 border-t border-border pt-2">
            <span className="w-16 shrink-0 font-mono text-xs text-primary">shared x</span>
            <div className="grid flex-1 gap-1" style={{ gridTemplateColumns: `repeat(${num_slots}, minmax(0, 1fr))` }}>
              {x_trace.map((x, t) => (
                <div
                  key={t}
                  className={`flex h-7 items-center justify-center rounded font-mono text-xs ${
                    t < step ? 'bg-primary/15 text-primary' : 'bg-background text-transparent'
                  }`}
                >
                  {t < step ? x : '·'}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Transport controls */}
      <div className="mt-4 flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={() => {
            setPlaying(false);
            setStep(0);
          }}
          aria-label="重設動畫"
          className="rounded-md border border-border p-2 text-muted-foreground transition hover:border-primary hover:text-foreground"
        >
          <RotateCcw className="h-4 w-4" aria-hidden />
        </button>
        <button
          type="button"
          onClick={() => {
            setPlaying(false);
            setStep((s) => Math.max(0, s - 1));
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
              setPlaying(false);
              return;
            }
            if (step >= last_step) setStep(0);
            setPlaying(true);
          }}
          aria-label={playing ? '暫停' : '播放'}
          className="rounded-md border border-primary bg-primary/10 p-2 text-primary transition hover:bg-primary/20"
        >
          {playing ? <Pause className="h-4 w-4" aria-hidden /> : <Play className="h-4 w-4" aria-hidden />}
        </button>
        <button
          type="button"
          onClick={() => {
            setPlaying(false);
            setStep((s) => Math.min(last_step, s + 1));
          }}
          aria-label="下一步"
          className="rounded-md border border-border p-2 text-muted-foreground transition hover:border-primary hover:text-foreground"
        >
          <SkipForward className="h-4 w-4" aria-hidden />
        </button>
        <p className="ml-2 text-sm" role="status">
          {finished ? (
            <span className={correct ? 'text-primary' : 'text-[#ff7b72]'}>
              {correct ? `✓ 最終 x = ${final_x}, 正確` : `✗ 最終 x = ${final_x}, 應為 2 — 一次 update 被覆蓋 (lost update)`}
            </span>
          ) : (
            <span className="text-muted-foreground">
              step <span className="font-mono text-foreground">{step}</span>/{last_step}
            </span>
          )}
        </p>
      </div>

      <p className="mt-3 text-xs text-muted-foreground">
        {mode === 'race' &&
          '兩個 threads 的 read-modify-write 交錯執行: 都讀到 0、都算出 1、後寫的把先寫的蓋掉。 這就是 data race 造成的 lost update — 結果取決於交錯順序, 每次執行可能不同。'}
        {mode === 'atomic' &&
          'atomicAdd 把 read-modify-write 合併成一個不可分割的操作, 硬體保證兩次加法依序生效。 GPU 上這是處理競爭的首選: 比 lock 便宜得多, 也不會有 deadlock。'}
        {mode === 'mutex' &&
          'lock 讓整段 critical section 互斥: Thread 1 必須等 Thread 0 unlock 才能進入, 讀到的是更新後的值。 CPU 上這是通用解; GPU 上數千個 threads 排隊拿鎖會序列化到不可用, 所以 GPU 幾乎總是改用 atomic 或重新設計演算法。'}
      </p>
    </div>
  );
}
