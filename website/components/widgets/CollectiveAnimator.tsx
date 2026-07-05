'use client';

import { useEffect, useMemo, useState } from 'react';
import { Pause, Play, RotateCcw, SkipBack, SkipForward } from 'lucide-react';

type Primitive = 'all-reduce' | 'reduce-scatter' | 'all-gather';

const kPrimitiveLabels: Record<Primitive, string> = {
  'all-reduce': 'All-Reduce',
  'reduce-scatter': 'Reduce-Scatter',
  'all-gather': 'All-Gather',
};

const kGpuOptions = [2, 4, 8];
const kPlayIntervalMs = 1200;

// contributions[gpu][chunk] = how many ranks' data is merged into that chunk
// (0 = the GPU does not hold this chunk at all).
type State = number[][];

interface Step {
  state: State;
  description: string;
}

function simulate(primitive: Primitive, n: number): Step[] {
  const steps: Step[] = [];
  let state: State;

  if (primitive === 'all-gather') {
    // Each GPU starts with only its own chunk, already final.
    state = Array.from({ length: n }, (_, g) => Array.from({ length: n }, (_, c) => (c === g ? n : 0)));
  } else {
    // Each GPU has its own partial value for every chunk.
    state = Array.from({ length: n }, () => Array.from({ length: n }, () => 1));
  }
  steps.push({ state, description: '初始狀態' });

  const clone = (s: State) => s.map((row) => [...row]);

  if (primitive !== 'all-gather') {
    // Reduce-scatter phase: at step s, GPU g sends chunk (g-s) mod n to g+1,
    // which adds it into its own partial copy.
    for (let s = 0; s < n - 1; ++s) {
      const next = clone(state);
      for (let g = 0; g < n; ++g) {
        const c = (g - s + n * 2) % n;
        const recv = (g + 1) % n;
        next[recv][c] = state[recv][c] + state[g][c];
        // Sender hands the chunk off; it no longer keeps the partial sum.
        if (next[g][c] === state[g][c]) next[g][c] = 0;
      }
      state = next;
      steps.push({
        state,
        description: `Reduce-scatter step ${s + 1}/${n - 1}: 每個 GPU g 把 chunk (g−${s}) mod ${n} 傳給 g+1 並累加`,
      });
    }
  }

  if (primitive !== 'reduce-scatter') {
    // All-gather phase: at step s, GPU g sends its complete chunk
    // (g+1-s) mod n to g+1, which just copies it.
    for (let s = 0; s < n - 1; ++s) {
      const next = clone(state);
      for (let g = 0; g < n; ++g) {
        const c = primitive === 'all-gather' ? (g - s + n * 2) % n : (g + 1 - s + n * 2) % n;
        const recv = (g + 1) % n;
        if (state[g][c] === n) next[recv][c] = n;
      }
      state = next;
      steps.push({
        state,
        description: `All-gather step ${s + 1}/${n - 1}: 每個 GPU 把手上已完成的 chunk 傳給下一個 GPU (純複製)`,
      });
    }
  }

  return steps;
}

/**
 * CollectiveAnimator steps through ring all-reduce / reduce-scatter /
 * all-gather on 2-8 GPUs, showing per-chunk state and per-step transfer
 * volume, and how the busbw factor 2(n-1)/n falls out of the step count.
 */
export function CollectiveAnimator() {
  const [primitive, setPrimitive] = useState<Primitive>('all-reduce');
  const [num_gpus, setNumGpus] = useState(4);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);

  const steps = useMemo(() => simulate(primitive, num_gpus), [primitive, num_gpus]);
  const last_step = steps.length - 1;
  const current = steps[Math.min(step, last_step)];

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

  const reset = (p: Primitive, n: number) => {
    setPrimitive(p);
    setNumGpus(n);
    setStep(0);
    setPlaying(false);
  };

  const n = num_gpus;
  const total_steps = primitive === 'all-reduce' ? 2 * (n - 1) : n - 1;
  // Per GPU, every step moves 1/n of the message size S in a ring.
  const busbw_factor = primitive === 'all-reduce' ? `2(n−1)/n = ${((2 * (n - 1)) / n).toFixed(2)}` : `(n−1)/n = ${((n - 1) / n).toFixed(2)}`;

  return (
    <div className="my-6 rounded-lg border border-border bg-card/40 p-5">
      <p className="mb-4 text-base font-semibold text-foreground">Ring Collective Animator</p>

      <div className="flex flex-wrap items-center gap-4">
        <div role="radiogroup" aria-label="collective primitive" className="flex overflow-hidden rounded-md border border-border">
          {(Object.keys(kPrimitiveLabels) as Primitive[]).map((p) => (
            <button
              key={p}
              type="button"
              role="radio"
              aria-checked={primitive === p}
              onClick={() => reset(p, num_gpus)}
              className={`px-3 py-1.5 text-sm font-medium transition ${
                primitive === p ? 'bg-primary/15 text-primary' : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              {kPrimitiveLabels[p]}
            </button>
          ))}
        </div>
        <label className="flex items-center gap-2 text-sm text-muted-foreground">
          GPUs:
          <select
            value={num_gpus}
            onChange={(e) => reset(primitive, Number(e.target.value))}
            aria-label="GPU 數量"
            className="rounded-md border border-border bg-background px-2 py-1 text-foreground"
          >
            {kGpuOptions.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>
      </div>

      {/* GPU cards in ring order */}
      <div className="mt-5 grid gap-2" style={{ gridTemplateColumns: `repeat(${n}, minmax(0, 1fr))` }}>
        {current.state.map((chunks, g) => (
          <div key={g} className="rounded-md bg-background p-2">
            <p className="mb-1 text-center font-mono text-xs text-muted-foreground">GPU {g}</p>
            <div className="space-y-1" role="img" aria-label={`GPU ${g} 的 ${n} 個 chunk 狀態`}>
              {chunks.map((count, c) => {
                const fraction = count / n;
                const complete = count === n;
                return (
                  <div
                    key={c}
                    className={`flex h-6 items-center justify-center rounded text-[10px] font-medium ${
                      complete ? 'text-primary-foreground' : 'text-foreground'
                    }`}
                    style={{
                      backgroundColor: complete
                        ? 'hsl(var(--primary))'
                        : count > 0
                          ? `hsl(var(--primary) / ${0.15 + fraction * 0.4})`
                          : 'hsl(var(--muted) / 0.6)',
                    }}
                    title={`chunk ${c}: ${count === 0 ? '不在此 GPU' : complete ? '完整結果' : `${count}/${n} 份 partial 已累加`}`}
                  >
                    {count === 0 ? '—' : complete ? `Σ c${c}` : `c${c}:${count}/${n}`}
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
      <p className="mt-1 text-center text-[10px] text-muted-foreground" aria-hidden>
        ring: GPU 0 → 1 → … → {n - 1} → 0, 每步每個 GPU 同時送出並接收 S/{n}
      </p>

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
        <p className="ml-2 text-sm text-muted-foreground" role="status">
          <span className="font-mono text-foreground">
            {step}/{last_step}
          </span>{' '}
          — {current.description}
        </p>
      </div>

      <dl className="mt-4 grid grid-cols-2 gap-3 text-sm sm:grid-cols-3">
        <div className="rounded-md bg-background p-3">
          <dt className="text-xs text-muted-foreground">總步數</dt>
          <dd className="font-mono text-lg text-foreground">{total_steps}</dd>
        </div>
        <div className="rounded-md bg-background p-3">
          <dt className="text-xs text-muted-foreground">每步每 GPU 傳輸量</dt>
          <dd className="font-mono text-lg text-foreground">S/{n}</dd>
        </div>
        <div className="rounded-md bg-background p-3">
          <dt className="text-xs text-muted-foreground">busbw / algbw</dt>
          <dd className="font-mono text-lg text-foreground">{busbw_factor}</dd>
        </div>
      </dl>

      <p className="mt-3 text-xs text-muted-foreground">
        Ring {kPrimitiveLabels[primitive]} 共 {total_steps} 步, 每步每個 GPU 同時送出並收到 S/{n} 的資料 (S = message size),
        所以每個 GPU 實際搬運 {primitive === 'all-reduce' ? `2(n−1)·S/n` : `(n−1)·S/n`} 的資料量 — 這正是 bus bandwidth 校正係數{' '}
        {busbw_factor} 的來源。 GPU 數量越多, 係數越接近 {primitive === 'all-reduce' ? 2 : 1}, ring 演算法的頻寬利用率也越接近最優。
      </p>
    </div>
  );
}
