import type { ReactNode } from 'react';

// Static, self-contained figures for the Software Pipelining track (Track 11).
// Same visual language as GemmFigures: presentational only (no hooks / client
// JS), theme-aware via Tailwind tokens plus a few fixed semantic colors.

const kStage = '#58a6ff'; // a generic active stage / forward pass
const kBack = '#ffa657'; // backward pass
const kCompute = '#39d353'; // compute
const kD2H = '#a371f7'; // device->host copy
const kAccent = '#39c5cf'; // "the point"
const kIdleBg = 'rgba(110,118,129,0.18)'; // bubble / idle

function Fig({ title, caption, children, scroll }: { title: string; caption: ReactNode; children: ReactNode; scroll?: boolean }) {
  return (
    <figure className="my-6 rounded-lg border border-border bg-card/40 p-5">
      <p className="mb-4 text-base font-semibold text-foreground">{title}</p>
      <div className={scroll ? 'overflow-x-auto' : undefined}>{children}</div>
      <figcaption className="mt-4 text-xs leading-5 text-muted-foreground">{caption}</figcaption>
    </figure>
  );
}

function Chip({ color, dashed, children }: { color: string; dashed?: boolean; children: ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
      <span
        aria-hidden
        className="h-3 w-3 rounded-sm"
        style={dashed ? { border: `1px dashed ${color}` } : { backgroundColor: color }}
      />
      {children}
    </span>
  );
}

type Cell = { t: 'idle' } | { t: 'op'; label: string; color: string };

/** A row-per-lane gantt. `rows` is [label, cells[]]. Fixed-width cells. */
function Gantt({ rows, cell = 26, rowLabelW = 52 }: { rows: { label: string; cells: Cell[] }[]; cell?: number; rowLabelW?: number }) {
  const cols = Math.max(...rows.map((r) => r.cells.length));
  return (
    <div className="inline-block min-w-full" style={{ display: 'grid', gridTemplateColumns: `${rowLabelW}px repeat(${cols}, ${cell}px)`, gap: 2 }}>
      {rows.map((row) => (
        <div key={row.label} style={{ display: 'contents' }}>
          <div className="flex items-center pr-2 font-mono text-[11px] text-muted-foreground" style={{ height: cell }}>
            {row.label}
          </div>
          {Array.from({ length: cols }, (_, c) => {
            const cellData = row.cells[c];
            if (!cellData || cellData.t === 'idle') {
              return <div key={c} className="rounded-[2px]" style={{ height: cell, backgroundColor: kIdleBg }} />;
            }
            return (
              <div
                key={c}
                className="flex items-center justify-center rounded-[2px] text-[10px] font-semibold"
                style={{ height: cell, backgroundColor: cellData.color, color: 'rgba(0,0,0,0.78)' }}
              >
                {cellData.label}
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* 1. Generic software pipeline: fill / steady-state / drain           */
/* ------------------------------------------------------------------ */

export function PipelineConceptFigure() {
  // 3 stages (A/B/C), 5 items. Item j is in stage s at time j+s.
  const stages = ['A', 'B', 'C'];
  const items = 5;
  const colors = ['#58a6ff', '#39c5cf', '#39d353'];
  const total = items + stages.length - 1;
  const rows = stages.map((name, s) => ({
    label: `stage ${name}`,
    cells: Array.from({ length: total }, (_, t): Cell => {
      const j = t - s;
      return j >= 0 && j < items ? { t: 'op', label: `#${j}`, color: colors[s] } : { t: 'idle' };
    }),
  }));
  return (
    <Fig
      title="Software pipeline: fill、steady-state、drain"
      scroll
      caption={
        <>
          三個 stage (A→B→C)、五個工作項。 開頭的 <strong>fill</strong> (前 2 步) 與結尾的 <strong>drain</strong>
          (後 2 步) 只有部分 stage 在忙 — 那是 pipeline 的 <span style={{ color: kAccent }}>bubble</span>。 中間
          <strong> steady state</strong> 三個 stage 同時滿載, 這才是加速的來源。 處理 N 項、深度 k 的 pipeline 需要
          <span className="font-mono"> N + k − 1</span> 步而非 <span className="font-mono">N · k</span> 步;
          當 N ≫ k, 吞吐趨近「每步完成一項」, 加速 ≈ k 倍。
        </>
      }
    >
      <Gantt rows={rows} />
      <div className="mt-3 flex flex-wrap gap-4">
        <Chip color={colors[0]}>stage A</Chip>
        <Chip color={colors[1]}>stage B</Chip>
        <Chip color={colors[2]}>stage C</Chip>
        <Chip color={kIdleBg}>bubble (idle)</Chip>
      </div>
    </Fig>
  );
}

/* ------------------------------------------------------------------ */
/* 2. Host-device stream overlap: H2D / compute / D2H                  */
/* ------------------------------------------------------------------ */

export function StreamOverlapFigure() {
  const chunks = 4;
  const serialRows = [
    {
      label: 'GPU',
      cells: Array.from({ length: chunks * 3 }, (_, t): Cell => {
        const phase = t % 3;
        const j = Math.floor(t / 3);
        const c = phase === 0 ? kStage : phase === 1 ? kCompute : kD2H;
        const lbl = phase === 0 ? `H2D${j}` : phase === 1 ? `C${j}` : `D2H${j}`;
        return { t: 'op', label: lbl, color: c };
      }),
    },
  ];
  const mk = (color: string, tag: string, offset: number): { label: string; cells: Cell[] } => ({
    label: tag,
    cells: Array.from({ length: chunks + 2 }, (_, t): Cell => {
      const j = t - offset;
      return j >= 0 && j < chunks ? { t: 'op', label: `${tag}${j}`, color } : { t: 'idle' };
    }),
  });
  return (
    <Fig
      title="Host–device pipeline: 用多個 stream 疊 copy 與 compute"
      scroll
      caption={
        <>
          把輸入切成 chunk, 用三條 stream 分別跑 <span style={{ color: kStage }}>H2D copy</span>、
          <span style={{ color: kCompute }}> compute</span>、<span style={{ color: kD2H }}>D2H copy</span>。
          單 stream 串行要 <span className="font-mono">3·chunks</span> 步; 三 stream pipeline 後只要
          <span className="font-mono"> chunks + 2</span> 步 — copy 引擎 (DMA) 與 compute 引擎同時工作。
          前提: pinned memory + <span className="font-mono">non_blocking=True</span>, 否則 copy 會退回同步。
        </>
      }
    >
      <p className="mb-1 text-xs text-muted-foreground">單 stream — 串行</p>
      <Gantt rows={serialRows} rowLabelW={40} />
      <p className="mb-1 mt-4 text-xs text-muted-foreground">三 stream — 疊起來</p>
      <Gantt rows={[mk(kStage, 'H2D', 0), mk(kCompute, 'C', 1), mk(kD2H, 'D2H', 2)]} rowLabelW={40} />
      <div className="mt-3 flex flex-wrap gap-4">
        <Chip color={kStage}>H2D copy (DMA)</Chip>
        <Chip color={kCompute}>compute (SM)</Chip>
        <Chip color={kD2H}>D2H copy (DMA)</Chip>
      </div>
    </Fig>
  );
}

/* ------------------------------------------------------------------ */
/* 3. Pipeline parallelism: GPipe vs 1F1B schedule                     */
/* ------------------------------------------------------------------ */

type Op = { type: 'F' | 'B'; j: number };

// In-order, dependency-driven greedy simulation of a pipeline-parallel
// schedule. F(d,j) needs F(d-1,j); B(d,j) needs B(d+1,j). Each op takes one
// step. This reproduces the canonical GPipe and 1F1B gantts (and their bubbles)
// rather than hand-drawing them, so the diagram stays honest.
function simulate(p: number, opsPerDevice: Op[][]): { grid: Cell[][]; steps: number } {
  const done = new Map<string, number>(); // key d,type,j -> finish step
  const key = (d: number, t: string, j: number) => `${d},${t},${j}`;
  const ptr = new Array(p).fill(0);
  const busyUntil = new Array(p).fill(-1); // last step index the device was busy
  const grid: Cell[][] = Array.from({ length: p }, () => []);
  const maxSteps = 400;
  let step = 0;
  const remaining = () => opsPerDevice.reduce((s, ops, d) => s + (ops.length - ptr[d]), 0);
  while (remaining() > 0 && step < maxSteps) {
    for (let d = 0; d < p; d++) {
      const ops = opsPerDevice[d];
      let placed = false;
      if (ptr[d] < ops.length && busyUntil[d] < step) {
        const op = ops[ptr[d]];
        let depOk = true;
        if (op.type === 'F' && d > 0) depOk = (done.get(key(d - 1, 'F', op.j)) ?? Infinity) < step;
        if (op.type === 'B' && d < p - 1) depOk = (done.get(key(d + 1, 'B', op.j)) ?? Infinity) < step;
        if (depOk) {
          grid[d][step] = { t: 'op', label: `${op.type}${op.j}`, color: op.type === 'F' ? kStage : kBack };
          done.set(key(d, op.type, op.j), step);
          busyUntil[d] = step;
          ptr[d]++;
          placed = true;
        }
      }
      if (!placed) grid[d][step] = { t: 'idle' };
    }
    step++;
  }
  return { grid, steps: step };
}

function gpipeOps(p: number, m: number): Op[][] {
  return Array.from({ length: p }, () => {
    const ops: Op[] = [];
    for (let j = 0; j < m; j++) ops.push({ type: 'F', j });
    for (let j = m - 1; j >= 0; j--) ops.push({ type: 'B', j }); // backward in reverse micro-batch order
    return ops;
  });
}

function f1b1Ops(p: number, m: number): Op[][] {
  return Array.from({ length: p }, (_, d) => {
    const warmup = Math.min(p - 1 - d, m);
    const ops: Op[] = [];
    let f = 0;
    let b = 0;
    for (let i = 0; i < warmup; i++) ops.push({ type: 'F', j: f++ });
    const steady = m - warmup;
    for (let i = 0; i < steady; i++) {
      ops.push({ type: 'F', j: f++ });
      ops.push({ type: 'B', j: b++ });
    }
    for (let i = 0; i < warmup; i++) ops.push({ type: 'B', j: b++ });
    return ops;
  });
}

function toRows(grid: Cell[][]): { label: string; cells: Cell[] }[] {
  return grid.map((cells, d) => ({ label: `GPU${d}`, cells }));
}

export function PipelineParallelFigure() {
  const p = 4;
  const m = 6;
  const gpipe = simulate(p, gpipeOps(p, m));
  const f1b1 = simulate(p, f1b1Ops(p, m));
  return (
    <Fig
      title="Pipeline parallelism: GPipe vs 1F1B (p=4 stages, m=6 micro-batches)"
      scroll
      caption={
        <>
          兩者把 model 切成 4 段放到 4 顆 GPU, 再把 batch 切成 6 個 micro-batch 灌進去。
          <strong> GPipe</strong> 先跑完所有 <span style={{ color: kStage }}>forward</span> 再跑所有
          <span style={{ color: kBack }}> backward</span> — 中間一大塊 bubble, 且要同時存住 m 份 activation。
          <strong> 1F1B</strong> 在 steady state 交錯 1 forward + 1 backward: <em>bubble 比例一樣</em>{' '}
          <span className="font-mono">(p−1)/(m+p−1)</span>, 但 backward 提早發生 → 每顆 GPU 最多只需存 ~p 份
          activation, <span style={{ color: kAccent }}>peak memory 大幅下降</span>。 想再縮小 bubble 要用
          interleaved 1F1B (virtual stages)。
        </>
      }
    >
      <p className="mb-1 text-xs text-muted-foreground">GPipe — fill-drain, 大 bubble, 高 activation memory</p>
      <Gantt rows={toRows(gpipe.grid)} cell={22} />
      <p className="mb-1 mt-4 text-xs text-muted-foreground">1F1B — 交錯, 同樣 bubble 但低 activation memory</p>
      <Gantt rows={toRows(f1b1.grid)} cell={22} />
      <div className="mt-3 flex flex-wrap gap-4">
        <Chip color={kStage}>forward (F)</Chip>
        <Chip color={kBack}>backward (B)</Chip>
        <Chip color={kIdleBg}>bubble (idle)</Chip>
      </div>
    </Fig>
  );
}

/* ------------------------------------------------------------------ */
/* 4. SGLang-style overlap scheduler: CPU sched vs GPU forward          */
/* ------------------------------------------------------------------ */

export function OverlapSchedulerFigure() {
  const n = 4;
  // Naive: CPU schedules batch j, THEN GPU runs it; each waits for the other.
  const naiveCpu: Cell[] = [];
  const naiveGpu: Cell[] = [];
  for (let j = 0; j < n; j++) {
    naiveCpu.push({ t: 'op', label: `S${j}`, color: kAccent });
    naiveCpu.push({ t: 'idle' });
    naiveGpu.push({ t: 'idle' });
    naiveGpu.push({ t: 'op', label: `R${j}`, color: kCompute });
  }
  // Overlap: CPU schedules batch j+1 while GPU runs batch j (one batch ahead).
  const ovCpu: Cell[] = [{ t: 'op', label: 'S0', color: kAccent }];
  const ovGpu: Cell[] = [{ t: 'idle' }];
  for (let j = 0; j < n; j++) {
    ovCpu.push(j + 1 < n ? { t: 'op', label: `S${j + 1}`, color: kAccent } : { t: 'idle' });
    ovGpu.push({ t: 'op', label: `R${j}`, color: kCompute });
  }
  return (
    <Fig
      title="推論服務的 overlap scheduler (SGLang zero-overhead 排程)"
      scroll
      caption={
        <>
          每個 decode step 之間, CPU 要做 batch 排程、radix cache lookup、sampling metadata、token 處理等工作。
          <strong> 樸素</strong>做法讓 GPU 等 CPU 排完才跑, 兩者輪流閒置。 <strong>Overlap scheduler</strong> 讓
          scheduler <span style={{ color: kAccent }}>領先一個 batch</span>: GPU 在跑 batch <span className="font-mono">n</span>
          {' '}時, CPU 已經在準備 batch <span className="font-mono">n+1</span> 的 metadata, 於是 GPU 幾乎不再有 CPU 造成的空檔。
          這與 host–device stream overlap 是同一個 idea, 只是 stage 換成「CPU 排程」與「GPU forward」。
        </>
      }
    >
      <p className="mb-1 text-xs text-muted-foreground">樸素 — CPU 與 GPU 輪流等待</p>
      <Gantt rows={[{ label: 'CPU', cells: naiveCpu }, { label: 'GPU', cells: naiveGpu }]} rowLabelW={40} />
      <p className="mb-1 mt-4 text-xs text-muted-foreground">overlap — scheduler 領先一個 batch</p>
      <Gantt rows={[{ label: 'CPU', cells: ovCpu }, { label: 'GPU', cells: ovGpu }]} rowLabelW={40} />
      <div className="mt-3 flex flex-wrap gap-4">
        <Chip color={kAccent}>CPU: schedule / cache / sampling (S)</Chip>
        <Chip color={kCompute}>GPU: model forward (R)</Chip>
      </div>
    </Fig>
  );
}

/* ------------------------------------------------------------------ */
/* 5. Warp specialization: producer / consumer inside one kernel        */
/* ------------------------------------------------------------------ */

export function WarpSpecializationFigure() {
  const slots = 4;
  return (
    <Fig
      title="Kernel 內的 pipeline: warp specialization (producer / consumer)"
      caption={
        <>
          現代 GEMM/attention kernel 把 block 內的 warp 分工: <span style={{ color: kStage }}>producer warp</span>{' '}
          用 async copy (cp.async / TMA) 把下一塊 tile 搬進 shared memory 的環狀 buffer,{' '}
          <span style={{ color: kCompute }}>consumer warp</span> 從 buffer 取出資料餵給 Tensor Core。 兩組 warp 靠
          buffer 的 full/empty 狀態 (barrier / mbarrier) 同步 — 這就是把 software pipeline 的 producer–consumer 模型
          直接搬進一個 kernel, 讓 memory 與 compute 引擎同時飽和。
        </>
      }
    >
      <div className="flex flex-wrap items-center justify-center gap-3">
        <div className="rounded-md border-2 px-3 py-3 text-center" style={{ borderColor: kStage }}>
          <p className="text-xs font-semibold text-foreground">Producer warps</p>
          <p className="text-[10px] text-muted-foreground">cp.async / TMA load</p>
        </div>
        <span className="text-lg text-muted-foreground" aria-hidden>→</span>
        <div className="text-center">
          <div className="flex gap-1">
            {Array.from({ length: slots }, (_, i) => (
              <div
                key={i}
                className="flex h-10 w-10 items-center justify-center rounded border text-[10px] font-mono"
                style={
                  i < 2
                    ? { backgroundColor: `${kAccent}22`, borderColor: kAccent, color: kAccent }
                    : { borderColor: 'var(--border, #30363d)', color: '#8b949e' }
                }
              >
                {i < 2 ? 'full' : '·'}
              </div>
            ))}
          </div>
          <p className="mt-1 text-[10px] text-muted-foreground">shared-memory ring buffer</p>
        </div>
        <span className="text-lg text-muted-foreground" aria-hidden>→</span>
        <div className="rounded-md border-2 px-3 py-3 text-center" style={{ borderColor: kCompute }}>
          <p className="text-xs font-semibold text-foreground">Consumer warps</p>
          <p className="text-[10px] text-muted-foreground">MMA / Tensor Core</p>
        </div>
      </div>
    </Fig>
  );
}

/* ------------------------------------------------------------------ */
/* 6. NVIDIA vs AMD library pipeline stack                             */
/* ------------------------------------------------------------------ */

const kNv = '#76b900'; // NVIDIA green
const kAmd = '#ed1c24'; // AMD red

export function VendorPipelineFigure() {
  // Each layer of the software-pipeline stack and how each vendor implements it.
  const layers: { role: string; nv: string; amd: string }[] = [
    { role: '非同步載入', nv: 'cp.async / TMA', amd: 'buffer_load → LDS · local prefetch' },
    { role: 'pipeline 抽象', nv: 'cutlass::Pipeline (mbarrier full/empty)', amd: 'CK BlockwiseGemmPipeline' },
    { role: 'warp/wave 排程', nv: 'warp-specialized (producer/consumer)', amd: 'intrawave / interwave scheduling' },
    { role: 'matrix engine', nv: 'WGMMA · Tensor Core', amd: 'MFMA · Matrix Core' },
    { role: 'tuned GEMM 庫', nv: 'cuBLASLt (Tensile)', amd: 'hipBLASLt (Tensile)' },
    { role: '推論 op 庫', nv: 'CUTLASS · FlashInfer · TRT-LLM', amd: 'AITER (CK / Triton / ASM)' },
  ];
  return (
    <Fig
      title="同一套 pipeline 概念, 兩家的實作對照"
      scroll
      caption={
        <>
          由下 (最靠硬體) 到上 (serving): 兩家用不同名字實作<em>同一套</em> software-pipeline 機制。
          NVIDIA 走 <span style={{ color: kNv }}>cp.async/TMA + cutlass::Pipeline + warp specialization</span>;
          AMD 走 <span style={{ color: kAmd }}>local prefetch + CK BlockwiseGemmPipeline + intra/interwave 排程</span>。
          最上層的 <strong>AITER</strong> 把這些 CK/Triton/assembly kernel 打包成算子庫, 當 vLLM/SGLang 在 MI300X 上的
          預設 backend — 對應 NVIDIA 側的 CUTLASS/FlashInfer。
        </>
      }
    >
      <div style={{ display: 'grid', gridTemplateColumns: '120px 1fr 1fr', gap: 6, minWidth: 560 }}>
        <div />
        <div className="rounded-md px-2 py-1.5 text-center text-sm font-semibold" style={{ backgroundColor: `${kNv}22`, color: kNv }}>
          NVIDIA (CUDA)
        </div>
        <div className="rounded-md px-2 py-1.5 text-center text-sm font-semibold" style={{ backgroundColor: `${kAmd}22`, color: kAmd }}>
          AMD (ROCm)
        </div>
        {layers.map((l) => (
          <div key={l.role} style={{ display: 'contents' }}>
            <div className="flex items-center justify-end pr-1 text-right text-[11px] font-medium text-muted-foreground">{l.role}</div>
            <div className="rounded-md border px-2 py-2 text-center text-[11px] text-foreground" style={{ borderColor: kNv }}>
              <span className="font-mono">{l.nv}</span>
            </div>
            <div className="rounded-md border px-2 py-2 text-center text-[11px] text-foreground" style={{ borderColor: kAmd }}>
              <span className="font-mono">{l.amd}</span>
            </div>
          </div>
        ))}
      </div>
    </Fig>
  );
}
