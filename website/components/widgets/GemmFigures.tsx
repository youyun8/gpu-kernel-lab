import type { ReactNode } from 'react';

// Static, self-contained figures for the GEMM deep-dive track (Track 6). Each
// is a presentational component (no hooks, no client JS) so the chapter pages
// stay fast to render and export cleanly. Colors follow the site convention:
// A operand = blue, B operand = pink, C / accumulator = green, and the cyan
// track accent for whatever a figure wants to draw the eye to.

const kA = '#58a6ff'; // A operand
const kB = '#f778ba'; // B operand
const kC = '#39d353'; // C output / accumulator
const kAccent = '#39c5cf'; // track accent / "the point"
const kIdle = '#6e7681'; // idle / inactive

/** Shared frame + caption used by every figure below. */
function Fig({ title, caption, children }: { title: string; caption: ReactNode; children: ReactNode }) {
  return (
    <figure className="my-6 rounded-lg border border-border bg-card/40 p-5">
      <p className="mb-4 text-base font-semibold text-foreground">{title}</p>
      {children}
      <figcaption className="mt-4 text-xs leading-5 text-muted-foreground">{caption}</figcaption>
    </figure>
  );
}

/** A small legend chip: colored square + label. */
function Chip({ color, children }: { color: string; children: ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
      <span aria-hidden className="h-3 w-3 rounded-sm" style={{ backgroundColor: color }} />
      {children}
    </span>
  );
}

/** Render an r×c grid of unit cells; `fill(row,col)` returns a css color. */
function Grid({
  rows,
  cols,
  fill,
  cell = 14,
  label,
}: {
  rows: number;
  cols: number;
  fill: (row: number, col: number) => string;
  cell?: number;
  label?: string;
}) {
  return (
    <div
      role="img"
      aria-label={label}
      className="grid gap-px"
      style={{ gridTemplateColumns: `repeat(${cols}, ${cell}px)`, width: cols * (cell + 1) }}
    >
      {Array.from({ length: rows * cols }, (_, i) => {
        const row = Math.floor(i / cols);
        const col = i % cols;
        return <div key={i} className="rounded-[2px]" style={{ height: cell, backgroundColor: fill(row, col) }} />;
      })}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* 1. Reuse / arithmetic intensity: naive vs shared-memory tiling      */
/* ------------------------------------------------------------------ */

export function GemmReuseFigure() {
  return (
    <Fig
      title="為什麼 naive GEMM 卡在 bandwidth roof"
      caption={
        <>
          Naive: 每個 thread 為一個 <span style={{ color: kC }}>C</span> 元素讀 2K 個值只做 2K 次 FLOP,
          arithmetic intensity ≈ 0.25 FLOP/byte。 Tiled: 一個 BM×BN block 協力把 BK 寬的 tile 載入 shared memory,
          每筆 load 被 tile 內 O(BM) 或 O(BN) 個 thread 重複使用, intensity 隨 tile 尺寸線性上升 —
          點才會從 bandwidth roof 往 compute roof 爬。
        </>
      }
    >
      <div className="grid gap-6 sm:grid-cols-2">
        <div>
          <p className="mb-2 text-sm font-medium text-foreground">Naive — 無重複使用</p>
          <div className="flex items-center gap-3">
            <Grid rows={8} cols={1} cell={13} fill={(r) => (r >= 0 ? kA : kA)} label="A 的一整列 (K 個元素)" />
            <span className="text-lg text-muted-foreground" aria-hidden>×</span>
            <Grid rows={1} cols={8} cell={13} fill={() => kB} label="B 的一整行 (K 個元素)" />
            <span className="text-lg text-muted-foreground" aria-hidden>→</span>
            <div className="h-[13px] w-[13px] rounded-[2px]" style={{ backgroundColor: kC }} aria-hidden />
          </div>
          <p className="mt-2 font-mono text-xs text-muted-foreground">
            reuse = <span style={{ color: '#ff7b72' }}>1×</span> · AI ≈ 0.25
          </p>
        </div>

        <div>
          <p className="mb-2 text-sm font-medium text-foreground">Tiled — 進 shared memory 反覆用</p>
          <div className="flex items-center gap-3">
            <Grid rows={8} cols={2} cell={13} fill={() => kA} label="A 的 BM×BK tile" />
            <span className="text-lg text-muted-foreground" aria-hidden>×</span>
            <Grid rows={2} cols={8} cell={13} fill={() => kB} label="B 的 BK×BN tile" />
            <span className="text-lg text-muted-foreground" aria-hidden>→</span>
            <Grid rows={8} cols={8} cell={7} fill={() => kC} label="C 的 BM×BN block" />
          </div>
          <p className="mt-2 font-mono text-xs text-muted-foreground">
            reuse ≈ <span style={{ color: kAccent }}>BN× / BM×</span> · AI ↑ 隨 tile 尺寸
          </p>
        </div>
      </div>
    </Fig>
  );
}

/* ------------------------------------------------------------------ */
/* 2. Hierarchical tiling: block → warp → thread on the C matrix       */
/* ------------------------------------------------------------------ */

export function HierarchicalTilingFigure() {
  // 12x12 C-block; a warp tile is a 6x6 quadrant; a thread tile is a 2x3 cell.
  const warpRow = 0;
  const warpCol = 0; // highlighted warp tile = top-left quadrant
  const thRow = 0;
  const thCol = 0; // highlighted thread tile inside that warp

  function fill(row: number, col: number): string {
    const inWarp = Math.floor(row / 6) === warpRow && Math.floor(col / 6) === warpCol;
    const inThread = inWarp && Math.floor((row % 6) / 2) === thRow && Math.floor((col % 6) / 3) === thCol;
    if (inThread) return kAccent;
    if (inWarp) return kC;
    return kIdle;
  }

  return (
    <Fig
      title="Tiling 的三層階層 (在 C 上切)"
      caption={
        <>
          同一塊 <span style={{ color: kC }}>output C</span> 被三層切分:{' '}
          一個 <strong>threadblock</strong> 負責整塊 BM×BN;{' '}
          塊內每個 <strong>warp</strong> 認領一個 warp tile (<span style={{ color: kC }}>綠色象限</span>);{' '}
          warp 內每個 <strong>thread</strong> 再算一個 register-resident 的 micro-tile (
          <span style={{ color: kAccent }}>青色格</span>)。 每往下一層, operand 就被搬進更靠近 ALU、更快的儲存
          (shared memory → register), reuse 再乘一次。
        </>
      }
    >
      <div className="flex flex-wrap items-center gap-6">
        <div className="rounded-md border-2 border-dashed p-1.5" style={{ borderColor: kA }}>
          <Grid rows={12} cols={12} cell={16} fill={fill} label="threadblock 的 C block, 分成 warp tile 與 thread tile" />
        </div>
        <div className="space-y-2">
          <Chip color={kA}>threadblock tile (BM×BN)</Chip>
          <br />
          <Chip color={kC}>warp tile</Chip>
          <br />
          <Chip color={kAccent}>thread tile (registers, TM×TN)</Chip>
          <br />
          <Chip color={kIdle}>其他 warp 負責的區域</Chip>
        </div>
      </div>
    </Fig>
  );
}

/* ------------------------------------------------------------------ */
/* 3. Register tiling as an outer product                              */
/* ------------------------------------------------------------------ */

export function OuterProductFigure() {
  const TM = 4;
  const TN = 4;
  return (
    <Fig
      title="Register tiling = outer product 累加"
      caption={
        <>
          主迴圈每前進一個 k, 每個 thread 從 shared memory 讀 <span style={{ color: kA }}>TM 個 a</span> 和{' '}
          <span style={{ color: kB }}>TN 個 b</span> (共 TM+TN 筆), 做 <span style={{ color: kC }}>TM×TN 次 FMA</span>{' '}
          更新 register 裡的 accumulator。 讀 8 筆、算 16 次 — shared-memory 存取相對 FLOP 的比例又降一階,
          這通常是整條優化階梯上最大的單步跳躍。
        </>
      }
    >
      <div className="flex items-center gap-4">
        {/* b vector (row) sits above C */}
        <div className="w-[64px]" />
        <Grid rows={1} cols={TN} cell={22} fill={() => kB} label="b[TN] 從 shared memory 讀入 register" />
      </div>
      <div className="mt-1 flex items-center gap-4">
        <Grid rows={TM} cols={1} cell={22} fill={() => kA} label="a[TM] 從 shared memory 讀入 register" />
        <span className="text-lg text-muted-foreground" aria-hidden>⊗</span>
        <Grid rows={TM} cols={TN} cell={22} fill={() => kC} label="TM×TN accumulator, 全部在 register" />
        <span className="ml-2 font-mono text-xs text-muted-foreground">acc[i][j] += a[i] * b[j]</span>
      </div>
      <div className="mt-3 flex flex-wrap gap-4">
        <Chip color={kA}>a[TM] (registers)</Chip>
        <Chip color={kB}>b[TN] (registers)</Chip>
        <Chip color={kC}>acc[TM][TN] (registers)</Chip>
      </div>
    </Fig>
  );
}

/* ------------------------------------------------------------------ */
/* 4. Double buffering / software pipelining timeline                  */
/* ------------------------------------------------------------------ */

function TimelineRow({ segments }: { segments: { label: string; color: string; span: number }[] }) {
  return (
    <div className="flex gap-px">
      {segments.map((s, i) => (
        <div
          key={i}
          className="flex items-center justify-center rounded-[3px] py-1 text-[10px] font-medium text-black/80"
          style={{ backgroundColor: s.color, flexGrow: s.span, minWidth: 0 }}
        >
          <span className="truncate px-1">{s.label}</span>
        </div>
      ))}
    </div>
  );
}

export function DoubleBufferingFigure() {
  const load = kA;
  const compute = kC;
  const idle = 'transparent';
  return (
    <Fig
      title="Double buffering: 把 load 藏進 compute"
      caption={
        <>
          單 buffer 的每個 k-tile 成本是 <span className="font-mono">L + C</span> (load 時 ALU 閒著);
          double buffering 用兩組 shared-memory buffer, 算 <span style={{ color: compute }}>buffer 0</span> 的同時預取下一個 tile 進{' '}
          <span style={{ color: load }}>buffer 1</span>, 每個 k-tile 攤成 <span className="font-mono">max(L, C)</span>。
          當 compute 足以覆蓋 load, memory latency 就被完全藏掉 — Ampere 之後的 cp.async 讓 pipeline 能疊到 3–5 個 stage。
        </>
      }
    >
      <div className="space-y-3">
        <div>
          <p className="mb-1 text-xs text-muted-foreground">單 buffer — load 與 compute 串行</p>
          <TimelineRow
            segments={[
              { label: 'load k0', color: load, span: 2 },
              { label: 'compute k0', color: compute, span: 3 },
              { label: 'load k1', color: load, span: 2 },
              { label: 'compute k1', color: compute, span: 3 },
              { label: 'load k2', color: load, span: 2 },
              { label: 'compute k2', color: compute, span: 3 },
            ]}
          />
        </div>
        <div>
          <p className="mb-1 text-xs text-muted-foreground">double buffer — load 疊在 compute 之下</p>
          <TimelineRow
            segments={[
              { label: 'load k0', color: load, span: 2 },
              { label: 'compute k0', color: compute, span: 3 },
              { label: 'compute k1', color: compute, span: 3 },
              { label: 'compute k2', color: compute, span: 3 },
            ]}
          />
          <TimelineRow
            segments={[
              { label: '', color: idle, span: 2 },
              { label: 'prefetch k1', color: load, span: 3 },
              { label: 'prefetch k2', color: load, span: 3 },
              { label: '', color: idle, span: 3 },
            ]}
          />
        </div>
      </div>
      <div className="mt-3 flex gap-4">
        <Chip color={load}>global → shared load</Chip>
        <Chip color={compute}>compute (FMA / MMA)</Chip>
      </div>
    </Fig>
  );
}

/* ------------------------------------------------------------------ */
/* 5. Tensor Core / MFMA fragment                                      */
/* ------------------------------------------------------------------ */

export function TensorCoreFragmentFigure() {
  return (
    <Fig
      title="MMA: 一個 warp 共同持有一塊小矩陣"
      caption={
        <>
          Tensor Core (NVIDIA <span className="font-mono">wmma/mma</span>) 與 AMD <span className="font-mono">MFMA</span>{' '}
          以「warp/wavefront 共同持有 fragment」為單位運算: 一條指令就完成一整塊 M×N×K 的乘加 (圖示為 16×16×16)。
          fragment 在 register 的擺放由硬體規定, 所以 shared memory 必須用 <strong>swizzle</strong> 配合,
          才能無 bank conflict 地餵進 MMA — 這正是手寫困難、CUTLASS/CK 有價值之處。
        </>
      }
    >
      <div className="flex flex-wrap items-center gap-3">
        <div className="text-center">
          <Grid rows={8} cols={8} cell={12} fill={() => kA} label="A fragment 16×16 fp16" />
          <p className="mt-1 font-mono text-[10px]" style={{ color: kA }}>A frag 16×16</p>
        </div>
        <span className="text-lg text-muted-foreground" aria-hidden>×</span>
        <div className="text-center">
          <Grid rows={8} cols={8} cell={12} fill={() => kB} label="B fragment 16×16 fp16" />
          <p className="mt-1 font-mono text-[10px]" style={{ color: kB }}>B frag 16×16</p>
        </div>
        <span className="text-lg text-muted-foreground" aria-hidden>→</span>
        <div className="text-center">
          <Grid rows={8} cols={8} cell={12} fill={() => kC} label="C accumulator fragment 16×16 fp32" />
          <p className="mt-1 font-mono text-[10px]" style={{ color: kC }}>C acc 16×16 (fp32)</p>
        </div>
        <div className="ml-2 rounded-md border border-border bg-background px-3 py-2 text-center">
          <p className="font-mono text-sm" style={{ color: kAccent }}>1 條 mma_sync</p>
          <p className="text-[10px] text-muted-foreground">= 16×16×16 MACs</p>
          <p className="mt-1 text-[10px] text-muted-foreground">by 1 warp (32 lanes)</p>
        </div>
      </div>
    </Fig>
  );
}

/* ------------------------------------------------------------------ */
/* 6. Split-K                                                          */
/* ------------------------------------------------------------------ */

export function SplitKFigure() {
  const splits = [kA, kB, kC, kAccent];
  return (
    <Fig
      title="Split-K: 把 K 迴圈切給多個 CTA, 再 reduce"
      caption={
        <>
          當 M、N 小、K 很長時, 輸出 tile 數不足以填滿所有 SM (低 occupancy)。 Split-K 把 K 維切成{' '}
          <span className="font-mono">splitK</span> 段, <strong>每段一個 CTA</strong> 各算同一塊 C 的 partial sum,
          最後把 partials 相加 — 用平行度換一次額外的 reduction。 累加方式二選一:{' '}
          <span style={{ color: kAccent }}>atomicAdd</span> 直接加到 C (簡單但要處理 float 非結合性與競爭),
          或寫到 workspace 後跑一個獨立的 <strong>reduction kernel</strong> (可確定性)。
        </>
      }
    >
      <div className="flex flex-wrap items-center gap-4">
        <div>
          <p className="mb-1 text-xs text-muted-foreground">K 維被切成 4 段, 各由一個 CTA 處理</p>
          <div className="flex gap-1">
            {splits.map((c, i) => (
              <div key={i} className="text-center">
                <Grid rows={6} cols={2} cell={13} fill={() => c} label={`K 段 ${i} 由 CTA${i} 計算`} />
                <p className="mt-1 font-mono text-[10px] text-muted-foreground">CTA{i}</p>
              </div>
            ))}
          </div>
        </div>
        <div className="flex flex-col items-center">
          <span className="text-2xl text-muted-foreground" aria-hidden>→</span>
          <span className="text-[10px] text-muted-foreground">reduce</span>
        </div>
        <div className="text-center">
          <div className="relative">
            <Grid rows={6} cols={6} cell={13} fill={() => kC} label="reduce 後的完整 C tile" />
          </div>
          <p className="mt-1 font-mono text-[10px]" style={{ color: kC }}>
            C = Σ partials
          </p>
        </div>
      </div>
    </Fig>
  );
}

/* ------------------------------------------------------------------ */
/* 7. Stream-K                                                         */
/* ------------------------------------------------------------------ */

export function StreamKFigure() {
  // Total MAC work laid out as one 1D stream, evenly divided among 3 fixed CTAs.
  const total = 18; // work units
  const ctas = 3;
  const perCta = total / ctas;
  const ctaColors = [kA, kB, kC];
  return (
    <Fig
      title="Stream-K: 把所有 MAC 攤成一條 stream 均分"
      caption={
        <>
          Split-K 仍以「整塊 tile」為單位, 段數挑不好還是會有尾巴。 Stream-K 更進一步: 把所有 output tile 的
          K-loop 迭代<strong>接成一條連續的工作流</strong>, 再平均切給<strong>固定數量</strong>的常駐 CTA (通常 = SM 數)。
          每個 CTA 拿到等量的工作, 一個 tile 可能由多個 CTA 接力完成, 交界處做一次 partial reduction —
          幾乎消除 wave quantization 的尾端浪費, 代價是排程與 fix-up 邏輯更複雜。
        </>
      }
    >
      <div>
        <p className="mb-1 text-xs text-muted-foreground">6 個 output tile 的 K-iterations 串成一條 stream</p>
        <div className="flex gap-px">
          {Array.from({ length: total }, (_, i) => {
            const cta = Math.floor(i / perCta);
            const tileIdx = Math.floor(i / 3);
            const startOfTile = i % 3 === 0;
            return (
              <div
                key={i}
                className="flex h-8 items-center justify-center border-l text-[9px] text-black/70"
                style={{
                  backgroundColor: ctaColors[cta],
                  flexGrow: 1,
                  borderColor: startOfTile ? 'rgba(0,0,0,0.45)' : 'transparent',
                }}
                title={`tile ${tileIdx}`}
              >
                {startOfTile ? `t${tileIdx}` : ''}
              </div>
            );
          })}
        </div>
        <div className="mt-2 flex gap-4">
          {ctaColors.map((c, i) => (
            <Chip key={i} color={c}>
              CTA{i} (等量工作)
            </Chip>
          ))}
        </div>
      </div>
      <p className="mt-3 text-xs text-muted-foreground">
        深色分隔線 = tile 邊界。 注意 tile <span className="font-mono">t1</span>、<span className="font-mono">t4</span>{' '}
        橫跨兩個 CTA — 這種被切開的 tile 需要在結尾把兩段 partial 相加。
      </p>
    </Fig>
  );
}

/* ------------------------------------------------------------------ */
/* 8. Wave quantization / tail effect                                  */
/* ------------------------------------------------------------------ */

export function WaveQuantizationFigure() {
  // 10 tiles onto 4 SMs -> waves of 4,4,2. Third wave is the wasteful tail.
  const sms = 4;
  const waves = [
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9],
  ];
  return (
    <Fig
      title="Wave quantization: 半滿的最後一波"
      caption={
        <>
          GPU 一次能同時跑的 CTA 數有限 (受 SM 數與 occupancy 決定)。 CTA 以「wave」為單位排上去:{' '}
          10 個 tile 灑到 4 個 slot, 前兩波各 4 個滿載, 第三波只有 2 個 —{' '}
          <span style={{ color: '#ff7b72' }}>另一半 slot 空轉</span>, 但整個 kernel 的時間仍以完整的一波計。
          tile 數不是 slot 數的整數倍時, 這段尾巴就是浪費, 也是 Split-K / Stream-K / persistent scheduling 想解掉的東西。
        </>
      }
    >
      <div className="space-y-2">
        {waves.map((wave, w) => (
          <div key={w} className="flex items-center gap-2">
            <span className="w-14 shrink-0 text-xs text-muted-foreground">wave {w + 1}</span>
            <div className="flex gap-1">
              {Array.from({ length: sms }, (_, s) => {
                const tile = wave[s];
                const filled = tile !== undefined;
                return (
                  <div
                    key={s}
                    className="flex h-9 w-16 items-center justify-center rounded border text-[10px] font-mono"
                    style={
                      filled
                        ? { backgroundColor: kC, borderColor: kC, color: 'rgba(0,0,0,0.75)' }
                        : { borderColor: '#ff7b72', borderStyle: 'dashed', color: '#ff7b72' }
                    }
                  >
                    {filled ? `tile ${tile}` : '空轉'}
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </Fig>
  );
}

/* ------------------------------------------------------------------ */
/* 9. CTA swizzle for L2 locality                                      */
/* ------------------------------------------------------------------ */

export function CtaSwizzleFigure() {
  const dim = 6;
  // Linear order: row-major index. Swizzle: group columns into bands so
  // consecutively-launched CTAs stay spatially close -> reuse A/B rows in L2.
  function linearOrder(row: number, col: number) {
    return row * dim + col;
  }
  function swizzleOrder(row: number, col: number) {
    const band = 2;
    const bandId = Math.floor(col / band);
    const colInBand = col % band;
    return bandId * (dim * band) + row * band + colInBand;
  }
  function shade(order: number) {
    // map order 0..35 to a lightness ramp of the accent color
    const t = order / (dim * dim - 1);
    const light = 25 + t * 55; // %
    return `hsl(187 60% ${light}%)`;
  }
  return (
    <Fig
      title="CTA swizzle: 換 launch 順序換 L2 命中"
      caption={
        <>
          CTA 的排程順序決定同時在跑的 tile 落在 C 的哪裡。 <strong>Row-major</strong> 順序讓同時活躍的 tile 攤在一整列,
          共用的 A row / B column 很快被擠出 L2; <strong>swizzle</strong> (rasterization) 把 launch 順序重排成一塊塊,
          讓鄰近時間啟動的 tile 在空間上也相鄰, 於是它們共享的 <span style={{ color: kA }}>A</span> /{' '}
          <span style={{ color: kB }}>B</span> 條帶留在 L2 被重複命中。 色深 = launch 先後。
        </>
      }
    >
      <div className="flex flex-wrap items-center gap-8">
        <div className="text-center">
          <Grid rows={dim} cols={dim} cell={22} fill={(r, c) => shade(linearOrder(r, c))} label="row-major launch 順序" />
          <p className="mt-2 text-xs text-muted-foreground">row-major (差)</p>
        </div>
        <div className="text-center">
          <Grid rows={dim} cols={dim} cell={22} fill={(r, c) => shade(swizzleOrder(r, c))} label="swizzled launch 順序" />
          <p className="mt-2 text-xs text-muted-foreground">swizzled (好)</p>
        </div>
        <p className="text-xs text-muted-foreground">
          深 → 淺 = 先 → 後啟動。 swizzle 後<br />同期 tile 聚成方塊, L2 reuse ↑
        </p>
      </div>
    </Fig>
  );
}

/* ------------------------------------------------------------------ */
/* 10. Epilogue fusion pipeline                                        */
/* ------------------------------------------------------------------ */

export function EpilogueFigure() {
  const stages = [
    { label: 'K-loop mainloop', sub: 'MMA 累加 → acc (register)', color: kC },
    { label: 'α·acc', sub: 'scale', color: kAccent },
    { label: '+ β·C', sub: 'bias / residual', color: kA },
    { label: 'activation', sub: 'GELU / ReLU …', color: kB },
    { label: 'cast', sub: 'fp32 → fp16/fp8/int8', color: kIdle },
    { label: 'store C', sub: 'global memory', color: kC },
  ];
  return (
    <Fig
      title="Epilogue fusion: 寫回前把 elementwise 一起做完"
      caption={
        <>
          完整 GEMM 是 <span className="font-mono">C = α·AB + β·C</span>, 尾端還常掛 bias、activation、dtype cast。
          這段「accumulator 算完後、寫回 global memory 前」的程式叫 <strong>epilogue</strong>。 把這些 elementwise 融進
          epilogue, 就省掉一次完整的 C 讀寫 (否則要多一個 kernel 讀回 C 再寫出)。 這是 cuBLASLt 的 epilogue 選項、
          CUTLASS 的 epilogue visitor tree 提供 fused GEMM 的機制。
        </>
      }
    >
      <div className="flex flex-wrap items-stretch gap-1">
        {stages.map((s, i) => (
          <div key={i} className="flex items-center gap-1">
            <div
              className="flex min-w-[92px] flex-col justify-center rounded-md border px-2 py-2 text-center"
              style={{ borderColor: s.color }}
            >
              <span className="text-xs font-medium text-foreground">{s.label}</span>
              <span className="text-[10px] text-muted-foreground">{s.sub}</span>
            </div>
            {i < stages.length - 1 && (
              <span className="text-muted-foreground" aria-hidden>→</span>
            )}
          </div>
        ))}
      </div>
      <p className="mt-3 text-xs text-muted-foreground">
        <span style={{ color: kAccent }}>中間四格</span>全部在 register 上就地完成, 不落地 global memory — 這就是 fusion 省下的 traffic。
      </p>
    </Fig>
  );
}
