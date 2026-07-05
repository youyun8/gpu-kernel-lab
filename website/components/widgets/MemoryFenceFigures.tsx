import type { ReactNode } from 'react';

// Static, self-contained figures for the C++ memory-order / CUDA memory-fence
// deep dive (Track 2, "Barriers and memory fences"). Same visual language as
// GemmFigures / PipelineFigures: presentational only (no hooks / client JS).

const kWrite = '#58a6ff'; // ordinary (non-atomic) write
const kRelease = '#39d353'; // release store / synchronizes-with source
const kAcquire = '#39c5cf'; // acquire load / synchronizes-with sink
const kBad = '#ff7b72'; // reordering / stale read / bug
const kIdle = 'rgba(110,118,129,0.18)'; // spin-wait / idle

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

/** One instruction/step box in a per-thread timeline column. */
function Step({ color, dim, children }: { color: string; dim?: boolean; children: ReactNode }) {
  return (
    <div
      className="rounded-[4px] px-2 py-1.5 text-center text-[11px] font-medium"
      style={{ backgroundColor: dim ? kIdle : color, color: dim ? 'inherit' : 'rgba(0,0,0,0.8)' }}
    >
      {children}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* 1. C++ release/acquire: synchronizes-with establishes happens-before */
/* ------------------------------------------------------------------ */

export function ReleaseAcquireFigure() {
  return (
    <Fig
      title="std::memory_order: release/acquire 如何建立 happens-before"
      scroll
      caption={
        <>
          左邊 (release/acquire) 的箭頭是 <strong>synchronizes-with</strong> 邊: C++ 標準保證, 只要 Thread B 的{' '}
          <span style={{ color: kAcquire }}>acquire load</span> 讀到 Thread A 那次{' '}
          <span style={{ color: kRelease }}>release store</span> 寫入的值, A 在 release 之前的所有記憶體操作 (包含普通的{' '}
          <span style={{ color: kWrite }}>payload = value</span>) 就對 B 在 acquire 之後的操作全部可見 — 這叫{' '}
          <strong>happens-before</strong>。 右邊把 store/load 都換成 <span className="font-mono">relaxed</span>:
          它仍然是 atomic (不會撕裂、不會 lost update), 但編譯器與 CPU 可以自由重排兩個「彼此無依賴」的寫入, 於是 B 可能看到{' '}
          <span className="font-mono">ready == true</span> 卻讀到還沒寫入的 <span className="font-mono">payload</span>
          — 這不是 data race (兩個 access 都是 atomic 或有同步), 而是<strong>同步不足</strong>。
        </>
      }
    >
      <div className="grid grid-cols-2 gap-6">
        <div>
          <p className="mb-2 text-center text-xs font-semibold" style={{ color: kAcquire }}>
            release / acquire — 正確
          </p>
          <div className="grid grid-cols-2 gap-x-6 gap-y-2">
            <div className="space-y-2">
              <p className="text-center font-mono text-[10px] text-muted-foreground">Thread A</p>
              <Step color={kWrite}>payload = value;</Step>
              <Step color={kRelease}>ready.store(true, release);</Step>
            </div>
            <div className="space-y-2">
              <p className="text-center font-mono text-[10px] text-muted-foreground">Thread B</p>
              <Step color={kIdle} dim>
                while (!ready.load(acquire)) {'{}'}
              </Step>
              <Step color={kAcquire}>ready.load(acquire) == true</Step>
            </div>
          </div>
          <div className="my-1 flex items-center justify-center text-[10px]" style={{ color: kAcquire }}>
            <span aria-hidden>release ─────synchronizes-with─────▶ acquire</span>
          </div>
          <div className="grid grid-cols-2 gap-x-6">
            <div />
            <Step color={kRelease}>use(payload); // guaranteed to see the new value</Step>
          </div>
        </div>
        <div>
          <p className="mb-2 text-center text-xs font-semibold" style={{ color: kBad }}>
            全用 relaxed — 同步不足
          </p>
          <div className="grid grid-cols-2 gap-x-6 gap-y-2">
            <div className="space-y-2">
              <p className="text-center font-mono text-[10px] text-muted-foreground">Thread A</p>
              <Step color={kWrite}>payload = value;</Step>
              <Step color={kBad}>ready.store(true, relaxed);</Step>
            </div>
            <div className="space-y-2">
              <p className="text-center font-mono text-[10px] text-muted-foreground">Thread B</p>
              <Step color={kIdle} dim>
                while (!ready.load(relaxed)) {'{}'}
              </Step>
              <Step color={kBad}>ready.load(relaxed) == true</Step>
            </div>
          </div>
          <div className="my-1 flex items-center justify-center text-[10px]" style={{ color: kBad }}>
            <span aria-hidden>編譯器/CPU 可能重排 ⇄ 無 happens-before</span>
          </div>
          <div className="grid grid-cols-2 gap-x-6">
            <div />
            <Step color={kBad}>use(payload); // may read a stale value</Step>
          </div>
        </div>
      </div>
      <div className="mt-4 flex flex-wrap gap-4">
        <Chip color={kWrite}>普通寫入 (non-atomic)</Chip>
        <Chip color={kRelease}>release store</Chip>
        <Chip color={kAcquire}>acquire load</Chip>
        <Chip color={kBad}>relaxed (無序保證)</Chip>
        <Chip color={kIdle} dashed>
          spin-wait
        </Chip>
      </div>
    </Fig>
  );
}

/* ------------------------------------------------------------------ */
/* 2. CUDA fence scope: block / device / system                        */
/* ------------------------------------------------------------------ */

function ScopeCard({
  fn,
  scopeLabel,
  contains,
  cost,
  note,
}: {
  fn: string;
  scopeLabel: string;
  contains: ReactNode;
  cost: number; // 1..3, relative
  note: ReactNode;
}) {
  return (
    <div className="flex flex-col rounded-md border border-border bg-background p-3">
      <p className="font-mono text-sm" style={{ color: kAcquire }}>
        {fn}
      </p>
      <p className="mt-1 text-[11px] font-semibold text-foreground">{scopeLabel}</p>
      <div className="my-2">{contains}</div>
      <div className="mt-auto">
        <p className="mb-1 text-[10px] text-muted-foreground">相對延遲</p>
        <div className="flex gap-0.5" aria-hidden>
          {[1, 2, 3].map((i) => (
            <div
              key={i}
              className="h-1.5 flex-1 rounded-full"
              style={{ backgroundColor: i <= cost ? kBad : kIdle }}
            />
          ))}
        </div>
      </div>
      <p className="mt-2 text-[11px] leading-4 text-muted-foreground">{note}</p>
    </div>
  );
}

export function CudaFenceScopeFigure() {
  const blockBox = (n: number, highlight = false) => (
    <div className="inline-flex gap-0.5 rounded-sm border border-border/60 p-0.5">
      {Array.from({ length: n }, (_, i) => (
        <div
          key={i}
          className="h-3 w-3 rounded-[2px]"
          style={{ backgroundColor: highlight && i === 0 ? kRelease : kWrite, opacity: highlight && i !== 0 ? 0.35 : 1 }}
        />
      ))}
    </div>
  );
  return (
    <Fig
      title="CUDA fence 的三種 scope: block / device / system"
      caption={
        <>
          Fence 本身<strong>不會讓你等任何人</strong> (那是 barrier 的工作), 它只保證「呼叫它的 thread, 在 fence 之前對記憶體的寫入,
          會依照 scope 所宣告的範圍變成對其他 thread 可見」。 Scope 越大, 需要硬體推得越遠 (SM 內 → 跨 SM 的 L2/device 一致點 →
          跨 PCIe/NVLink 到 host), 延遲也越高 — 這正是「只買你需要的可見範圍」的直接體現。
        </>
      }
    >
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <ScopeCard
          fn="__threadfence_block()"
          scopeLabel="同一個 block 內可見"
          contains={blockBox(6, true)}
          cost={1}
          note="用途: block-local 的 shared-memory producer/consumer 協議 (很少見, 通常 __syncthreads() 就夠)。"
        />
        <ScopeCard
          fn="__threadfence()"
          scopeLabel="同一個 device 上所有 block 可見"
          contains={
            <div className="flex flex-wrap gap-1">
              {Array.from({ length: 4 }, (_, i) => (
                <div key={i}>{blockBox(3, i === 0)}</div>
              ))}
            </div>
          }
          cost={2}
          note="用途: single-pass cross-block reduction 的 fence + atomic ticket 協議, 保證 partial sum 先發佈、其他 block 才看得到。"
        />
        <ScopeCard
          fn="__threadfence_system()"
          scopeLabel="host + 所有 device 可見"
          contains={
            <div className="space-y-1">
              <div className="flex flex-wrap gap-1">
                {Array.from({ length: 4 }, (_, i) => (
                  <div key={i}>{blockBox(3, i === 0)}</div>
                ))}
              </div>
              <p className="font-mono text-[9px] text-muted-foreground">+ host / 其他 GPU</p>
            </div>
          }
          cost={3}
          note="用途: mapped/pinned memory 或 multi-GPU 的 host-device 交握, 例如 GPU 寫完結果後舉旗給 host 輪詢的 signal。"
        />
      </div>
    </Fig>
  );
}

/* ------------------------------------------------------------------ */
/* 3. Single-pass reduction: fence + atomic ticket protocol             */
/* ------------------------------------------------------------------ */

function ProtocolRow({ label, cells }: { label: string; cells: { text: string; color: string; dim?: boolean }[] }) {
  return (
    <div className="flex items-stretch gap-1">
      <div className="flex w-24 shrink-0 items-center font-mono text-[10px] text-muted-foreground">{label}</div>
      <div className="flex flex-1 gap-1">
        {cells.map((c, i) => (
          <div
            key={i}
            className="flex flex-1 items-center justify-center rounded-[3px] px-1 py-1.5 text-center text-[10px] font-medium"
            style={{ backgroundColor: c.dim ? kIdle : c.color, color: c.dim ? 'inherit' : 'rgba(0,0,0,0.8)' }}
          >
            {c.text}
          </div>
        ))}
      </div>
    </div>
  );
}

export function CrossBlockFenceProtocolFigure() {
  return (
    <Fig
      title="跨 block reduction: 有 fence vs 沒有 fence"
      scroll
      caption={
        <>
          三個 block 各自算完 partial sum 後, 用 <span className="font-mono">atomicAdd(done, 1)</span> 領 ticket,
          最後一個抵達的 block (ticket == gridDim.x − 1) 負責讀所有 partials 加總。 <strong>有 fence</strong>{' '}
          (上半部) 時, <span style={{ color: kRelease }}>write partial</span> 保證排在{' '}
          <span style={{ color: kAcquire }}>atomicAdd ticket</span> 之前對其他 block 可見, 所以最後一個 block 讀到的每個
          partial 都是最終值。 <strong>沒有 fence</strong> (下半部) 時, atomic ticket 本身雖然還是正確地數到
          gridDim.x, 但硬體/編譯器可能讓「write partial」晚於「atomicAdd」才對其他 SM 可見 — 於是最後一個 block 可能在
          ticket 顯示全員到齊的當下, 讀到某個 block 尚未發佈的 <span style={{ color: kBad }}>stale partial</span>,
          算出錯誤的總和且結果不固定重現。
        </>
      }
    >
      <div className="space-y-4">
        <div>
          <p className="mb-1 text-xs font-semibold" style={{ color: kRelease }}>
            有 __threadfence() — 正確
          </p>
          <div className="space-y-1">
            <ProtocolRow
              label="block 0"
              cells={[
                { text: 'compute partial', color: kWrite },
                { text: 'write partial[0]', color: kRelease },
                { text: '__threadfence()', color: kAcquire },
                { text: 'atomicAdd(done)', color: kAcquire },
              ]}
            />
            <ProtocolRow
              label="block 1"
              cells={[
                { text: 'compute partial', color: kWrite },
                { text: 'write partial[1]', color: kRelease },
                { text: '__threadfence()', color: kAcquire },
                { text: 'atomicAdd(done)', color: kAcquire },
              ]}
            />
            <ProtocolRow
              label="block 2 (last)"
              cells={[
                { text: 'compute partial', color: kWrite },
                { text: 'write partial[2]', color: kRelease },
                { text: '__threadfence()', color: kAcquire },
                { text: 'ticket==2 → 讀 partial[0..2] 求和 ✓', color: kRelease },
              ]}
            />
          </div>
        </div>
        <div>
          <p className="mb-1 text-xs font-semibold" style={{ color: kBad }}>
            拿掉 __threadfence() — bug
          </p>
          <div className="space-y-1">
            <ProtocolRow
              label="block 0"
              cells={[
                { text: 'compute partial', color: kWrite },
                { text: 'write partial[0]', color: kBad },
                { text: 'atomicAdd(done)', color: kAcquire },
                { text: '(尚未全域可見)', color: kIdle, dim: true },
              ]}
            />
            <ProtocolRow
              label="block 1"
              cells={[
                { text: 'compute partial', color: kWrite },
                { text: 'write partial[1]', color: kBad },
                { text: 'atomicAdd(done)', color: kAcquire },
                { text: '(尚未全域可見)', color: kIdle, dim: true },
              ]}
            />
            <ProtocolRow
              label="block 2 (last)"
              cells={[
                { text: 'compute partial', color: kWrite },
                { text: 'write partial[2]', color: kBad },
                { text: 'atomicAdd(done)', color: kAcquire },
                { text: 'ticket==2 → 讀到 stale partial[0] ✗', color: kBad },
              ]}
            />
          </div>
        </div>
      </div>
      <div className="mt-4 flex flex-wrap gap-4">
        <Chip color={kWrite}>私有計算</Chip>
        <Chip color={kRelease}>寫入 + 已保證可見</Chip>
        <Chip color={kAcquire}>fence / atomic ticket</Chip>
        <Chip color={kBad}>可見性未保證 / 讀到 stale 值</Chip>
      </div>
    </Fig>
  );
}
