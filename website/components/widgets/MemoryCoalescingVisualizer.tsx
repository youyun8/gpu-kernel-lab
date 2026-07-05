'use client';

import { useMemo, useState } from 'react';

const kWarpSize = 32;
const kBytesPerElement = 4; // float
const kSegmentBytes = 128; // one memory transaction covers a 128-byte segment
const kElementsPerSegment = kSegmentBytes / kBytesPerElement; // 32

/**
 * MemoryCoalescingVisualizer lets the reader change the access stride and base
 * offset for a single warp of 32 threads and shows how the accesses map onto
 * 128-byte memory transactions, plus the resulting bandwidth efficiency.
 */
export function MemoryCoalescingVisualizer() {
  const [stride, setStride] = useState(1);
  const [offset, setOffset] = useState(0);

  const { addresses, segments, efficiency } = useMemo(() => {
    const addrs = Array.from({ length: kWarpSize }, (_, lane) => (offset + lane * stride) * kBytesPerElement);
    const touched_segments = new Set(addrs.map((addr) => Math.floor(addr / kSegmentBytes)));
    const useful_bytes = kWarpSize * kBytesPerElement;
    const moved_bytes = touched_segments.size * kSegmentBytes;
    return {
      addresses: addrs,
      segments: [...touched_segments].sort((a, b) => a - b),
      efficiency: useful_bytes / moved_bytes,
    };
  }, [stride, offset]);

  const segment_index = new Map(segments.map((seg, i) => [seg, i]));
  const palette = ['#39d353', '#58a6ff', '#f778ba', '#ffa657', '#a371f7', '#f85149'];

  return (
    <div className="my-6 rounded-lg border border-border bg-card/40 p-5">
      <p className="mb-4 text-base font-semibold text-foreground">Memory Coalescing Visualizer</p>

      <div className="grid gap-4 sm:grid-cols-2">
        <label className="text-sm text-muted-foreground">
          Stride:<span className="ml-2 font-mono text-primary">{stride}</span>
          <input
            type="range"
            min={1}
            max={8}
            value={stride}
            onChange={(e) => setStride(Number(e.target.value))}
            className="mt-1 w-full accent-brand"
            aria-label="stride"
          />
        </label>
        <label className="text-sm text-muted-foreground">
          Offset:<span className="ml-2 font-mono text-primary">{offset}</span>
          <input
            type="range"
            min={0}
            max={31}
            value={offset}
            onChange={(e) => setOffset(Number(e.target.value))}
            className="mt-1 w-full accent-brand"
            aria-label="offset"
          />
        </label>
      </div>

      <div className="mt-5">
        <p className="mb-2 text-xs uppercase tracking-wide text-muted-foreground">32 threads → memory transactions</p>
        <div className="grid grid-cols-8 gap-1" role="img" aria-label={`一個 warp 的 32 個 threads,目前使用 ${segments.length} 個 memory transactions`}>
          {addresses.map((addr, lane) => {
            const seg = Math.floor(addr / kSegmentBytes);
            const color = palette[(segment_index.get(seg) ?? 0) % palette.length];
            return (
              <div
                key={lane}
                className="flex h-9 items-center justify-center rounded text-[10px] font-medium text-black"
                style={{ backgroundColor: color }}
                title={`lane ${lane} → byte ${addr} (segment ${seg})`}
              >
                {lane}
              </div>
            );
          })}
        </div>
      </div>

      <dl className="mt-5 grid grid-cols-2 gap-3 text-sm sm:grid-cols-3">
        <div className="rounded-md bg-background p-3">
          <dt className="text-xs text-muted-foreground">Transactions</dt>
          <dd className="font-mono text-lg text-foreground">{segments.length}</dd>
        </div>
        <div className="rounded-md bg-background p-3">
          <dt className="text-xs text-muted-foreground">最佳情況</dt>
          <dd className="font-mono text-lg text-foreground">{Math.ceil((kWarpSize * stride) / kElementsPerSegment) === 1 ? 1 : Math.ceil(kWarpSize / kElementsPerSegment)}</dd>
        </div>
        <div className="rounded-md bg-background p-3">
          <dt className="text-xs text-muted-foreground">Bandwidth efficiency</dt>
          <dd className={`font-mono text-lg ${efficiency > 0.9 ? 'text-primary' : efficiency > 0.4 ? 'text-[#ffa657]' : 'text-[#ff7b72]'}`}>
            {(efficiency * 100).toFixed(0)}%
          </dd>
        </div>
      </dl>

      <p className="mt-3 text-xs text-muted-foreground">
        當 <code className="text-primary">stride = 1</code> 且 offset 對齊時,32 個 threads 落在同一個 128-byte segment,只需 1 次 transaction,efficiency 達 100%。stride 越大,touched segments 越多,搬進來卻用不到的 bytes 就是浪費的 bandwidth。
      </p>
    </div>
  );
}
