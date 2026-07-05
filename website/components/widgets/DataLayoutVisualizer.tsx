'use client';

import { useMemo, useState } from 'react';

type Layout = 'aos' | 'soa' | 'aosoa';
type Field = 0 | 1 | 2;

const kWarpSize = 32;
const kNumParticles = 64; // memory window: 64 particles x 3 float fields = 6 segments
const kNumFields = 3;
const kBytesPerElement = 4;
const kSegmentBytes = 128;
const kFloatsPerSegment = kSegmentBytes / kBytesPerElement; // 32
const kTotalFloats = kNumParticles * kNumFields;

const kFieldNames = ['x', 'y', 'z'] as const;
const kFieldColors = ['#39d353', '#58a6ff', '#f778ba'] as const;

const kLayoutLabels: Record<Layout, string> = {
  aos: 'AoS',
  soa: 'SoA',
  aosoa: 'AoSoA (chunk = 32)',
};

// Byte-level position of field `f` of particle `i` for each layout, in floats.
function elementIndex(layout: Layout, i: number, f: Field): number {
  switch (layout) {
    case 'aos':
      return i * kNumFields + f;
    case 'soa':
      return f * kNumParticles + i;
    case 'aosoa': {
      const chunk = Math.floor(i / kWarpSize);
      const lane = i % kWarpSize;
      return chunk * kWarpSize * kNumFields + f * kWarpSize + lane;
    }
  }
}

/**
 * DataLayoutVisualizer shows how AoS / SoA / AoSoA place the same particle
 * data in memory, and which 128-byte segments one warp touches when every
 * lane reads the same field in the same cycle.
 */
export function DataLayoutVisualizer() {
  const [layout, setLayout] = useState<Layout>('aos');
  const [field, setField] = useState<Field>(0);

  const { field_of_cell, touched, segmentsTouched, efficiency } = useMemo(() => {
    // Map every float slot in the window back to the field stored there.
    const cell_field = new Array<Field>(kTotalFloats).fill(0);
    for (let i = 0; i < kNumParticles; ++i) {
      for (let f = 0; f < kNumFields; ++f) {
        cell_field[elementIndex(layout, i, f as Field)] = f as Field;
      }
    }
    // One warp: lanes 0..31 read `field` of particles 0..31 in the same cycle.
    const touched_cells = new Set<number>();
    for (let lane = 0; lane < kWarpSize; ++lane) {
      touched_cells.add(elementIndex(layout, lane, field));
    }
    const segments = new Set([...touched_cells].map((c) => Math.floor(c / kFloatsPerSegment)));
    const useful_bytes = kWarpSize * kBytesPerElement;
    return {
      field_of_cell: cell_field,
      touched: touched_cells,
      segmentsTouched: segments.size,
      efficiency: useful_bytes / (segments.size * kSegmentBytes),
    };
  }, [layout, field]);

  const rows = Array.from({ length: kTotalFloats / kFloatsPerSegment }, (_, r) => r);

  return (
    <div className="my-6 rounded-lg border border-border bg-card/40 p-5">
      <p className="mb-4 text-base font-semibold text-foreground">Data Layout Visualizer: AoS / SoA / AoSoA</p>

      <div className="flex flex-wrap items-center gap-4">
        <div role="radiogroup" aria-label="記憶體佈局" className="flex overflow-hidden rounded-md border border-border">
          {(Object.keys(kLayoutLabels) as Layout[]).map((key) => (
            <button
              key={key}
              type="button"
              role="radio"
              aria-checked={layout === key}
              onClick={() => setLayout(key)}
              className={`px-3 py-1.5 text-sm font-medium transition ${
                layout === key ? 'bg-primary/15 text-primary' : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              {kLayoutLabels[key]}
            </button>
          ))}
        </div>

        <div role="radiogroup" aria-label="warp 讀取的欄位" className="flex items-center gap-2 text-sm text-muted-foreground">
          <span>warp 讀取欄位:</span>
          {kFieldNames.map((name, f) => (
            <button
              key={name}
              type="button"
              role="radio"
              aria-checked={field === f}
              onClick={() => setField(f as Field)}
              className={`rounded px-2 py-1 font-mono transition ${
                field === f ? 'text-foreground ring-1 ring-primary' : 'hover:text-foreground'
              }`}
              style={{ color: kFieldColors[f] }}
            >
              .{name}
            </button>
          ))}
        </div>
      </div>

      <div className="mt-5 overflow-x-auto">
        <p className="mb-2 text-xs uppercase tracking-wide text-muted-foreground">
          記憶體視窗 (每列 = 一個 128-byte segment, 每格 = 一個 float)
        </p>
        <div
          role="img"
          aria-label={`${kLayoutLabels[layout]} 佈局下, 一個 warp 讀取欄位 ${kFieldNames[field]} 需要 ${segmentsTouched} 個 memory transaction`}
          className="min-w-[560px] space-y-1"
        >
          {rows.map((row) => {
            const row_touched = Array.from({ length: kFloatsPerSegment }, (_, c) => touched.has(row * kFloatsPerSegment + c)).some(Boolean);
            return (
              <div key={row} className="flex items-center gap-2">
                <span className={`w-14 shrink-0 text-right font-mono text-[10px] ${row_touched ? 'text-primary' : 'text-muted-foreground'}`}>
                  {row * kSegmentBytes}B
                </span>
                <div className={`grid flex-1 grid-cols-[repeat(32,minmax(0,1fr))] gap-px rounded p-0.5 ${row_touched ? 'bg-primary/20' : 'bg-background'}`}>
                  {Array.from({ length: kFloatsPerSegment }, (_, col) => {
                    const cell = row * kFloatsPerSegment + col;
                    const is_touched = touched.has(cell);
                    return (
                      <div
                        key={col}
                        className="h-4 rounded-[2px]"
                        style={{
                          backgroundColor: kFieldColors[field_of_cell[cell]],
                          opacity: is_touched ? 1 : 0.22,
                          outline: is_touched ? '1px solid hsl(var(--foreground))' : undefined,
                        }}
                        title={`float #${cell} = 欄位 ${kFieldNames[field_of_cell[cell]]}${is_touched ? ' (本 cycle 被 warp 讀取)' : ''}`}
                      />
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <dl className="mt-5 grid grid-cols-2 gap-3 text-sm sm:grid-cols-3">
        <div className="rounded-md bg-background p-3">
          <dt className="text-xs text-muted-foreground">Transactions (128B)</dt>
          <dd className="font-mono text-lg text-foreground">{segmentsTouched}</dd>
        </div>
        <div className="rounded-md bg-background p-3">
          <dt className="text-xs text-muted-foreground">Coalesced?</dt>
          <dd className={`font-mono text-lg ${segmentsTouched === 1 ? 'text-primary' : 'text-[#ff7b72]'}`}>
            {segmentsTouched === 1 ? 'Yes' : 'No'}
          </dd>
        </div>
        <div className="rounded-md bg-background p-3">
          <dt className="text-xs text-muted-foreground">Bandwidth efficiency</dt>
          <dd className={`font-mono text-lg ${efficiency > 0.9 ? 'text-primary' : 'text-[#ff7b72]'}`}>
            {(efficiency * 100).toFixed(0)}%
          </dd>
        </div>
      </dl>

      <p className="mt-3 text-xs text-muted-foreground">
        AoS 讓同一欄位相隔 3 個 float (stride 3), 一個 warp 讀 <code className="text-primary">.{kFieldNames[field]}</code> 要碰 3 個
        segment; SoA 與 AoSoA 都把同一欄位的 32 個值排在同一個 128-byte segment 裡, 1 次 transaction 完成。 兩者的差別在整體排列:
        AoSoA 以 warp 大小分塊, 同一 particle 的三個欄位仍在鄰近的 chunk 內, 對「同時要多個欄位」的 kernel 保留 locality。
      </p>
    </div>
  );
}
