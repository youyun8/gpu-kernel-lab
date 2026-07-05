'use client';

import { useMemo, useState } from 'react';

const kNumLanes = 8;
const kGatherInputSize = 16;
const kNumBins = 8;
const kCellsPerSegment = 4; // schematic transaction granularity

const kLaneColors = ['#39d353', '#58a6ff', '#f778ba', '#ffa657', '#a371f7', '#f85149', '#2f81f7', '#d29922'];

function randomIndices(count: number, range: number): number[] {
  return Array.from({ length: count }, () => Math.floor(Math.random() * range));
}

const kDefaultGatherIdx = [14, 2, 9, 2, 7, 11, 0, 5];
const kDefaultScatterIdx = [3, 6, 3, 1, 6, 3, 4, 6];

interface PanelGeometry {
  cell_h: number;
  array_top: number;
  laneTop: number;
}

const kGeom: PanelGeometry = { cell_h: 22, array_top: 28, laneTop: 60 };

function segmentsTouched(indices: number[]): number {
  return new Set(indices.map((i) => Math.floor(i / kCellsPerSegment))).size;
}

/**
 * GatherScatterVisualizer puts gather (scattered reads, contiguous writes)
 * and scatter (contiguous reads, scattered writes + atomic collisions) side
 * by side, with a toggle showing how warp-level aggregation reduces the
 * number of atomic operations a scatter needs.
 */
export function GatherScatterVisualizer() {
  const [gather_idx, setGatherIdx] = useState<number[]>(kDefaultGatherIdx);
  const [scatter_idx, setScatterIdx] = useState<number[]>(kDefaultScatterIdx);
  const [aggregate, setAggregate] = useState(false);

  const gather_read_segments = useMemo(() => segmentsTouched(gather_idx), [gather_idx]);
  const gather_write_segments = kNumLanes / kCellsPerSegment; // contiguous output
  const distinct_bins = useMemo(() => new Set(scatter_idx).size, [scatter_idx]);
  const atomic_ops = aggregate ? distinct_bins : kNumLanes;

  const shuffle = () => {
    setGatherIdx(randomIndices(kNumLanes, kGatherInputSize));
    setScatterIdx(randomIndices(kNumLanes, kNumBins));
  };

  const laneY = (lane: number) => kGeom.laneTop + lane * (kGeom.cell_h + 4);

  return (
    <div className="my-6 rounded-lg border border-border bg-card/40 p-5">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <p className="text-base font-semibold text-foreground">Gather vs. Scatter</p>
        <div className="flex items-center gap-3">
          <label className="flex cursor-pointer items-center gap-2 text-sm text-muted-foreground">
            <input
              type="checkbox"
              checked={aggregate}
              onChange={(e) => setAggregate(e.target.checked)}
              className="accent-brand"
              aria-label="scatter 端先做 warp aggregation"
            />
            Warp aggregation (scatter)
          </label>
          <button
            type="button"
            onClick={shuffle}
            className="rounded-md border border-border px-3 py-1.5 text-sm text-muted-foreground transition hover:border-primary hover:text-foreground"
          >
            換一組 indices
          </button>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        {/* Gather: out[i] = in[idx[i]] */}
        <div className="rounded-md bg-background p-3">
          <p className="mb-1 text-sm font-medium text-foreground">
            Gather <code className="text-xs text-muted-foreground">out[i] = in[idx[i]]</code>
          </p>
          <svg
            viewBox="0 0 300 260"
            className="w-full"
            role="img"
            aria-label={`gather: 8 個 lanes 從分散位置讀取, 觸及 ${gather_read_segments} 個 read segment, 寫出連續, 只需 ${gather_write_segments} 個 write segment`}
          >
            {/* input array (scattered reads) */}
            <text x="8" y="18" fontSize="10" fill="hsl(var(--muted-foreground))">
              in[] (讀: 分散)
            </text>
            {Array.from({ length: kGatherInputSize }, (_, i) => {
              const y = 26 + i * 14;
              const lane = gather_idx.indexOf(i);
              return (
                <g key={i}>
                  <rect x="8" y={y} width="36" height="12" rx="2" fill={lane >= 0 ? kLaneColors[lane] : 'hsl(var(--muted))'} opacity={lane >= 0 ? 0.9 : 0.5} />
                  <text x="26" y={y + 9} fontSize="8" textAnchor="middle" fill="hsl(var(--foreground))">
                    {i}
                  </text>
                </g>
              );
            })}
            {/* lanes */}
            <text x="128" y="18" fontSize="10" fill="hsl(var(--muted-foreground))">
              lanes
            </text>
            {Array.from({ length: kNumLanes }, (_, lane) => {
              const y = laneY(lane);
              const src_y = 26 + gather_idx[lane] * 14 + 6;
              const dst_y = 26 + lane * 22 + 8;
              return (
                <g key={lane}>
                  <line x1="44" y1={src_y} x2="128" y2={y + 10} stroke={kLaneColors[lane]} strokeWidth="1.5" opacity="0.75" />
                  <rect x="128" y={y} width="40" height="20" rx="3" fill={kLaneColors[lane]} opacity="0.9" />
                  <text x="148" y={y + 13} fontSize="9" textAnchor="middle" fill="#000">
                    L{lane}
                  </text>
                  <line x1="168" y1={y + 10} x2="252" y2={dst_y} stroke={kLaneColors[lane]} strokeWidth="1.5" opacity="0.75" />
                </g>
              );
            })}
            {/* output array (contiguous writes) */}
            <text x="252" y="18" fontSize="10" fill="hsl(var(--muted-foreground))">
              out[] (寫: 連續)
            </text>
            {Array.from({ length: kNumLanes }, (_, i) => {
              const y = 26 + i * 22;
              return (
                <g key={i}>
                  <rect x="252" y={y} width="36" height="16" rx="2" fill={kLaneColors[i]} opacity="0.9" />
                  <text x="270" y={y + 11} fontSize="8" textAnchor="middle" fill="#000">
                    {i}
                  </text>
                </g>
              );
            })}
          </svg>
          <p className="mt-1 text-xs text-muted-foreground">
            read segments: <span className="font-mono text-[#ffa657]">{gather_read_segments}</span> · write segments:{' '}
            <span className="font-mono text-primary">{gather_write_segments}</span> · 不需要 atomic
          </p>
        </div>

        {/* Scatter: out[idx[i]] += in[i] */}
        <div className="rounded-md bg-background p-3">
          <p className="mb-1 text-sm font-medium text-foreground">
            Scatter <code className="text-xs text-muted-foreground">atomicAdd(&out[idx[i]], in[i])</code>
          </p>
          <svg
            viewBox="0 0 300 260"
            className="w-full"
            role="img"
            aria-label={`scatter: 8 個 lanes 寫到 ${distinct_bins} 個不同 bin, ${aggregate ? '先做 warp aggregation 後' : '未做 aggregation 時'}需要 ${atomic_ops} 次 atomic`}
          >
            {/* input (contiguous reads) */}
            <text x="8" y="18" fontSize="10" fill="hsl(var(--muted-foreground))">
              in[] (讀: 連續)
            </text>
            {Array.from({ length: kNumLanes }, (_, i) => {
              const y = 26 + i * 22;
              return (
                <g key={i}>
                  <rect x="8" y={y} width="36" height="16" rx="2" fill={kLaneColors[i]} opacity="0.9" />
                  <text x="26" y={y + 11} fontSize="8" textAnchor="middle" fill="#000">
                    {i}
                  </text>
                </g>
              );
            })}
            {/* lanes and lines to bins */}
            <text x="128" y="18" fontSize="10" fill="hsl(var(--muted-foreground))">
              lanes
            </text>
            {Array.from({ length: kNumLanes }, (_, lane) => {
              const y = laneY(lane);
              const src_y = 26 + lane * 22 + 8;
              const dst_y = 26 + scatter_idx[lane] * 26 + 10;
              // With aggregation only the first lane hitting a bin issues the atomic.
              const is_leader = scatter_idx.indexOf(scatter_idx[lane]) === lane;
              const dimmed = aggregate && !is_leader;
              return (
                <g key={lane}>
                  <line x1="44" y1={src_y} x2="128" y2={y + 10} stroke={kLaneColors[lane]} strokeWidth="1.5" opacity="0.75" />
                  <rect x="128" y={y} width="40" height="20" rx="3" fill={kLaneColors[lane]} opacity="0.9" />
                  <text x="148" y={y + 13} fontSize="9" textAnchor="middle" fill="#000">
                    L{lane}
                  </text>
                  <line
                    x1="168"
                    y1={y + 10}
                    x2="252"
                    y2={dst_y}
                    stroke={kLaneColors[lane]}
                    strokeWidth={dimmed ? 1 : 1.5}
                    strokeDasharray={dimmed ? '3 3' : undefined}
                    opacity={dimmed ? 0.3 : 0.75}
                  />
                </g>
              );
            })}
            {/* output bins with collision counts */}
            <text x="240" y="18" fontSize="10" fill="hsl(var(--muted-foreground))">
              out[] (寫: 發散)
            </text>
            {Array.from({ length: kNumBins }, (_, bin) => {
              const y = 26 + bin * 26;
              const writers = scatter_idx.filter((t) => t === bin).length;
              return (
                <g key={bin}>
                  <rect
                    x="252"
                    y={y}
                    width="36"
                    height="20"
                    rx="2"
                    fill={writers > 1 ? '#f85149' : writers === 1 ? 'hsl(var(--accent))' : 'hsl(var(--muted))'}
                    opacity={writers > 0 ? 0.85 : 0.45}
                  />
                  <text x="270" y={y + 13} fontSize="8" textAnchor="middle" fill="hsl(var(--foreground))">
                    {bin}
                    {writers > 1 ? ` ×${writers}` : ''}
                  </text>
                </g>
              );
            })}
          </svg>
          <p className="mt-1 text-xs text-muted-foreground">
            atomic ops: <span className={`font-mono ${aggregate ? 'text-primary' : 'text-[#ff7b72]'}`}>{atomic_ops}</span>
            {aggregate ? ' (warp 內同 bin 先合併, 每個 bin 只發 1 次 atomic)' : ` (每 lane 各發 1 次; 紅色 bin 表示 ${kNumLanes - distinct_bins} 次衝突競爭)`}
          </p>
        </div>
      </div>

      <p className="mt-3 text-xs text-muted-foreground">
        Gather 的成本在讀端: index 越分散, touched read segments 越多; 但寫端天生連續, 也不需要 atomic。 Scatter 反過來: 讀端連續,
        寫端不但發散, 多個 lanes 撞到同一位置時還必須用 atomic 保證正確 — 勾選 warp aggregation 可以看到「warp 內先合併、再由一個
        lane 代表發 atomic」如何把 atomic 次數從 lanes 數降到 distinct bins 數。
      </p>
    </div>
  );
}
