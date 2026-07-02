'use client';

import { useEffect, useMemo, useState } from 'react';

const kMatrixDim = 8;

type Phase = 'load-a' | 'load-b' | 'compute';

const phaseLabels: Record<Phase, string> = {
  'load-a': '① global → shared:載入 A 的一個 tile',
  'load-b': '② global → shared:載入 B 的一個 tile',
  'compute': '③ shared → register:累加 partial products',
};

/**
 * TilingAnimator steps through GEMM shared-memory tiling: for a chosen tile
 * size it animates loading A/B tiles into shared memory and accumulating into
 * registers, and reports the data-reuse factor each tile achieves.
 */
export function TilingAnimator() {
  const [tileSize, setTileSize] = useState(2);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);

  const numTiles = kMatrixDim / tileSize;
  const totalSteps = numTiles * 3; // per k-tile: load A, load B, compute

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => setStep((s) => (s + 1) % totalSteps), 900);
    return () => clearInterval(id);
  }, [playing, totalSteps]);

  useEffect(() => {
    setStep(0);
  }, [tileSize]);

  const kTile = Math.floor(step / 3);
  const phase = (['load-a', 'load-b', 'compute'] as Phase[])[step % 3];

  // Reuse factor: each element loaded into shared memory is read tileSize times
  // by the threads computing a tile row/column.
  const reuse = tileSize;

  const cells = useMemo(() => Array.from({ length: kMatrixDim * kMatrixDim }), []);

  function cellColor(row: number, col: number): string {
    const inKBand = Math.floor(col / tileSize) === kTile; // for A: columns are k
    const inKBandRow = Math.floor(row / tileSize) === kTile; // for B: rows are k
    if (phase === 'load-a' && inKBand) return '#58a6ff';
    if (phase === 'load-b' && inKBandRow) return '#f778ba';
    if (phase === 'compute') return '#39d353';
    return '#21262d';
  }

  return (
    <div className="my-6 rounded-lg border border-surface-border bg-surface-raised/40 p-5">
      <p className="mb-4 text-base font-semibold text-white">Tiling Animator</p>

      <div className="grid gap-4 sm:grid-cols-2">
        <label className="text-sm text-slate-300">
          Tile size:<span className="ml-2 font-mono text-brand">{tileSize}×{tileSize}</span>
          <input type="range" min={1} max={4} step={1} value={tileSize} onChange={(e) => setTileSize(Number(e.target.value))} className="mt-1 w-full accent-brand" aria-label="tile size" />
        </label>
        <div className="flex items-end gap-2">
          <button onClick={() => setPlaying((p) => !p)} className="rounded-md border border-surface-border px-3 py-1.5 text-sm text-white hover:border-brand">
            {playing ? '暫停' : '播放'}
          </button>
          <button onClick={() => setStep((s) => (s + 1) % totalSteps)} className="rounded-md border border-surface-border px-3 py-1.5 text-sm text-white hover:border-brand">
            下一步
          </button>
        </div>
      </div>

      <p className="mt-4 text-sm text-slate-200">{phaseLabels[phase]}<span className="ml-2 text-slate-400">(k-tile {kTile + 1} / {numTiles})</span></p>

      <div className="mt-3 grid gap-1" style={{ gridTemplateColumns: `repeat(${kMatrixDim}, minmax(0, 1fr))` }} role="img" aria-label={`GEMM tiling 動畫,目前階段:${phaseLabels[phase]}`}>
        {cells.map((_, i) => {
          const row = Math.floor(i / kMatrixDim);
          const col = i % kMatrixDim;
          return <div key={i} className="aspect-square rounded-sm transition-colors" style={{ backgroundColor: cellColor(row, col) }} />;
        })}
      </div>

      <dl className="mt-5 grid grid-cols-2 gap-3 text-sm sm:grid-cols-3">
        <div className="rounded-md bg-surface p-3">
          <dt className="text-xs text-slate-400">K-tiles</dt>
          <dd className="font-mono text-lg text-white">{numTiles}</dd>
        </div>
        <div className="rounded-md bg-surface p-3">
          <dt className="text-xs text-slate-400">Reuse / element</dt>
          <dd className="font-mono text-lg text-brand">{reuse}×</dd>
        </div>
        <div className="rounded-md bg-surface p-3">
          <dt className="text-xs text-slate-400">Global loads 減少</dt>
          <dd className="font-mono text-lg text-white">≈ {(1 - 1 / reuse) * 100 || 0}%</dd>
        </div>
      </dl>
      <p className="mt-3 text-xs text-slate-400">
        tile 越大,每筆從 global memory 載入 shared memory 的資料被 register 重複使用的次數越多,arithmetic intensity 越高。代價是更多 shared memory 與 register 壓力,可能壓低 occupancy。
      </p>
    </div>
  );
}
