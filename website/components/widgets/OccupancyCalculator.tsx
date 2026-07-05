'use client';

import { useMemo, useState } from 'react';
import {
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  CartesianGrid,
  ReferenceLine,
} from 'recharts';

interface Arch {
  name: string;
  laneCount: number; // warp / wavefront size
  maxThreadsPerSM: number;
  maxWarpsPerSM: number;
  maxBlocksPerSM: number;
  regsPerSM: number;
  smem_per_block_bytes: number;
  regAllocUnit: number; // registers rounded per warp allocation
}

// Values are representative public figures; treated as illustrative defaults.
const kArchs: Record<string, Arch> = {
  A100: {
    name: 'NVIDIA A100 (SM80)',
    laneCount: 32,
    maxThreadsPerSM: 2048,
    maxWarpsPerSM: 64,
    maxBlocksPerSM: 32,
    regsPerSM: 65536,
    smem_per_block_bytes: 163 * 1024,
    regAllocUnit: 256,
  },
  H100: {
    name: 'NVIDIA H100 (SM90)',
    laneCount: 32,
    maxThreadsPerSM: 2048,
    maxWarpsPerSM: 64,
    maxBlocksPerSM: 32,
    regsPerSM: 65536,
    smem_per_block_bytes: 227 * 1024,
    regAllocUnit: 256,
  },
  MI250: {
    name: 'AMD MI250 (CDNA2)',
    laneCount: 64,
    maxThreadsPerSM: 2048,
    maxWarpsPerSM: 32,
    maxBlocksPerSM: 32,
    regsPerSM: 65536,
    smem_per_block_bytes: 64 * 1024,
    regAllocUnit: 512,
  },
  MI300: {
    name: 'AMD MI300 (CDNA3)',
    laneCount: 64,
    maxThreadsPerSM: 2048,
    maxWarpsPerSM: 32,
    maxBlocksPerSM: 32,
    regsPerSM: 65536,
    smem_per_block_bytes: 64 * 1024,
    regAllocUnit: 512,
  },
};

interface OccResult {
  occupancy: number;
  active_warps: number;
  limiter: string;
  blocks_by_warp: number;
  blocks_by_reg: number;
  blocks_by_smem: number;
  blocks_by_hw: number;
}

function computeOccupancy(arch: Arch, regsPerThread: number, smem_per_block: number, block_size: number): OccResult {
  const warps_per_block = Math.ceil(block_size / arch.laneCount);

  // Register limit: registers are allocated per warp, rounded to an allocation unit.
  const regsPerWarp = Math.ceil((regsPerThread * arch.laneCount) / arch.regAllocUnit) * arch.regAllocUnit;
  const warps_by_reg = regsPerWarp > 0 ? Math.floor(arch.regsPerSM / regsPerWarp) : arch.maxWarpsPerSM;
  const blocks_by_reg = Math.floor(warps_by_reg / warps_per_block);

  // Shared memory limit.
  const blocks_by_smem = smem_per_block > 0 ? Math.floor(arch.smem_per_block_bytes / smem_per_block) : arch.maxBlocksPerSM;

  // Warp-count limit.
  const blocks_by_warp = Math.floor(arch.maxWarpsPerSM / warps_per_block);

  // Hardware block limit.
  const blocks_by_hw = arch.maxBlocksPerSM;

  const active_blocks = Math.max(0, Math.min(blocks_by_reg, blocks_by_smem, blocks_by_warp, blocks_by_hw));
  const active_warps = Math.min(active_blocks * warps_per_block, arch.maxWarpsPerSM);
  const occupancy = active_warps / arch.maxWarpsPerSM;

  const limits: { name: string; value: number }[] = [
    { name: 'registers', value: blocks_by_reg },
    { name: 'shared memory', value: blocks_by_smem },
    { name: 'warps per SM', value: blocks_by_warp },
    { name: 'blocks per SM', value: blocks_by_hw },
  ];
  const min_value = Math.min(...limits.map((l) => l.value));
  const limiter = limits.filter((l) => l.value === min_value).map((l) => l.name).join(' + ');

  return { occupancy, active_warps, limiter, blocks_by_reg, blocks_by_smem, blocks_by_warp, blocks_by_hw };
}

export function OccupancyCalculator() {
  const [arch_key, setArchKey] = useState<keyof typeof kArchs>('A100');
  const [regs, setRegs] = useState(32);
  const [smem, setSmem] = useState(8192);
  const [block_size, setBlockSize] = useState(256);

  const arch = kArchs[arch_key];
  const result = useMemo(() => computeOccupancy(arch, regs, smem, block_size), [arch, regs, smem, block_size]);

  const curve = useMemo(() => {
    const points: { regs: number; occupancy: number }[] = [];
    for (let r = 16; r <= 128; r += 4) {
      points.push({ regs: r, occupancy: computeOccupancy(arch, r, smem, block_size).occupancy * 100 });
    }
    return points;
  }, [arch, smem, block_size]);

  return (
    <div className="my-6 rounded-lg border border-border bg-card/40 p-5">
      <p className="mb-4 text-base font-semibold text-foreground">Occupancy Calculator</p>

      <div className="grid gap-4 sm:grid-cols-2">
        <label className="text-sm text-muted-foreground">
          GPU 架構
          <select
            value={arch_key}
            onChange={(e) => setArchKey(e.target.value as keyof typeof kArchs)}
            className="mt-1 w-full rounded-md border border-border bg-background px-2 py-1.5 text-foreground"
          >
            {Object.entries(kArchs).map(([key, a]) => (
              <option key={key} value={key}>
                {a.name}
              </option>
            ))}
          </select>
        </label>
        <label className="text-sm text-muted-foreground">
          Block size (threads):<span className="ml-2 font-mono text-primary">{block_size}</span>
          <input type="range" min={32} max={1024} step={32} value={block_size} onChange={(e) => setBlockSize(Number(e.target.value))} className="mt-1 w-full accent-brand" aria-label="block size" />
        </label>
        <label className="text-sm text-muted-foreground">
          Registers / thread:<span className="ml-2 font-mono text-primary">{regs}</span>
          <input type="range" min={16} max={128} step={1} value={regs} onChange={(e) => setRegs(Number(e.target.value))} className="mt-1 w-full accent-brand" aria-label="registers per thread" />
        </label>
        <label className="text-sm text-muted-foreground">
          Shared memory / block (bytes):<span className="ml-2 font-mono text-primary">{smem}</span>
          <input type="range" min={0} max={49152} step={1024} value={smem} onChange={(e) => setSmem(Number(e.target.value))} className="mt-1 w-full accent-brand" aria-label="shared memory per block" />
        </label>
      </div>

      <dl className="mt-5 grid grid-cols-2 gap-3 text-sm sm:grid-cols-3">
        <div className="rounded-md bg-background p-3">
          <dt className="text-xs text-muted-foreground">Theoretical occupancy</dt>
          <dd className="font-mono text-lg text-primary">{(result.occupancy * 100).toFixed(1)}%</dd>
        </div>
        <div className="rounded-md bg-background p-3">
          <dt className="text-xs text-muted-foreground">Active warps / SM</dt>
          <dd className="font-mono text-lg text-foreground">
            {result.active_warps} / {arch.maxWarpsPerSM}
          </dd>
        </div>
        <div className="rounded-md bg-background p-3">
          <dt className="text-xs text-muted-foreground">Limiter</dt>
          <dd className="text-sm font-medium text-[#ffa657]">{result.limiter}</dd>
        </div>
      </dl>

      <div className="mt-5 h-56" aria-hidden>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={curve} margin={{ top: 8, right: 12, bottom: 8, left: -12 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
            <XAxis dataKey="regs" stroke="#8b949e" tick={{ fontSize: 11 }} label={{ value: 'registers / thread', position: 'insideBottom', offset: -4, fill: '#8b949e', fontSize: 11 }} />
            <YAxis stroke="#8b949e" tick={{ fontSize: 11 }} domain={[0, 100]} />
            <Tooltip contentStyle={{ background: '#161b22', border: '1px solid #30363d', color: '#fff' }} formatter={(v: number) => [`${v.toFixed(0)}%`, 'occupancy']} />
            <ReferenceLine x={regs} stroke="#f778ba" strokeDasharray="4 4" />
            <Line type="monotone" dataKey="occupancy" stroke="#39d353" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <p className="mt-1 text-xs text-muted-foreground">
        曲線:固定目前的 shared memory 與 block size,occupancy 隨 registers/thread 變化。粉紅線是目前的設定。注意 occupancy 常呈階梯狀下降 — 這是 register allocation 以 warp 為單位量化造成的。
      </p>
    </div>
  );
}
