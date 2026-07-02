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
  smemPerBlockBytes: number;
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
    smemPerBlockBytes: 163 * 1024,
    regAllocUnit: 256,
  },
  H100: {
    name: 'NVIDIA H100 (SM90)',
    laneCount: 32,
    maxThreadsPerSM: 2048,
    maxWarpsPerSM: 64,
    maxBlocksPerSM: 32,
    regsPerSM: 65536,
    smemPerBlockBytes: 227 * 1024,
    regAllocUnit: 256,
  },
  MI250: {
    name: 'AMD MI250 (CDNA2)',
    laneCount: 64,
    maxThreadsPerSM: 2048,
    maxWarpsPerSM: 32,
    maxBlocksPerSM: 32,
    regsPerSM: 65536,
    smemPerBlockBytes: 64 * 1024,
    regAllocUnit: 512,
  },
  MI300: {
    name: 'AMD MI300 (CDNA3)',
    laneCount: 64,
    maxThreadsPerSM: 2048,
    maxWarpsPerSM: 32,
    maxBlocksPerSM: 32,
    regsPerSM: 65536,
    smemPerBlockBytes: 64 * 1024,
    regAllocUnit: 512,
  },
};

interface OccResult {
  occupancy: number;
  activeWarps: number;
  limiter: string;
  blocksByWarp: number;
  blocksByReg: number;
  blocksBySmem: number;
  blocksByHw: number;
}

function computeOccupancy(arch: Arch, regsPerThread: number, smemPerBlock: number, blockSize: number): OccResult {
  const warpsPerBlock = Math.ceil(blockSize / arch.laneCount);

  // Register limit: registers are allocated per warp, rounded to an allocation unit.
  const regsPerWarp = Math.ceil((regsPerThread * arch.laneCount) / arch.regAllocUnit) * arch.regAllocUnit;
  const warpsByReg = regsPerWarp > 0 ? Math.floor(arch.regsPerSM / regsPerWarp) : arch.maxWarpsPerSM;
  const blocksByReg = Math.floor(warpsByReg / warpsPerBlock);

  // Shared memory limit.
  const blocksBySmem = smemPerBlock > 0 ? Math.floor(arch.smemPerBlockBytes / smemPerBlock) : arch.maxBlocksPerSM;

  // Warp-count limit.
  const blocksByWarp = Math.floor(arch.maxWarpsPerSM / warpsPerBlock);

  // Hardware block limit.
  const blocksByHw = arch.maxBlocksPerSM;

  const activeBlocks = Math.max(0, Math.min(blocksByReg, blocksBySmem, blocksByWarp, blocksByHw));
  const activeWarps = Math.min(activeBlocks * warpsPerBlock, arch.maxWarpsPerSM);
  const occupancy = activeWarps / arch.maxWarpsPerSM;

  const limits: { name: string; value: number }[] = [
    { name: 'registers', value: blocksByReg },
    { name: 'shared memory', value: blocksBySmem },
    { name: 'warps per SM', value: blocksByWarp },
    { name: 'blocks per SM', value: blocksByHw },
  ];
  const minValue = Math.min(...limits.map((l) => l.value));
  const limiter = limits.filter((l) => l.value === minValue).map((l) => l.name).join(' + ');

  return { occupancy, activeWarps, limiter, blocksByReg, blocksBySmem, blocksByWarp, blocksByHw };
}

export function OccupancyCalculator() {
  const [archKey, setArchKey] = useState<keyof typeof kArchs>('A100');
  const [regs, setRegs] = useState(32);
  const [smem, setSmem] = useState(8192);
  const [blockSize, setBlockSize] = useState(256);

  const arch = kArchs[archKey];
  const result = useMemo(() => computeOccupancy(arch, regs, smem, blockSize), [arch, regs, smem, blockSize]);

  const curve = useMemo(() => {
    const points: { regs: number; occupancy: number }[] = [];
    for (let r = 16; r <= 128; r += 4) {
      points.push({ regs: r, occupancy: computeOccupancy(arch, r, smem, blockSize).occupancy * 100 });
    }
    return points;
  }, [arch, smem, blockSize]);

  return (
    <div className="my-6 rounded-lg border border-border bg-card/40 p-5">
      <p className="mb-4 text-base font-semibold text-foreground">Occupancy Calculator</p>

      <div className="grid gap-4 sm:grid-cols-2">
        <label className="text-sm text-muted-foreground">
          GPU 架構
          <select
            value={archKey}
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
          Block size (threads):<span className="ml-2 font-mono text-primary">{blockSize}</span>
          <input type="range" min={32} max={1024} step={32} value={blockSize} onChange={(e) => setBlockSize(Number(e.target.value))} className="mt-1 w-full accent-brand" aria-label="block size" />
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
            {result.activeWarps} / {arch.maxWarpsPerSM}
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
