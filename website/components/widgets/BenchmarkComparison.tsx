'use client';

import { useState } from 'react';
import { Bar, BarChart, CartesianGrid, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import benchmarks from '@/content/data/benchmarks.json';

type Platform = 'cuda' | 'rocm';

export function BenchmarkComparison() {
  const [platform, setPlatform] = useState<Platform>('cuda');
  const dataset = benchmarks.gemm[platform];
  const chartData = dataset.steps.map((s) => ({ ...s, shortName: s.name.replace(/^Step \d+: /, '') }));

  return (
    <div className="my-6 rounded-lg border border-surface-border bg-surface-raised/40 p-5">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <p className="text-base font-semibold text-white">GEMM 優化每一步 (% of {dataset.reference})</p>
        <PlatformToggle platform={platform} onChange={setPlatform} />
      </div>
      <p className="mb-4 text-xs text-slate-400">
        {dataset.device} · peak ≈ {dataset.peakTflops} TFLOP/s · <span className="text-[#ffa657]">示意數據 (illustrative)</span>
      </p>
      <div className="h-80" aria-hidden>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 16, right: 8, bottom: 60, left: -8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
            <XAxis dataKey="shortName" stroke="#8b949e" tick={{ fontSize: 10 }} angle={-30} textAnchor="end" interval={0} height={70} />
            <YAxis stroke="#8b949e" tick={{ fontSize: 11 }} domain={[0, 100]} />
            <Tooltip contentStyle={{ background: '#161b22', border: '1px solid #30363d', color: '#fff' }} formatter={(v: number, _n, p) => [`${v}% · ${p.payload.gflops} GFLOP/s`, 'performance']} />
            <Bar dataKey="pctOfRef" radius={[4, 4, 0, 0]}>
              <LabelList dataKey="pctOfRef" position="top" fill="#c9d1d9" fontSize={10} formatter={(v: number) => `${v}%`} />
              {chartData.map((_, i) => (
                <Cell key={i} fill={`hsl(${135 + i * 8}, 55%, ${45 + i * 2}%)`} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <table className="mt-4 w-full text-left text-xs">
        <thead>
          <tr className="text-slate-400">
            <th className="py-1">Step</th>
            <th className="py-1">GFLOP/s</th>
            <th className="py-1">% of {dataset.reference}</th>
          </tr>
        </thead>
        <tbody className="font-mono text-slate-200">
          {dataset.steps.map((s) => (
            <tr key={s.name} className="border-t border-surface-border">
              <td className="py-1 pr-2 font-sans">{s.name}</td>
              <td className="py-1">{s.gflops}</td>
              <td className="py-1">{s.pctOfRef}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function PlatformToggle({ platform, onChange }: { platform: Platform; onChange: (p: Platform) => void }) {
  return (
    <div role="group" aria-label="平台切換" className="inline-flex overflow-hidden rounded-md border border-surface-border text-sm">
      {(['cuda', 'rocm'] as Platform[]).map((p) => (
        <button
          key={p}
          onClick={() => onChange(p)}
          aria-pressed={platform === p}
          className={`px-3 py-1 transition ${platform === p ? 'bg-brand text-black' : 'text-slate-300 hover:text-white'}`}
        >
          {p === 'cuda' ? 'CUDA' : 'ROCm'}
        </button>
      ))}
    </div>
  );
}
