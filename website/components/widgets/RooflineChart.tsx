'use client';

import { useMemo, useState } from 'react';
import {
  CartesianGrid,
  Line,
  ComposedChart,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
  Legend,
} from 'recharts';
import benchmarks from '@/content/data/benchmarks.json';

type Platform = 'cuda' | 'rocm';

/**
 * RooflineChart plots achieved GFLOP/s against arithmetic intensity on log-log
 * axes. The user can add their own kernel point; built-in points come from the
 * illustrative benchmark JSON.
 */
export function RooflineChart() {
  const [platform, setPlatform] = useState<Platform>('cuda');
  const [flops, setFlops] = useState(2_000_000_000);
  const [bytes, setBytes] = useState(1_000_000_000);
  const [achieved, setAchieved] = useState(1500);

  const cfg = benchmarks.rooflinePoints[platform];
  const ridge_point = cfg.peakGflops / cfg.peakBandwidthGBs;

  const roofline = useMemo(() => {
    const pts: { intensity: number; roof: number }[] = [];
    for (let e = -2; e <= 3; e += 0.1) {
      const intensity = Math.pow(10, e);
      const roof = Math.min(cfg.peakBandwidthGBs * intensity, cfg.peakGflops);
      pts.push({ intensity, roof });
    }
    return pts;
  }, [cfg]);

  const builtin_points = cfg.kernels.map((k) => ({ x: k.intensity, y: k.gflops, name: k.name }));
  const user_intensity = bytes > 0 ? flops / bytes : 0;
  const user_point = [{ x: user_intensity, y: achieved, name: '你的 kernel' }];

  return (
    <div className="my-6 rounded-lg border border-border bg-card/40 p-5">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <p className="text-base font-semibold text-foreground">Interactive Roofline</p>
        <div role="group" aria-label="平台切換" className="inline-flex overflow-hidden rounded-md border border-border text-sm">
          {(['cuda', 'rocm'] as Platform[]).map((p) => (
            <button key={p} onClick={() => setPlatform(p)} aria-pressed={platform === p} className={`px-3 py-1 transition ${platform === p ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:text-foreground'}`}>
              {p === 'cuda' ? 'CUDA' : 'ROCm'}
            </button>
          ))}
        </div>
      </div>
      <p className="mb-4 text-xs text-muted-foreground">
        peak {(cfg.peakGflops / 1000).toFixed(1)} TFLOP/s · BW {cfg.peakBandwidthGBs} GB/s · ridge point AI ≈ {ridge_point.toFixed(1)} FLOP/byte · <span className="text-[#ffa657]">示意數據</span>
      </p>

      <div className="h-80" aria-hidden>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart margin={{ top: 8, right: 12, bottom: 20, left: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
            <XAxis type="number" dataKey="x" name="AI" scale="log" domain={[0.01, 1000]} stroke="#8b949e" tick={{ fontSize: 10 }} tickFormatter={(v) => String(v)} label={{ value: 'arithmetic intensity (FLOP/byte)', position: 'insideBottom', offset: -10, fill: '#8b949e', fontSize: 11 }} allowDataOverflow />
            <YAxis type="number" dataKey="y" name="GFLOP/s" scale="log" domain={[10, cfg.peakGflops * 1.5]} stroke="#8b949e" tick={{ fontSize: 10 }} tickFormatter={(v) => String(v)} allowDataOverflow />
            <ZAxis range={[80, 80]} />
            <Tooltip contentStyle={{ background: '#161b22', border: '1px solid #30363d', color: '#fff' }} formatter={(v: number, n) => [typeof v === 'number' ? v.toFixed(1) : v, n]} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Line data={roofline} dataKey="roof" name="roofline" type="monotone" dot={false} stroke="#39d353" strokeWidth={2} isAnimationActive={false} xAxisId={0} />
            <Scatter data={builtin_points} name="內建 kernels" fill="#58a6ff" />
            <Scatter data={user_point} name="你的 kernel" fill="#f778ba" />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-4 grid gap-4 sm:grid-cols-3">
        <label className="text-sm text-muted-foreground">
          FLOPs:<span className="ml-1 font-mono text-primary">{flops.toExponential(1)}</span>
          <input type="range" min={7} max={12} step={0.1} value={Math.log10(flops)} onChange={(e) => setFlops(Math.round(Math.pow(10, Number(e.target.value))))} className="mt-1 w-full accent-brand" aria-label="FLOPs" />
        </label>
        <label className="text-sm text-muted-foreground">
          Bytes:<span className="ml-1 font-mono text-primary">{bytes.toExponential(1)}</span>
          <input type="range" min={6} max={11} step={0.1} value={Math.log10(bytes)} onChange={(e) => setBytes(Math.round(Math.pow(10, Number(e.target.value))))} className="mt-1 w-full accent-brand" aria-label="Bytes" />
        </label>
        <label className="text-sm text-muted-foreground">
          Achieved GFLOP/s:<span className="ml-1 font-mono text-primary">{achieved}</span>
          <input type="range" min={10} max={cfg.peakGflops} step={10} value={Math.min(achieved, cfg.peakGflops)} onChange={(e) => setAchieved(Number(e.target.value))} className="mt-1 w-full accent-brand" aria-label="achieved GFLOP/s" />
        </label>
      </div>
      <p className="mt-3 text-xs text-muted-foreground">
        你的 kernel arithmetic intensity = FLOPs / bytes = <span className="font-mono text-primary">{user_intensity.toFixed(2)}</span> FLOP/byte。若這個點落在斜線 (bandwidth roof) 上,代表 bandwidth-bound;落在水平線 (compute roof) 上則是 compute-bound。點與屋頂的垂直距離就是還能榨出的效能空間。
      </p>
    </div>
  );
}
