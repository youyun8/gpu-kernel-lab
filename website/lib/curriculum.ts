export interface ChapterMeta {
  slug: string;
  num: number;
  title: string;
  summary: string;
  lab: string; // relative path into kernels/
}

export interface TrackMeta {
  id: string;
  label: string;
  level: string;
  color: string;
  description: string;
  chapters: ChapterMeta[];
}

export const tracks: TrackMeta[] = [
  {
    id: 'a-basics',
    label: 'Track A — 入門',
    level: 'Beginner',
    color: '#39d353',
    description: '建立 GPU 的心智模型:從硬體設計哲學到第一個能跑的 kernel,以及如何正確測量效能。',
    chapters: [
      {
        slug: 'a1-what-is-a-gpu',
        num: 1,
        title: 'GPU 是什麼樣的機器?',
        summary: 'CPU vs. GPU 設計哲學、throughput vs. latency、SIMT 執行模型,以及 CUDA↔ROCm 生態系對照。',
        lab: 'kernels/01-basics',
      },
      {
        slug: 'a2-first-kernel',
        num: 2,
        title: '第一個 kernel',
        summary: 'thread/block/grid hierarchy、kernel launch 語法、vector add,以及 nvcc/hipcc compile 流程。',
        lab: 'kernels/01-basics',
      },
      {
        slug: 'a3-memory-hierarchy',
        num: 3,
        title: 'Memory hierarchy 全景圖',
        summary: 'registers、shared memory/LDS、L1/L2 cache、global memory/HBM 的 latency 與 bandwidth 數量級。',
        lab: 'kernels/02-memory',
      },
      {
        slug: 'a4-measuring-performance',
        num: 4,
        title: '如何測量效能',
        summary: 'events、warmup、多次 iteration、achieved bandwidth 計算、speed-of-light 分析,以及第一次使用 profiler。',
        lab: 'kernels/02-memory',
      },
      {
        slug: 'a5-roofline-model',
        num: 5,
        title: 'Roofline model',
        summary: 'arithmetic intensity、bandwidth-bound vs. compute-bound,以及如何判斷 kernel 的優化天花板。',
        lab: 'kernels/02-memory',
      },
    ],
  },
  {
    id: 'b-intermediate',
    label: 'Track B — 進階',
    level: 'Intermediate',
    color: '#58a6ff',
    description: '深入 memory 與 execution model:coalescing、bank conflicts、occupancy、warp-level programming 與 reduction。',
    chapters: [
      {
        slug: 'b6-memory-coalescing',
        num: 6,
        title: 'Memory coalescing',
        summary: 'access patterns、AoS vs. SoA、misaligned access 的代價,並用不同 stride 量測 bandwidth 曲線。',
        lab: 'kernels/02-memory',
      },
      {
        slug: 'b7-shared-memory-bank-conflicts',
        num: 7,
        title: 'Shared memory 與 bank conflicts',
        summary: 'tiling 動機、bank 結構 (NVIDIA 32 banks vs. AMD LDS)、padding 技巧,matrix transpose 三部曲。',
        lab: 'kernels/02-memory',
      },
      {
        slug: 'b8-occupancy-latency-hiding',
        num: 8,
        title: 'Occupancy 與 latency hiding',
        summary: 'registers/shared memory 對 occupancy 的影響、__launch_bounds__,以及 occupancy vs. ILP 的 trade-off。',
        lab: 'kernels/01-basics',
      },
      {
        slug: 'b9-warp-level-programming',
        num: 9,
        title: 'Warp-level programming',
        summary: 'shuffle instructions、warp reduce、ballot/vote,以及 warp (32) 與 wavefront (64/32) 的可攜性陷阱。',
        lab: 'kernels/04-reductions-softmax',
      },
      {
        slug: 'b10-reduction-softmax',
        num: 10,
        title: 'Reduction 與 softmax 優化系列',
        summary: '從 naive reduction 優化到 warp shuffle + vectorized loads;online softmax 與 numerical stability。',
        lab: 'kernels/04-reductions-softmax',
      },
      {
        slug: 'b11-instruction-level-optimization',
        num: 11,
        title: 'Instruction-level 優化',
        summary: 'vectorized access (float4)、loop unrolling、FMA、fast math 取捨、#pragma unroll。',
        lab: 'kernels/02-memory',
      },
    ],
  },
  {
    id: 'c-expert',
    label: 'Track C — 專家',
    level: 'Expert',
    color: '#f778ba',
    description: '貫穿式 GEMM 優化、小矩陣尾端效應、warp specialization pipeline,以及 profiling 的系統化 workflow。',
    chapters: [
      {
        slug: 'c12-gemm-optimization',
        num: 12,
        title: 'GEMM 優化全流程',
        summary: '從 naive 到接近 library:shared memory tiling、register tiling、vectorized loads、double buffering、Tensor Core/MFMA。',
        lab: 'kernels/03-gemm',
      },
      {
        slug: 'c13-small-matrices-tail-effects',
        num: 13,
        title: '小矩陣與尾端效應',
        summary: 'wave quantization、tail effect、Split-K 與 Stream-K、CTA swizzling 對 L2 locality 的影響。',
        lab: 'kernels/05-advanced-scheduling',
      },
      {
        slug: 'c14-warp-specialization-pipelines',
        num: 14,
        title: 'Warp specialization 與 producer–consumer pipelines',
        summary: '現代 GEMM/attention kernel 結構、非同步複製 (cp.async/TMA 概念與 AMD 對應機制)。',
        lab: 'kernels/05-advanced-scheduling',
      },
      {
        slug: 'c15-kernel-library-ecosystem',
        num: 15,
        title: '認識 kernel library 生態',
        summary: 'CUTLASS 分層設計、Composable Kernel、Triton 定位,以及自己寫 vs. autotune library 的取捨。',
        lab: 'kernels/03-gemm',
      },
      {
        slug: 'c16-profiling-deep-dive',
        num: 16,
        title: 'Profiling 深入',
        summary: 'Nsight Compute metrics 導讀、omniperf 對應功能,以及從 metrics 回推 bottleneck 的 systematic workflow。',
        lab: 'kernels/05-advanced-scheduling',
      },
    ],
  },
  {
    id: 'd-pytorch',
    label: 'Track D — PyTorch 實戰',
    level: 'Practical',
    color: '#ffa657',
    description: '把前面所學帶進 PyTorch:profiler 找瓶頸、custom extension、autograd、fused kernel、Triton 與 torch.compile 整合。',
    chapters: [
      {
        slug: 'd17-finding-kernels-to-optimize',
        num: 17,
        title: '在 PyTorch 中找到值得優化的 kernel',
        summary: 'torch.profiler 與 trace 分析、辨識 launch overhead 與 memory-bound ops、什麼樣的 op 值得 fuse。',
        lab: 'kernels/06-pytorch-integration',
      },
      {
        slug: 'd18-first-custom-extension',
        num: 18,
        title: '第一個 custom CUDA/HIP extension',
        summary: 'cpp_extension 與 load_inline 流程、tensor 正確處理 (contiguity/dtype/device/stream)、autograd Function。',
        lab: 'kernels/06-pytorch-integration/cpp_extension',
      },
      {
        slug: 'd19-fused-elementwise-reduction',
        num: 19,
        title: '案例研究:fused elementwise + reduction',
        summary: '把 bias + activation + reduction 換成 fused kernel、量測 end-to-end speedup,並與 torch.compile 比較。',
        lab: 'kernels/06-pytorch-integration/load_inline',
      },
      {
        slug: 'd20-triton-intro',
        num: 20,
        title: 'Triton 入門與比較',
        summary: '用 Triton 重寫 softmax 與 GEMM,討論 productivity vs. peak performance,以及 CUDA/ROCm 可攜性。',
        lab: 'kernels/06-pytorch-integration/triton',
      },
      {
        slug: 'd21-integrating-into-training',
        num: 21,
        title: '整合回訓練/推論流程',
        summary: 'torch.library custom op registration、與 torch.compile 共存、多 GPU 注意事項、regression benchmark。',
        lab: 'kernels/06-pytorch-integration/custom_op_autograd',
      },
    ],
  },
];

export interface FlatChapter extends ChapterMeta {
  trackId: string;
  trackLabel: string;
  trackColor: string;
}

export const flatChapters: FlatChapter[] = tracks.flatMap((track) =>
  track.chapters.map((chapter) => ({
    ...chapter,
    trackId: track.id,
    trackLabel: track.label,
    trackColor: track.color,
  })),
);

export function getChapterNav(slug: string): {
  current?: FlatChapter;
  prev?: FlatChapter;
  next?: FlatChapter;
} {
  const index = flatChapters.findIndex((chapter) => chapter.slug === slug);
  if (index === -1) return {};
  return {
    current: flatChapters[index],
    prev: index > 0 ? flatChapters[index - 1] : undefined,
    next: index < flatChapters.length - 1 ? flatChapters[index + 1] : undefined,
  };
}

const repoBase = 'https://github.com/your-org/gpu-kernel-lab/tree/main/';

export function labUrl(lab: string): string {
  return `${repoBase}${lab}`;
}
