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
    id: 'p-parallelization',
    label: 'Track 1 — 平行化基礎',
    level: 'Foundations',
    color: '#d29922',
    description:
      '從工作切割開始, 用 C++ std::thread / OpenMP / Pthread 建立同步觀念, 再對照 CUDA/ROCm 的 blocks、warps、atomics、barriers 與 fences。',
    chapters: [
      {
        slug: 'p0-parallelization-design',
        num: 0,
        title: '平行化設計: 從工作切割開始',
        summary:
          'Decomposition、data/task parallelism、granularity、load balancing、critical path, 以及 CPU thread/core vs GPU block/warp/thread 的硬體 mapping。',
        lab: 'kernels/00-parallel-foundations',
      },
      {
        slug: 'p1-cpu-threading-primitives',
        num: 1,
        title: 'CPU 平行化: C++ / OpenMP / Pthread Primitives',
        summary:
          'std::thread、pthread_create、OpenMP parallel for、mutex、semaphore、condition variable、barrier、atomic 與 memory_order。',
        lab: 'kernels/00-parallel-foundations',
      },
      {
        slug: 'p2-data-races-locks-and-atomics',
        num: 2,
        title: 'Data Race、Locks、Atomics 與 Reduction',
        summary:
          'Data race 與 lost update; mutex/critical、std::atomic/device atomic、OpenMP reduction、GPU block reduction、barrier/fence 的雙邊對照。',
        lab: 'kernels/00-parallel-foundations',
      },
      {
        slug: 'p3-gpu-synchronization-and-debugging',
        num: 3,
        title: 'GPU 平行化: CUDA / ROCm Synchronization',
        summary:
          'GPU 為何少用 lock/semaphore、device atomics、__syncthreads、cooperative groups、__threadfence、Compute Sanitizer 與 ROCm debug workflow。',
        lab: 'kernels/00-parallel-foundations',
      },
    ],
  },
  {
    id: 'm-memory-hierarchy',
    label: 'Track 2 — Memory Hierarchy',
    level: 'Foundations',
    color: '#6e7681',
    description:
      '從 DRAM/SRAM/cache/register 的硬體模型出發, 學會 cycle、bandwidth、arithmetic intensity 的紙上估算, 再落到 coalescing、tiling、bank conflict 與 optimization workflow。',
    chapters: [
      {
        slug: 'm0-memory-hierarchy-model',
        num: 0,
        title: 'Memory Hierarchy: DRAM、Cache、SRAM',
        summary:
          'Register、shared memory/LDS、L1/L2 cache、DRAM/HBM 的 latency/capacity trade-off, temporal/spatial locality 與 working set。',
        lab: 'kernels/02-memory',
      },
      {
        slug: 'm1-cycle-and-bandwidth-calculus',
        num: 1,
        title: 'Cycle 與 Bandwidth 的紙上計算',
        summary:
          'Latency cycles、bytes moved、achieved bandwidth、arithmetic intensity、roofline ridge point、transaction/cache-line counting 與 latency hiding 粗估。',
        lab: 'kernels/02-memory',
      },
      {
        slug: 'm2-locality-tiling-coalescing',
        num: 2,
        title: 'Locality、Tiling、Coalescing',
        summary:
          'Coalesced access、stride waste、shared memory/LDS tiling、bank conflict padding、AoS/SoA layout 與 CPU cache blocking。',
        lab: 'kernels/02-memory',
      },
      {
        slug: 'm3-memory-optimization-workflow',
        num: 3,
        title: 'Memory Optimization Workflow',
        summary:
          '用 bytes/FLOP/counters 分類 bottleneck, 選擇 layout、tiling、vectorization、privatization 等 transformation, 並驗證 side effects。',
        lab: 'kernels/02-memory',
      },
    ],
  },
  {
    id: 'a-basics',
    label: 'Track 3 — 入門',
    level: 'Beginner',
    color: '#39d353',
    description: '建立 GPU 的心智模型: 從硬體設計哲學到第一個能跑的 kernel, 以及如何正確測量效能。',
    chapters: [
      {
        slug: 'a1-what-is-a-gpu',
        num: 1,
        title: 'GPU 是什麼樣的機器?',
        summary: 'CPU vs. GPU 設計哲學、throughput vs. latency、SIMT 執行模型, 以及 CUDA↔ROCm 生態系對照。',
        lab: 'kernels/01-basics',
      },
      {
        slug: 'a2-first-kernel',
        num: 2,
        title: '第一個 Kernel',
        summary: 'thread/block/grid hierarchy、kernel launch 語法、vector add, 以及 nvcc/hipcc compile 流程。',
        lab: 'kernels/01-basics',
      },
      {
        slug: 'a3-memory-hierarchy',
        num: 3,
        title: 'Memory Hierarchy 全景圖',
        summary: 'registers、shared memory/LDS、L1/L2 cache、global memory/HBM 的 latency 與 bandwidth 數量級。',
        lab: 'kernels/02-memory',
      },
      {
        slug: 'a4-measuring-performance',
        num: 4,
        title: '如何測量效能',
        summary: 'events、warmup、多次 iteration、achieved bandwidth 計算、speed-of-light 分析, 以及第一次使用 profiler。',
        lab: 'kernels/02-memory',
      },
      {
        slug: 'a5-roofline-model',
        num: 5,
        title: 'Roofline Model',
        summary: 'arithmetic intensity、bandwidth-bound vs. compute-bound, 以及如何判斷 kernel 的優化天花板。',
        lab: 'kernels/02-memory',
      },
    ],
  },
  {
    id: 'b-intermediate',
    label: 'Track 4 — 進階',
    level: 'Intermediate',
    color: '#58a6ff',
    description: '深入 memory 與 execution model: coalescing、bank conflicts、occupancy、warp-level programming 與 reduction。',
    chapters: [
      {
        slug: 'b6-memory-coalescing',
        num: 6,
        title: 'Memory Coalescing',
        summary: 'access patterns、AoS vs. SoA、misaligned access 的代價, 並用不同 stride 量測 bandwidth 曲線。',
        lab: 'kernels/02-memory',
      },
      {
        slug: 'b7-shared-memory-bank-conflicts',
        num: 7,
        title: 'Shared Memory 與 Bank Conflicts',
        summary: 'tiling 動機、bank 結構 (NVIDIA 32 banks vs. AMD LDS)、padding 技巧, matrix transpose 三部曲。',
        lab: 'kernels/02-memory',
      },
      {
        slug: 'b8-occupancy-latency-hiding',
        num: 8,
        title: 'Occupancy 與 Latency Hiding',
        summary: 'registers/shared memory 對 occupancy 的影響、__launch_bounds__, 以及 occupancy vs. ILP 的 trade-off。',
        lab: 'kernels/01-basics',
      },
      {
        slug: 'b9-warp-level-programming',
        num: 9,
        title: 'Warp-Level Programming',
        summary: 'shuffle instructions、warp reduce、ballot/vote, 以及 warp (32) 與 wavefront (64/32) 的可攜性陷阱。',
        lab: 'kernels/04-reductions-softmax',
      },
      {
        slug: 'b10-reduction-softmax',
        num: 10,
        title: 'Reduction 與 Softmax 優化系列',
        summary: '從 naive reduction 優化到 warp shuffle + vectorized loads; online softmax 與 numerical stability。',
        lab: 'kernels/04-reductions-softmax',
      },
      {
        slug: 'b11-instruction-level-optimization',
        num: 11,
        title: 'Instruction-Level 優化',
        summary: 'vectorized access (float4)、loop unrolling、FMA、fast math 取捨、#pragma unroll。',
        lab: 'kernels/02-memory',
      },
    ],
  },
  {
    id: 'c-expert',
    label: 'Track 5 — 專家',
    level: 'Expert',
    color: '#f778ba',
    description: '貫穿式 GEMM 優化、小矩陣尾端效應、warp specialization pipeline, 以及 profiling 的系統化 workflow。',
    chapters: [
      {
        slug: 'c12-gemm-optimization',
        num: 12,
        title: 'GEMM 優化全流程',
        summary: '從 naive 到接近 library: shared memory tiling、register tiling、vectorized loads、double buffering、Tensor Core/MFMA。',
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
        title: 'Warp Specialization 與 Producer–Consumer Pipelines',
        summary: '現代 GEMM/attention kernel 結構、非同步複製 (cp.async/TMA 概念與 AMD 對應機制)。',
        lab: 'kernels/05-advanced-scheduling',
      },
      {
        slug: 'c15-kernel-library-ecosystem',
        num: 15,
        title: '認識 Kernel Library 生態',
        summary: 'CUTLASS 分層設計、Composable Kernel、Triton 定位, 以及自己寫 vs. autotune library 的取捨。',
        lab: 'kernels/03-gemm',
      },
      {
        slug: 'c16-profiling-deep-dive',
        num: 16,
        title: 'Profiling 深入',
        summary: 'Nsight Compute metrics 導讀、omniperf 對應功能, 以及從 metrics 回推 bottleneck 的 systematic workflow。',
        lab: 'kernels/05-advanced-scheduling',
      },
    ],
  },
  {
    id: 'gemm-deep-dive',
    label: 'Track 6 — GEMM 優化深入',
    level: 'Expert',
    color: '#39c5cf',
    description:
      '把 GEMM 拆成獨立技巧, 每個一節: reuse/roofline、block/warp/thread tiling、vectorized load 與 double buffering/cp.async pipeline、Tensor Core/MFMA、Split-K 與 Stream-K、CTA swizzle 與 persistent kernel, 以及 epilogue fusion/量化/autotuning。 每節都附視覺化圖解。',
    chapters: [
      {
        slug: 'gm1-gemm-reuse-and-roofline',
        num: 1,
        title: 'GEMM 為什麼難: Reuse 與 Arithmetic Intensity',
        summary:
          'naive kernel 為何卡在 bandwidth roof、reuse 是唯一解藥、arithmetic intensity 與 roofline 框架, 以及整條優化階梯的全景。',
        lab: 'kernels/03-gemm',
      },
      {
        slug: 'gm2-hierarchical-tiling',
        num: 2,
        title: 'Tiling 階層: Block、Warp、Thread Tiling',
        summary:
          '同一塊 C 的三層切分: shared memory block tiling、register/thread tiling 的 outer product, 以及對齊 MMA 的 warp tiling。',
        lab: 'kernels/03-gemm',
      },
      {
        slug: 'gm3-memory-pipeline',
        num: 3,
        title: 'Memory Pipeline: Vectorized Load、Double Buffering、Async Copy',
        summary:
          'float4 向量化載入、double buffering software pipelining、cp.async/TMA 多 stage 非同步複製, 以及 swizzle 消除 bank conflict。',
        lab: 'kernels/03-gemm',
      },
      {
        slug: 'gm4-tensor-core-mfma',
        num: 4,
        title: 'Tensor Core / MFMA GEMM',
        summary:
          'warp 共同持有 fragment 的 MMA 抽象、硬體規定的 layout 與 swizzle 要求, 以及 fp16/bf16/tf32/fp8/int8 各精度路徑。',
        lab: 'kernels/03-gemm',
      },
      {
        slug: 'gm5-split-k-stream-k',
        num: 5,
        title: 'K 維平行化: Split-K 與 Stream-K',
        summary:
          'wave quantization 與尾端效應的動機、Split-K 沿 K 切段加 reduction、Stream-K 的 work-centric 均分, 以及 atomic vs 確定性 reduction。',
        lab: 'kernels/05-advanced-scheduling',
      },
      {
        slug: 'gm6-cta-scheduling-tail',
        num: 6,
        title: 'CTA Scheduling: Swizzle 與 Persistent Kernel',
        summary:
          'CTA swizzle/rasterization 提升 L2 reuse、persistent kernel 的 work queue 與動態負載平衡, 以及三種 scheduling 工具的取捨。',
        lab: 'kernels/05-advanced-scheduling',
      },
      {
        slug: 'gm7-epilogue-lowprecision-autotuning',
        num: 7,
        title: 'Epilogue、Low Precision 與 Autotuning',
        summary:
          'epilogue fusion 省尾端 traffic、量化 GEMM 的 scale/packing/dequant epilogue, 以及 template specialization 與 autotuning 收束所有旋鈕。',
        lab: 'kernels/03-gemm',
      },
    ],
  },
  {
    id: 'd-pytorch',
    label: 'Track 7 — PyTorch 實戰',
    level: 'Practical',
    color: '#ffa657',
    description: '把前面所學帶進 PyTorch: profiler 找瓶頸、custom extension、autograd、fused kernel、Triton 與 torch.compile 整合。',
    chapters: [
      {
        slug: 'd17-finding-kernels-to-optimize',
        num: 17,
        title: '在 PyTorch 中找到值得優化的 Kernel',
        summary: 'torch.profiler 與 trace 分析、辨識 launch overhead 與 memory-bound ops、什麼樣的 op 值得 fuse。',
        lab: 'kernels/06-pytorch-integration',
      },
      {
        slug: 'd18-first-custom-extension',
        num: 18,
        title: '第一個 Custom CUDA/HIP Extension',
        summary: 'cpp_extension 與 load_inline 流程、tensor 正確處理 (contiguity/dtype/device/stream)、autograd Function。',
        lab: 'kernels/06-pytorch-integration/cpp_extension',
      },
      {
        slug: 'd19-fused-elementwise-reduction',
        num: 19,
        title: '案例研究: Fused Elementwise + Reduction',
        summary: '把 bias + activation + reduction 換成 fused kernel、量測 end-to-end speedup, 並與 torch.compile 比較。',
        lab: 'kernels/06-pytorch-integration/load_inline',
      },
      {
        slug: 'd20-triton-intro',
        num: 20,
        title: 'Triton 入門與比較',
        summary: '用 Triton 重寫 softmax 與 GEMM, 討論 productivity vs. peak performance, 以及 CUDA/ROCm 可攜性。',
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
  {
    id: 'e-libraries',
    label: 'Track 8 — 生產級 library',
    level: 'Production',
    color: '#a371f7',
    description:
      '把地圖走到底: CUTLASS/CuTe 的分層設計、AMD 的 CK/hipBLASLt/AITER、FlashAttention 系列與 serving 場景的 attention, 以及 vLLM/SGLang/TensorRT-LLM 如何編排這一切。',
    chapters: [
      {
        slug: 'e22-cutlass-deep-dive',
        num: 22,
        title: 'CUTLASS 深入: 分層設計與 CuTe',
        summary:
          'device→kernel→threadblock→warp→MMA 的分層、CuTe 的 layout 代數、CUTLASS Profiler, 以及 FlashAttention-3/PyTorch fp8 GEMM 如何建立其上。',
        lab: 'kernels/03-gemm',
      },
      {
        slug: 'e23-amd-kernel-ecosystem',
        num: 23,
        title: 'AMD Kernel 生態: CK、hipBLASLt 與 AITER',
        summary:
          'Composable Kernel 對應 CUTLASS、Tensile 生成 hipBLASLt kernel、rocWMMA/MFMA, 以及 AITER 如何為 vLLM/SGLang 在 MI300X 上加速。',
        lab: 'kernels/03-gemm',
      },
      {
        slug: 'e24-attention-kernels',
        num: 24,
        title: 'Attention Kernel: FlashAttention 與 FlashInfer',
        summary:
          'naive attention 的 O(N²) 之痛、FlashAttention 的 tiling + online softmax、版本演進, 以及 serving 的 PagedAttention 與 FlashInfer。',
        lab: 'kernels/04-reductions-softmax',
      },
      {
        slug: 'e25-inference-engines',
        num: 25,
        title: '推論引擎的 Kernel 棧: vLLM、SGLang 與 TensorRT-LLM',
        summary:
          'prefill vs. decode 的效能特性、continuous batching, 以及三大推論引擎如何編排 CUTLASS/FlashInfer/AITER/Triton kernel。',
        lab: 'kernels/06-pytorch-integration',
      },
    ],
  },
  {
    id: 'f-production',
    label: 'Track 9 — Production Tactics',
    level: 'Production',
    color: '#db6d28',
    description:
      '補齊 production kernel 常見戰場: atomics、persistent scheduling、host-device pipeline、低精度、autotuning、correctness 與整體優化 checklist。',
    chapters: [
      {
        slug: 'f26-atomics-histograms-irregular-access',
        num: 26,
        title: 'Atomics、Histograms 與 Irregular Access',
        summary:
          'Atomic contention、warp aggregation、block-private histograms、scatter/gather trade-off 與 memory ordering。',
        lab: 'kernels/04-reductions-softmax',
      },
      {
        slug: 'f27-persistent-kernels-work-queues',
        num: 27,
        title: 'Persistent Kernels 與 Work Queues',
        summary:
          '常駐 CTA、dynamic work queues、chunking、work stealing、persistent GEMM scheduler 與 tail effect 對策。',
        lab: 'kernels/05-advanced-scheduling',
      },
      {
        slug: 'f28-streams-graphs-host-device-pipeline',
        num: 28,
        title: 'Streams、Graphs 與 Host-Device Pipeline',
        summary:
          'Pinned memory、async copy、多 stream double buffering、CUDA/HIP Graphs、隱性同步與 communication overlap。',
        lab: 'kernels/05-advanced-scheduling',
      },
      {
        slug: 'f29-low-precision-quantization-layouts',
        num: 29,
        title: 'Low-Precision、Quantization 與 Layout Conversion',
        summary:
          'fp16/bf16/tf32/fp8/int8/int4、scale 粒度、packing、prepack、epilogue dequant/requant 與 validation。',
        lab: 'kernels/03-gemm',
      },
      {
        slug: 'f30-autotuning-specialization-codegen',
        num: 30,
        title: 'Autotuning、Specialization 與 Code Generation',
        summary:
          'Tile knobs、template specialization、search pruning、cost model、dispatch table、tuning cache 與 codegen 取捨。',
        lab: 'kernels/03-gemm',
      },
      {
        slug: 'f31-correctness-determinism-debugging',
        num: 31,
        title: 'Correctness、Determinism 與 Debugging',
        summary:
          'Reference、tail shape、sanitizer、race detection、同步 bug、deterministic reduction、tolerance 與 CI。',
        lab: 'kernels/exercises',
      },
      {
        slug: 'f32-optimization-checklist',
        num: 32,
        title: 'GPU Kernel Optimization Checklist',
        summary:
          '從 end-to-end 量測、bottleneck 分類、memory/execution/algorithm 技巧到 production 化的一張總表。',
        lab: 'kernels',
      },
    ],
  },
  {
    id: 'g-multi-gpu',
    label: 'Track 10 — Multi-GPU / Collectives',
    level: 'Production',
    color: '#2f81f7',
    description:
      '單卡塞不下就得跨卡: collective primitives、ring/tree 演算法、algbw vs busbw、NVLink/xGMI 拓撲與 compute-communication overlap。',
    chapters: [
      {
        slug: 'g33-nccl-rccl-collectives',
        num: 33,
        title: 'NCCL / RCCL 與 Multi-GPU Collectives',
        summary:
          'All-reduce = reduce-scatter + all-gather、ring vs tree、busbw 推導、NVLink/NVSwitch vs xGMI、NCCL↔RCCL 可攜性與 overlap。',
        lab: 'kernels/07-multi-gpu-collectives',
      },
    ],
  },
  {
    id: 'sp-software-pipelining',
    label: 'Track 11 — Software Pipelining',
    level: 'Advanced',
    color: '#e3b341',
    description:
      '同一個「重疊」idea 貫穿六個層級: kernel 內的 double buffering、host–device stream/graph、PyTorch data prefetch、跨 GPU 的 pipeline parallelism (GPipe/1F1B), 到 SGLang 推論服務的 overlap scheduler。 每章附視覺化 schedule 圖、可執行的 programming 範例與逐節 paper-and-pencil 練習。',
    chapters: [
      {
        slug: 'sp1-what-is-software-pipelining',
        num: 1,
        title: 'Software Pipelining 是什麼',
        summary:
          '從序列到 pipeline、fill/steady-state/drain、bubble 佔比 (p−1)/(N+k−1)、stage 平衡與依賴 hazard, 以及 GPU 各層級的 pipeline 全景。',
        lab: 'kernels/05-advanced-scheduling',
      },
      {
        slug: 'sp2-kernel-level-pipelining',
        num: 2,
        title: 'Kernel 內的 Pipelining',
        summary:
          'Double buffering 回顧、多 stage cp.async pipeline、warp specialization (producer/consumer) 與 shared-memory ring buffer。',
        lab: 'kernels/03-gemm',
      },
      {
        slug: 'sp3-host-device-streams-graphs',
        num: 3,
        title: 'Host–Device Pipeline: Streams、Events、Graphs',
        summary:
          'CUDA/HIP stream 疊 copy 與 compute、pinned memory + non_blocking、event 建立跨 stream 依賴, 以及 CUDA Graph 消 launch overhead。',
        lab: 'kernels/05-advanced-scheduling',
      },
      {
        slug: 'sp4-pytorch-input-prefetch-pipeline',
        num: 4,
        title: 'PyTorch 的 Input & Prefetch Pipeline',
        summary:
          'DataLoader 的 workers/prefetch_factor/pin_memory、CUDA prefetcher 與 record_stream, 以及 torch.compile reduce-overhead 消 launch overhead。',
        lab: 'kernels/06-pytorch-integration',
      },
      {
        slug: 'sp5-pipeline-parallelism',
        num: 5,
        title: 'Pipeline Parallelism',
        summary:
          'DP/TP/PP 選型、GPipe 的 micro-batch 與 bubble、1F1B 省 activation memory (非 bubble)、interleaved 1F1B, 以及 torch.distributed.pipelining。',
        lab: 'kernels/07-multi-gpu-collectives',
      },
      {
        slug: 'sp6-serving-overlap-sglang',
        num: 6,
        title: '推論服務的 Pipelining: SGLang、vLLM',
        summary:
          'Continuous batching、SGLang overlap (zero-overhead) scheduler、chunked prefill 與 prefill/decode disaggregation。',
        lab: 'kernels/06-pytorch-integration',
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

const repoBase = 'https://github.com/youyun8/gpu-kernel-lab/tree/main/';

export function labUrl(lab: string): string {
  return `${repoBase}${lab}`;
}

export interface ExerciseSetMeta {
  slug: string;
  trackId: string;
  trackLabel: string;
  trackColor: string;
  title: string;
  summary: string;
  count: number;
}

// One exercise set per track. Slugs map to MDX files in content/exercises/.
export const exerciseSets: ExerciseSetMeta[] = [
  {
    slug: 'track-p',
    trackId: 'p-parallelization',
    trackLabel: 'Track 1 — 平行化基礎',
    trackColor: '#d29922',
    title: 'Track 1 練習: Parallelization Foundations',
    summary:
      'Decomposition、lost update interleaving、mutex/atomic/reduction/semaphore 選型、barrier 正確性、Amdahl/granularity/privatization, 以及 CPU/GPU race demos。',
    count: 11,
  },
  {
    slug: 'track-m',
    trackId: 'm-memory-hierarchy',
    trackLabel: 'Track 2 — Memory Hierarchy',
    trackColor: '#6e7681',
    title: 'Track 2 練習: Memory Hierarchy',
    summary:
      'Latency cycle 換算、bandwidth/roofline 計算、cache-line/transaction counting、tile 容量估算、latency hiding、stencil AI, 以及 stride/transpose/roofline programming labs。',
    count: 13,
  },
  {
    slug: 'track-a',
    trackId: 'a-basics',
    trackLabel: 'Track 3 — 入門',
    trackColor: '#39d353',
    title: 'Track 3 練習: GPU 基礎與效能測量',
    summary: 'SIMT、thread indexing、memory hierarchy latency、warp divergence、bandwidth 計算、roofline 判讀。',
    count: 11,
  },
  {
    slug: 'track-b',
    trackId: 'b-intermediate',
    trackLabel: 'Track 4 — 進階',
    trackColor: '#58a6ff',
    title: 'Track 4 練習: Coalescing、Bank Conflict、Occupancy、Reduction',
    summary: 'transaction 計數、bank conflict/padding、occupancy limiter、warp reduce、online softmax。',
    count: 12,
  },
  {
    slug: 'track-c',
    trackId: 'c-expert',
    trackLabel: 'Track 5 — 專家',
    trackColor: '#f778ba',
    title: 'Track 5 練習: GEMM、Tail Effect、Pipeline、Profiling',
    summary: 'arithmetic intensity、tile reuse、wave quantization、Split-K/Stream-K、epilogue fusion、bottleneck 診斷。',
    count: 11,
  },
  {
    slug: 'track-d',
    trackId: 'd-pytorch',
    trackLabel: 'Track 7 — PyTorch 實戰',
    trackColor: '#ffa657',
    title: 'Track 7 練習: Profiling、Custom Extension、Fusion、Triton',
    summary: 'launch overhead 估算、tensor 檢查、fused speedup、autograd、custom op 註冊、torch.compile 診斷。',
    count: 10,
  },
  {
    slug: 'track-e',
    trackId: 'e-libraries',
    trackLabel: 'Track 8 — 生產級 library',
    trackColor: '#a371f7',
    title: 'Track 8 練習: CUTLASS、AMD 生態、Attention、推論引擎',
    summary: 'CUTLASS 分層與 CuTe layout、CK/hipBLASLt/AITER 對照、FlashAttention 推導、prefill/decode AI、continuous batching 判讀。',
    count: 11,
  },
  {
    slug: 'track-sp',
    trackId: 'sp-software-pipelining',
    trackLabel: 'Track 11 — Software Pipelining',
    trackColor: '#e3b341',
    title: 'Track 11 練習: Software Pipelining',
    summary:
      'Pipeline 步數/加速上限、bubble 佔比、瓶頸 stage、latency hiding stage 數、GPipe vs 1F1B 記憶體、overlap scheduler 與 chunked prefill, 以及 stream/prefetcher/pipeline programming labs。',
    count: 11,
  },
];

export function getExerciseSet(slug: string): ExerciseSetMeta | undefined {
  return exerciseSets.find((s) => s.slug === slug);
}
