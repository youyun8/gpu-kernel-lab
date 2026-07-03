import type { ComponentType } from 'react';

// Static import map so Next.js can statically export every chapter page.
import P0 from './p0-parallelization-design.mdx';
import P1 from './p1-cpu-threading-primitives.mdx';
import P2 from './p2-data-races-locks-and-atomics.mdx';
import P3 from './p3-gpu-synchronization-and-debugging.mdx';
import M0 from './m0-memory-hierarchy-model.mdx';
import M1 from './m1-cycle-and-bandwidth-calculus.mdx';
import M2 from './m2-locality-tiling-coalescing.mdx';
import M3 from './m3-memory-optimization-workflow.mdx';
import A1 from './a1-what-is-a-gpu.mdx';
import A2 from './a2-first-kernel.mdx';
import A3 from './a3-memory-hierarchy.mdx';
import A4 from './a4-measuring-performance.mdx';
import A5 from './a5-roofline-model.mdx';
import B6 from './b6-memory-coalescing.mdx';
import B7 from './b7-shared-memory-bank-conflicts.mdx';
import B8 from './b8-occupancy-latency-hiding.mdx';
import B9 from './b9-warp-level-programming.mdx';
import B10 from './b10-reduction-softmax.mdx';
import B11 from './b11-instruction-level-optimization.mdx';
import C12 from './c12-gemm-optimization.mdx';
import GM1 from './gm1-gemm-reuse-and-roofline.mdx';
import GM2 from './gm2-hierarchical-tiling.mdx';
import GM3 from './gm3-memory-pipeline.mdx';
import GM4 from './gm4-tensor-core-mfma.mdx';
import GM5 from './gm5-split-k-stream-k.mdx';
import GM6 from './gm6-cta-scheduling-tail.mdx';
import GM7 from './gm7-epilogue-lowprecision-autotuning.mdx';
import C13 from './c13-small-matrices-tail-effects.mdx';
import C14 from './c14-warp-specialization-pipelines.mdx';
import C15 from './c15-kernel-library-ecosystem.mdx';
import C16 from './c16-profiling-deep-dive.mdx';
import D17 from './d17-finding-kernels-to-optimize.mdx';
import D18 from './d18-first-custom-extension.mdx';
import D19 from './d19-fused-elementwise-reduction.mdx';
import D20 from './d20-triton-intro.mdx';
import D21 from './d21-integrating-into-training.mdx';
import E22 from './e22-cutlass-deep-dive.mdx';
import E23 from './e23-amd-kernel-ecosystem.mdx';
import E24 from './e24-attention-kernels.mdx';
import E25 from './e25-inference-engines.mdx';
import F26 from './f26-atomics-histograms-irregular-access.mdx';
import F27 from './f27-persistent-kernels-work-queues.mdx';
import F28 from './f28-streams-graphs-host-device-pipeline.mdx';
import F29 from './f29-low-precision-quantization-layouts.mdx';
import F30 from './f30-autotuning-specialization-codegen.mdx';
import F31 from './f31-correctness-determinism-debugging.mdx';
import F32 from './f32-optimization-checklist.mdx';
import G33 from './g33-nccl-rccl-collectives.mdx';

export const chapterComponents: Record<string, ComponentType> = {
  'p0-parallelization-design': P0,
  'p1-cpu-threading-primitives': P1,
  'p2-data-races-locks-and-atomics': P2,
  'p3-gpu-synchronization-and-debugging': P3,
  'm0-memory-hierarchy-model': M0,
  'm1-cycle-and-bandwidth-calculus': M1,
  'm2-locality-tiling-coalescing': M2,
  'm3-memory-optimization-workflow': M3,
  'a1-what-is-a-gpu': A1,
  'a2-first-kernel': A2,
  'a3-memory-hierarchy': A3,
  'a4-measuring-performance': A4,
  'a5-roofline-model': A5,
  'b6-memory-coalescing': B6,
  'b7-shared-memory-bank-conflicts': B7,
  'b8-occupancy-latency-hiding': B8,
  'b9-warp-level-programming': B9,
  'b10-reduction-softmax': B10,
  'b11-instruction-level-optimization': B11,
  'c12-gemm-optimization': C12,
  'gm1-gemm-reuse-and-roofline': GM1,
  'gm2-hierarchical-tiling': GM2,
  'gm3-memory-pipeline': GM3,
  'gm4-tensor-core-mfma': GM4,
  'gm5-split-k-stream-k': GM5,
  'gm6-cta-scheduling-tail': GM6,
  'gm7-epilogue-lowprecision-autotuning': GM7,
  'c13-small-matrices-tail-effects': C13,
  'c14-warp-specialization-pipelines': C14,
  'c15-kernel-library-ecosystem': C15,
  'c16-profiling-deep-dive': C16,
  'd17-finding-kernels-to-optimize': D17,
  'd18-first-custom-extension': D18,
  'd19-fused-elementwise-reduction': D19,
  'd20-triton-intro': D20,
  'd21-integrating-into-training': D21,
  'e22-cutlass-deep-dive': E22,
  'e23-amd-kernel-ecosystem': E23,
  'e24-attention-kernels': E24,
  'e25-inference-engines': E25,
  'f26-atomics-histograms-irregular-access': F26,
  'f27-persistent-kernels-work-queues': F27,
  'f28-streams-graphs-host-device-pipeline': F28,
  'f29-low-precision-quantization-layouts': F29,
  'f30-autotuning-specialization-codegen': F30,
  'f31-correctness-determinism-debugging': F31,
  'f32-optimization-checklist': F32,
  'g33-nccl-rccl-collectives': G33,
};
