import type { ComponentType } from 'react';

// Static import map so Next.js can statically export every chapter page.
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
import C13 from './c13-small-matrices-tail-effects.mdx';
import C14 from './c14-warp-specialization-pipelines.mdx';
import C15 from './c15-kernel-library-ecosystem.mdx';
import C16 from './c16-profiling-deep-dive.mdx';
import D17 from './d17-finding-kernels-to-optimize.mdx';
import D18 from './d18-first-custom-extension.mdx';
import D19 from './d19-fused-elementwise-reduction.mdx';
import D20 from './d20-triton-intro.mdx';
import D21 from './d21-integrating-into-training.mdx';

export const chapterComponents: Record<string, ComponentType> = {
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
  'c13-small-matrices-tail-effects': C13,
  'c14-warp-specialization-pipelines': C14,
  'c15-kernel-library-ecosystem': C15,
  'c16-profiling-deep-dive': C16,
  'd17-finding-kernels-to-optimize': D17,
  'd18-first-custom-extension': D18,
  'd19-fused-elementwise-reduction': D19,
  'd20-triton-intro': D20,
  'd21-integrating-into-training': D21,
};
