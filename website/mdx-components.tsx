import type { MDXComponents } from 'mdx/types';
import type { AnchorHTMLAttributes } from 'react';
import Link from 'next/link';
import { MemoryCoalescingVisualizer } from '@/components/widgets/MemoryCoalescingVisualizer';
import { OccupancyCalculator } from '@/components/widgets/OccupancyCalculator';
import { TilingAnimator } from '@/components/widgets/TilingAnimator';
import { RooflineChart } from '@/components/widgets/RooflineChart';
import { BenchmarkComparison } from '@/components/widgets/BenchmarkComparison';
import { Quiz } from '@/components/widgets/Quiz';
import { PlatformTabs, Platform } from '@/components/widgets/PlatformTabs';
import { Callout } from '@/components/widgets/Callout';
import { LabBox } from '@/components/widgets/LabBox';
import { FurtherReading } from '@/components/widgets/FurtherReading';
import { Exercise } from '@/components/widgets/Exercise';
import { Solution } from '@/components/widgets/Solution';
import { DataLayoutVisualizer } from '@/components/widgets/DataLayoutVisualizer';
import { GatherScatterVisualizer } from '@/components/widgets/GatherScatterVisualizer';
import { CollectiveAnimator } from '@/components/widgets/CollectiveAnimator';
import { DataRaceTimeline } from '@/components/widgets/DataRaceTimeline';
import { ProducerConsumerDiagram, ReadersWritersDiagram, DiningPhilosophersDiagram } from '@/components/widgets/ClassicPatternDiagrams';
import {
  GemmReuseFigure,
  HierarchicalTilingFigure,
  OuterProductFigure,
  DoubleBufferingFigure,
  TensorCoreFragmentFigure,
  SplitKFigure,
  StreamKFigure,
  WaveQuantizationFigure,
  CtaSwizzleFigure,
  EpilogueFigure,
} from '@/components/widgets/GemmFigures';
import {
  PipelineConceptFigure,
  StreamOverlapFigure,
  PipelineParallelFigure,
  OverlapSchedulerFigure,
  WarpSpecializationFigure,
  VendorPipelineFigure,
} from '@/components/widgets/PipelineFigures';
import {
  ReleaseAcquireFigure,
  CudaFenceScopeFigure,
  CrossBlockFenceProtocolFigure,
} from '@/components/widgets/MemoryFenceFigures';

// Route internal markdown links through next/link so the static-export
// basePath is applied; external links open in a new tab.
function MdxAnchor({ href = '', children, ...rest }: AnchorHTMLAttributes<HTMLAnchorElement>) {
  if (href.startsWith('/')) {
    return (
      <Link href={href} {...rest}>
        {children}
      </Link>
    );
  }
  const isExternal = href.startsWith('http');
  return (
    <a href={href} {...(isExternal ? { target: '_blank', rel: 'noreferrer' } : {})} {...rest}>
      {children}
    </a>
  );
}

export function useMDXComponents(components: MDXComponents): MDXComponents {
  return {
    a: MdxAnchor,
    MemoryCoalescingVisualizer,
    OccupancyCalculator,
    TilingAnimator,
    RooflineChart,
    BenchmarkComparison,
    Quiz,
    PlatformTabs,
    Platform,
    Callout,
    LabBox,
    FurtherReading,
    Exercise,
    Solution,
    DataLayoutVisualizer,
    GatherScatterVisualizer,
    CollectiveAnimator,
    DataRaceTimeline,
    ProducerConsumerDiagram,
    ReadersWritersDiagram,
    DiningPhilosophersDiagram,
    GemmReuseFigure,
    HierarchicalTilingFigure,
    OuterProductFigure,
    DoubleBufferingFigure,
    TensorCoreFragmentFigure,
    SplitKFigure,
    StreamKFigure,
    WaveQuantizationFigure,
    CtaSwizzleFigure,
    EpilogueFigure,
    PipelineConceptFigure,
    StreamOverlapFigure,
    PipelineParallelFigure,
    OverlapSchedulerFigure,
    WarpSpecializationFigure,
    VendorPipelineFigure,
    ReleaseAcquireFigure,
    CudaFenceScopeFigure,
    CrossBlockFenceProtocolFigure,
    ...components,
  };
}
