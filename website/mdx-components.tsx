import type { MDXComponents } from 'mdx/types';
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

export function useMDXComponents(components: MDXComponents): MDXComponents {
  return {
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
    ...components,
  };
}
