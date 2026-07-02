'use client';

import type { ElementType, ReactNode } from 'react';
import { useSettings } from '@/components/SettingsProvider';
import type { ContentWidth } from '@/lib/settings';

const widthClass: Record<ContentWidth, string> = {
  standard: 'max-w-6xl',
  wide: 'max-w-[110rem]',
  full: 'max-w-none',
};

/** Wraps page sections so the "內容寬度" setting can resize them consistently. */
export function AppWidthContainer({
  as: Component = 'div',
  className = '',
  children,
}: {
  as?: ElementType;
  className?: string;
  children: ReactNode;
}) {
  const { contentWidth, mounted } = useSettings();
  const resolved = mounted ? contentWidth : 'standard';
  return <Component className={`mx-auto ${widthClass[resolved]} ${className}`}>{children}</Component>;
}
