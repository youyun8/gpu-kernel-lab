'use client';

import type { ElementType, ReactNode } from 'react';
import { useSettings } from '@/components/SettingsProvider';
import type { ContentWidth } from '@/lib/settings';

const kWidthClass: Record<ContentWidth, string> = {
  standard: 'max-w-6xl',
  wide: 'max-w-[110rem]',
  full: 'max-w-none',
};

/** Wraps page sections so the content-width setting can resize them consistently. */
export function AppWidthContainer({
  as: Component = 'div',
  className = '',
  children,
}: {
  as?: ElementType;
  className?: string;
  children: ReactNode;
}) {
  const { content_width, mounted } = useSettings();
  const resolved = mounted ? content_width : 'standard';
  return <Component className={`mx-auto ${kWidthClass[resolved]} ${className}`}>{children}</Component>;
}
