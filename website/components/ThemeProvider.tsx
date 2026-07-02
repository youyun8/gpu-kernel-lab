'use client';

import { ThemeProvider as NextThemesProvider } from 'next-themes';
import type { ReactNode } from 'react';

/**
 * Wraps the app in next-themes so the `.dark` class (and thus the HSL design
 * tokens in globals.css) can follow the user's light / dark / system choice.
 * Light is the default, matching cp-handbook; `enableSystem` keeps a "follow
 * OS" option in the toggle cycle.
 */
export function ThemeProvider({ children }: { children: ReactNode }) {
  return (
    <NextThemesProvider
      attribute="class"
      defaultTheme="light"
      enableSystem
      disableTransitionOnChange
    >
      {children}
    </NextThemesProvider>
  );
}
