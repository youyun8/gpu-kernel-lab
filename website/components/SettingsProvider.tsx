'use client';

import { createContext, useContext, useEffect, useState, type ReactNode } from 'react';
import {
  defaultSettings,
  SETTINGS_STORAGE_KEY,
  type ContentWidth,
  type SettingsState,
  type TextSize,
} from '@/lib/settings';

interface SettingsContextValue extends SettingsState {
  /** False until localStorage has been read; consumers should fall back to defaults until then. */
  mounted: boolean;
  setContentWidth: (value: ContentWidth) => void;
  setTextSize: (value: TextSize) => void;
  setCodeWrap: (value: boolean) => void;
}

const SettingsContext = createContext<SettingsContextValue | null>(null);

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<SettingsState>(defaultSettings);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(SETTINGS_STORAGE_KEY);
      if (raw) setState((current) => ({ ...current, ...JSON.parse(raw) }));
    } catch {
      // Storage unavailable (private mode, etc.) — fall back to defaults.
    }
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!mounted) return;
    localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(state));
  }, [state, mounted]);

  useEffect(() => {
    const root = document.documentElement;
    if (state.textSize === 'standard') delete root.dataset.textSize;
    else root.dataset.textSize = state.textSize;
  }, [state.textSize]);

  useEffect(() => {
    const root = document.documentElement;
    if (state.codeWrap) delete root.dataset.codeWrap;
    else root.dataset.codeWrap = 'off';
  }, [state.codeWrap]);

  return (
    <SettingsContext.Provider
      value={{
        ...state,
        mounted,
        setContentWidth: (contentWidth) => setState((s) => ({ ...s, contentWidth })),
        setTextSize: (textSize) => setState((s) => ({ ...s, textSize })),
        setCodeWrap: (codeWrap) => setState((s) => ({ ...s, codeWrap })),
      }}
    >
      {children}
    </SettingsContext.Provider>
  );
}

export function useSettings(): SettingsContextValue {
  const ctx = useContext(SettingsContext);
  if (!ctx) throw new Error('useSettings must be used within SettingsProvider');
  return ctx;
}
