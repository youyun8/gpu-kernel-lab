'use client';

import { createContext, useContext, useEffect, useState, type ReactNode } from 'react';
import {
  kDefaultSettings,
  kSettingsStorageKey,
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

const kSettingsContext = createContext<SettingsContextValue | null>(null);

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<SettingsState>(kDefaultSettings);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(kSettingsStorageKey);
      if (raw) setState((current) => ({ ...current, ...JSON.parse(raw) }));
    } catch {
      // Storage unavailable (private mode, etc.) — fall back to defaults.
    }
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!mounted) return;
    localStorage.setItem(kSettingsStorageKey, JSON.stringify(state));
  }, [state, mounted]);

  useEffect(() => {
    const root = document.documentElement;
    if (state.text_size === 'standard') delete root.dataset.textSize;
    else root.dataset.textSize = state.text_size;
  }, [state.text_size]);

  useEffect(() => {
    const root = document.documentElement;
    if (state.code_wrap) delete root.dataset.codeWrap;
    else root.dataset.codeWrap = 'off';
  }, [state.code_wrap]);

  return (
    <kSettingsContext.Provider
      value={{
        ...state,
        mounted,
        setContentWidth: (content_width) => setState((s) => ({ ...s, content_width })),
        setTextSize: (text_size) => setState((s) => ({ ...s, text_size })),
        setCodeWrap: (code_wrap) => setState((s) => ({ ...s, code_wrap })),
      }}
    >
      {children}
    </kSettingsContext.Provider>
  );
}

export function useSettings(): SettingsContextValue {
  const ctx = useContext(kSettingsContext);
  if (!ctx) throw new Error('useSettings must be used within SettingsProvider');
  return ctx;
}
