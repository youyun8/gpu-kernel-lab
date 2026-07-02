import type { Metadata } from 'next';
import './globals.css';
import { ThemeProvider } from '@/components/ThemeProvider';
import { SettingsProvider } from '@/components/SettingsProvider';
import { SiteHeader } from '@/components/SiteHeader';
import { SiteFooter } from '@/components/SiteFooter';
import { siteConfig } from '@/lib/site';

export const metadata: Metadata = {
  title: {
    default: `${siteConfig.name} — ${siteConfig.tagline}`,
    template: `%s | ${siteConfig.name}`,
  },
  description:
    'GPU Kernel Lab 是一個以繁體中文撰寫的 CUDA/ROCm kernel 優化學習網站, 從入門到專家, 搭配可執行的 kernels、benchmark 與 profiling 範例。',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-TW" suppressHydrationWarning>
      <body className="min-h-screen bg-background">
        <ThemeProvider>
          <SettingsProvider>
            <a
              href="#main-content"
              className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded focus:bg-primary focus:px-4 focus:py-2 focus:text-primary-foreground"
            >
              跳到主要內容
            </a>
            <SiteHeader />
            <div id="main-content">{children}</div>
            <SiteFooter />
          </SettingsProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
