'use client';

import type { ReactNode } from 'react';
import { useTheme } from 'next-themes';
import { useSettings } from '@/components/SettingsProvider';
import type { ContentWidth, TextSize } from '@/lib/settings';

const themeOptions = [
  { value: 'system', label: '跟隨系統', description: '使用作業系統目前的外觀設定。' },
  { value: 'light', label: '淺色', description: '固定使用淺色介面。' },
  { value: 'dark', label: '深色', description: '固定使用深色介面。' },
] as const;

const contentWidthOptions: { value: ContentWidth; label: string; description: string }[] = [
  { value: 'standard', label: '標準', description: '維持較集中的閱讀寬度，適合逐章閱讀。' },
  { value: 'wide', label: '寬', description: '放寬內容寬度至約 1760px，適合表格與大螢幕。' },
  { value: 'full', label: '全幅', description: '內容填滿視窗寬度，消除兩側空白，適合超寬螢幕。' },
];

const textSizeOptions: { value: TextSize; label: string; description: string }[] = [
  { value: 'small', label: '小', description: '提高資訊密度，適合大螢幕快速掃讀。' },
  { value: 'standard', label: '標準', description: '使用預設文字大小。' },
  { value: 'large', label: '大', description: '放大文字，適合長時間閱讀。' },
];

function OptionCard({
  selected,
  label,
  description,
  onClick,
}: {
  selected: boolean;
  label: string;
  description: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={selected}
      className={`flex flex-col items-start gap-1.5 rounded-xl border p-4 text-left transition ${
        selected ? 'border-primary bg-primary text-primary-foreground' : 'border-border text-foreground hover:border-primary'
      }`}
    >
      <span className="text-sm font-semibold">{label}</span>
      <span className={`text-xs leading-5 ${selected ? 'text-primary-foreground/85' : 'text-muted-foreground'}`}>
        {description}
      </span>
    </button>
  );
}

function SettingsSection({
  eyebrow,
  title,
  description,
  children,
}: {
  eyebrow: string;
  title: string;
  description: string;
  children: ReactNode;
}) {
  return (
    <section className="rounded-2xl border border-border bg-card/70 p-5">
      <p className="text-sm font-medium text-primary">{eyebrow}</p>
      <h3 className="mt-2 text-xl font-semibold text-foreground">{title}</h3>
      <p className="mt-2 text-sm leading-6 text-muted-foreground">{description}</p>
      {children}
    </section>
  );
}

export function SettingsPanel() {
  const { theme, resolvedTheme, setTheme } = useTheme();
  const { mounted, contentWidth, textSize, codeWrap, setContentWidth, setTextSize, setCodeWrap } = useSettings();

  const currentTheme = mounted ? theme ?? 'system' : 'system';

  return (
    <div className="grid gap-5 sm:grid-cols-2">
      <SettingsSection
        eyebrow="外觀"
        title="Theme"
        description="選擇網站的色彩模式。系統模式會依照瀏覽器或作業系統偏好自動切換。"
      >
        <div className="mt-5 grid gap-3 sm:grid-cols-3">
          {themeOptions.map((option) => (
            <OptionCard
              key={option.value}
              selected={currentTheme === option.value}
              label={option.label}
              description={option.description}
              onClick={() => setTheme(option.value)}
            />
          ))}
        </div>
        <p className="mt-4 text-sm text-muted-foreground">
          目前實際顯示：{mounted ? (resolvedTheme === 'dark' ? '深色' : '淺色') : '讀取中'}
        </p>
      </SettingsSection>

      <SettingsSection
        eyebrow="外觀"
        title="內容寬度"
        description="選擇主要內容區的最大寬度；寬版會讓大螢幕顯示更多內容。"
      >
        <div className="mt-5 grid gap-3">
          {contentWidthOptions.map((option) => (
            <OptionCard
              key={option.value}
              selected={contentWidth === option.value}
              label={option.label}
              description={option.description}
              onClick={() => setContentWidth(option.value)}
            />
          ))}
        </div>
      </SettingsSection>

      <SettingsSection
        eyebrow="外觀"
        title="文字大小"
        description="調整整個網站的文字比例，包含導覽、側欄與章節內容。"
      >
        <div className="mt-5 grid gap-3 sm:grid-cols-3">
          {textSizeOptions.map((option) => (
            <OptionCard
              key={option.value}
              selected={textSize === option.value}
              label={option.label}
              description={option.description}
              onClick={() => setTextSize(option.value)}
            />
          ))}
        </div>
      </SettingsSection>

      <SettingsSection
        eyebrow="閱讀"
        title="程式碼"
        description="控制程式碼區塊過寬時是否自動換行，避免水平捲動。"
      >
        <label className="mt-5 flex items-center gap-2 text-sm text-foreground">
          <input
            type="checkbox"
            checked={codeWrap}
            onChange={(event) => setCodeWrap(event.target.checked)}
            className="h-4 w-4 rounded border-border accent-primary"
          />
          程式碼自動換行
        </label>
      </SettingsSection>
    </div>
  );
}
