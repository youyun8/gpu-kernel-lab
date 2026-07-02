export type ContentWidth = 'standard' | 'wide' | 'full';
export type TextSize = 'small' | 'standard' | 'large';

export interface SettingsState {
  contentWidth: ContentWidth;
  textSize: TextSize;
  codeWrap: boolean;
}

export const defaultSettings: SettingsState = {
  contentWidth: 'standard',
  textSize: 'standard',
  codeWrap: true,
};

export const SETTINGS_STORAGE_KEY = 'gklab-settings';
