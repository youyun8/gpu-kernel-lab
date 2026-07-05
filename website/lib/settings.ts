export type ContentWidth = 'standard' | 'wide' | 'full';
export type TextSize = 'small' | 'standard' | 'large';

export interface SettingsState {
  content_width: ContentWidth;
  text_size: TextSize;
  code_wrap: boolean;
}

export const kDefaultSettings: SettingsState = {
  content_width: 'standard',
  text_size: 'standard',
  code_wrap: true,
};

export const kSettingsStorageKey = 'gklab-settings';
