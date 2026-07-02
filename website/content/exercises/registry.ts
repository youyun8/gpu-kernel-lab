import type { ComponentType } from 'react';

import TrackA from './track-a.mdx';
import TrackB from './track-b.mdx';
import TrackC from './track-c.mdx';
import TrackD from './track-d.mdx';

export const exerciseComponents: Record<string, ComponentType> = {
  'track-a': TrackA,
  'track-b': TrackB,
  'track-c': TrackC,
  'track-d': TrackD,
};
