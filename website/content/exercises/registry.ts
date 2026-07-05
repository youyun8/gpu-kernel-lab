import type { ComponentType } from 'react';

import TrackP from './track-p.mdx';
import TrackM from './track-m.mdx';
import TrackA from './track-a.mdx';
import TrackB from './track-b.mdx';
import TrackC from './track-c.mdx';
import TrackD from './track-d.mdx';
import TrackE from './track-e.mdx';
import TrackSp from './track-sp.mdx';

export const kExerciseComponents: Record<string, ComponentType> = {
  'track-p': TrackP,
  'track-m': TrackM,
  'track-a': TrackA,
  'track-b': TrackB,
  'track-c': TrackC,
  'track-d': TrackD,
  'track-e': TrackE,
  'track-sp': TrackSp,
};
