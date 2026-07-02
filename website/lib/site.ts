export const siteConfig = {
  name: 'GPU Kernel Lab',
  tagline: 'CUDA / ROCm kernel 優化的漸進式學習實驗室',
  repo: 'https://github.com/your-org/gpu-kernel-lab',
  requirements: 'CUDA ≥ 12.x 或 ROCm ≥ 6.x',
};

export const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? '';

/** Prefix an in-repo asset path with the configured basePath. */
export function asset(path: string): string {
  const normalized = path.startsWith('/') ? path : `/${path}`;
  return `${basePath}${normalized}`;
}
