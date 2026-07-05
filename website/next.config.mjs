import createMDX from '@next/mdx';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeSlug from 'rehype-slug';
import rehypePrettyCode from 'rehype-pretty-code';

/**
 * GitHub Pages project sites are served from /<repo>. Allow overriding the
 * base path via an environment variable so local dev can run at the root.
 */
const kIsProd = process.env.NODE_ENV === 'production';
const kBasePath = process.env.NEXT_PUBLIC_BASE_PATH ?? (kIsProd ? '/gpu-kernel-lab' : '');

const kPrettyCodeOptions = {
  theme: 'github-dark-dimmed',
  keepBackground: true,
};

const withMdx = createMDX({
  extension: /\.mdx?$/,
  options: {
    remarkPlugins: [remarkGfm, remarkMath],
    rehypePlugins: [rehypeSlug, [rehypePrettyCode, kPrettyCodeOptions], rehypeKatex],
  },
});

/** @type {import('next').NextConfig} */
const kNextConfig = {
  output: 'export',
  reactStrictMode: true,
  pageExtensions: ['ts', 'tsx', 'md', 'mdx'],
  basePath: kBasePath,
  images: { unoptimized: true },
  trailingSlash: true,
  env: {
    NEXT_PUBLIC_BASE_PATH: kBasePath,
  },
};

export default withMdx(kNextConfig);
