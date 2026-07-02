// attention_naive.cpp
//
// Minimal single-head attention forward pass, one block per query row, using
// online softmax so the row of scores never materializes in global memory.
// This is the pedagogical core of FlashAttention (chapter 24): tile over
// keys, keep running max / denominator / weighted accumulator, rescale when
// the max moves. Verified against a straightforward host reference.
#include <cmath>
#include <cstdio>
#include <vector>

#include "benchmark.h"
#include "gpu_portability.h"

namespace {

constexpr int kSeqLen = 1024;
constexpr int kHeadDim = 64;  // one thread per dim element; kHeadDim <= blockDim
constexpr int kBlockSize = 64;

// One block handles one query row. Threads cooperate across the head dim;
// the online-softmax state (m, d) lives in shared memory, updated by thread 0
// while everyone maintains the rescaled output accumulator for its element.
__global__ void attentionOnline(const float* q, const float* k, const float* v,
                                float* out, int seq_len, float scale) {
  int row = blockIdx.x;
  int e = threadIdx.x;  // which element of the head dim this thread owns

  __shared__ float score_parts[kHeadDim];
  __shared__ float m_shared;  // running max
  __shared__ float d_shared;  // running denominator
  __shared__ float p_shared;  // exp(score - new_max) for the current key

  float acc = 0.0f;  // running numerator for out[row][e]
  if (e == 0) {
    m_shared = -1e30f;
    d_shared = 0.0f;
  }
  __syncthreads();

  for (int key = 0; key < seq_len; ++key) {
    // score = scale * dot(q[row], k[key]) via a shared-memory tree reduction.
    score_parts[e] = q[row * kHeadDim + e] * k[key * kHeadDim + e];
    __syncthreads();
    for (int stride = kHeadDim / 2; stride > 0; stride /= 2) {
      if (e < stride) score_parts[e] += score_parts[e + stride];
      __syncthreads();
    }

    float m_old = m_shared;
    if (e == 0) {
      float s = score_parts[0] * scale;
      float m_new = fmaxf(m_old, s);
      // Rescale the old denominator to the new max, then add this key.
      p_shared = expf(s - m_new);
      d_shared = d_shared * expf(m_old - m_new) + p_shared;
      m_shared = m_new;
    }
    __syncthreads();

    // Every thread rescales its accumulator the same way and adds p * v.
    acc = acc * expf(m_old - m_shared) + p_shared * v[key * kHeadDim + e];
    __syncthreads();
  }

  out[row * kHeadDim + e] = acc / d_shared;
}

}  // namespace

int main() {
  const int n = kSeqLen * kHeadDim;
  const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));

  std::vector<float> q(n), k(n), v(n);
  for (int i = 0; i < n; ++i) {
    q[i] = static_cast<float>((i * 37) % 19 - 9) * 0.05f;
    k[i] = static_cast<float>((i * 53) % 17 - 8) * 0.05f;
    v[i] = static_cast<float>((i * 71) % 23 - 11) * 0.05f;
  }

  // Host reference: standard two-pass softmax attention.
  std::vector<float> expected(n, 0.0f);
  std::vector<float> scores(kSeqLen);
  for (int row = 0; row < kSeqLen; ++row) {
    float m = -1e30f;
    for (int key = 0; key < kSeqLen; ++key) {
      float s = 0.0f;
      for (int e = 0; e < kHeadDim; ++e) s += q[row * kHeadDim + e] * k[key * kHeadDim + e];
      scores[key] = s * scale;
      m = std::max(m, scores[key]);
    }
    float d = 0.0f;
    for (int key = 0; key < kSeqLen; ++key) {
      scores[key] = std::exp(scores[key] - m);
      d += scores[key];
    }
    for (int key = 0; key < kSeqLen; ++key) {
      for (int e = 0; e < kHeadDim; ++e) {
        expected[row * kHeadDim + e] += scores[key] / d * v[key * kHeadDim + e];
      }
    }
  }

  const size_t bytes = static_cast<size_t>(n) * sizeof(float);
  float *dev_q = nullptr, *dev_k = nullptr, *dev_v = nullptr, *dev_out = nullptr;
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_q), bytes));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_k), bytes));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_v), bytes));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_out), bytes));
  GPU_CHECK(gpuMemcpyHostToDevice(dev_q, q.data(), bytes));
  GPU_CHECK(gpuMemcpyHostToDevice(dev_k, k.data(), bytes));
  GPU_CHECK(gpuMemcpyHostToDevice(dev_v, v.data(), bytes));

  auto launch = [&]() {
    GPU_LAUNCH(attentionOnline, kSeqLen, kBlockSize, 0, dev_q, dev_k, dev_v, dev_out,
               kSeqLen, scale);
  };
  launch();
  GPU_CHECK(gpuDeviceSynchronize());

  std::vector<float> result(n);
  GPU_CHECK(gpuMemcpyDeviceToHost(result.data(), dev_out, bytes));
  if (!gklab::verifyClose(result, expected, 1e-3f)) return EXIT_FAILURE;
  std::printf("attention_naive: online softmax matches two-pass host reference "
              "(seq %d, head dim %d)\n", kSeqLen, kHeadDim);

  constexpr double kPeakGbPerSec = 1555.0;
  // Traffic estimate: every query row streams the full K and V once.
  const size_t moved = 2 * static_cast<size_t>(kSeqLen) * bytes;
  gklab::report("attention_online", gklab::benchmarkKernel(launch, moved, 0.0), kPeakGbPerSec, 0.0);

  GPU_CHECK(gpuFree(dev_q));
  GPU_CHECK(gpuFree(dev_k));
  GPU_CHECK(gpuFree(dev_v));
  GPU_CHECK(gpuFree(dev_out));
  return EXIT_SUCCESS;
}
