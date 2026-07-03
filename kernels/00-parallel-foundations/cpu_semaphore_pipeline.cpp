// cpu_semaphore_pipeline.cpp
//
// Bounded producer/consumer queue using POSIX semaphores. Semaphores count
// resource availability (empty slots and full slots); a mutex still protects
// the queue indices because semaphores do not make compound data-structure
// updates mutually exclusive.
#include <cstdio>
#include <pthread.h>
#include <semaphore.h>

namespace {

constexpr int kCapacity = 8;
constexpr int kNumItems = 100000;
constexpr int kPoisonPill = -1;

struct BoundedQueue {
  int buffer[kCapacity]{};
  int head = 0;
  int tail = 0;
  pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
  sem_t empty_slots;
  sem_t full_slots;
};

void push(BoundedQueue* q, int value) {
  sem_wait(&q->empty_slots);
  pthread_mutex_lock(&q->mutex);
  q->buffer[q->tail] = value;
  q->tail = (q->tail + 1) % kCapacity;
  pthread_mutex_unlock(&q->mutex);
  sem_post(&q->full_slots);
}

int pop(BoundedQueue* q) {
  sem_wait(&q->full_slots);
  pthread_mutex_lock(&q->mutex);
  const int value = q->buffer[q->head];
  q->head = (q->head + 1) % kCapacity;
  pthread_mutex_unlock(&q->mutex);
  sem_post(&q->empty_slots);
  return value;
}

void* producer(void* raw) {
  auto* q = static_cast<BoundedQueue*>(raw);
  for (int i = 1; i <= kNumItems; ++i) {
    push(q, i);
  }
  push(q, kPoisonPill);
  return nullptr;
}

void* consumer(void* raw) {
  auto* q = static_cast<BoundedQueue*>(raw);
  long long* sum = new long long(0);
  while (true) {
    const int value = pop(q);
    if (value == kPoisonPill) break;
    *sum += value;
  }
  return sum;
}

}  // namespace

int main() {
  BoundedQueue queue;
  sem_init(&queue.empty_slots, 0, kCapacity);
  sem_init(&queue.full_slots, 0, 0);

  pthread_t producer_thread;
  pthread_t consumer_thread;
  pthread_create(&producer_thread, nullptr, producer, &queue);
  pthread_create(&consumer_thread, nullptr, consumer, &queue);

  pthread_join(producer_thread, nullptr);
  void* raw_sum = nullptr;
  pthread_join(consumer_thread, &raw_sum);
  const long long got = *static_cast<long long*>(raw_sum);
  delete static_cast<long long*>(raw_sum);

  sem_destroy(&queue.empty_slots);
  sem_destroy(&queue.full_slots);
  pthread_mutex_destroy(&queue.mutex);

  constexpr long long expected =
      static_cast<long long>(kNumItems) * (kNumItems + 1) / 2;
  std::printf("cpu_semaphore_pipeline: produced 1..%d through capacity-%d queue\n",
              kNumItems, kCapacity);
  std::printf("  consumer sum = %lld, expected = %lld  %s\n", got, expected,
              got == expected ? "OK" : "WRONG");
  std::printf("  empty_slots/full_slots gate queue capacity; mutex protects head/tail.\n");
  return got == expected ? 0 : 1;
}
