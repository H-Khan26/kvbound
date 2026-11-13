#pragma once

#include <cstdint>
#include <vector>

namespace kvb {

// One token of one step, as the kernels see it. A step is a flat list of rows from any number of
// sequences, so prefill (many rows of one sequence) and decode (one row per sequence) share one
// code path. The array lives in a persistent device buffer that is refilled in place each step.
struct StepRow {
  std::int32_t token_id;
  std::int32_t position;  // index of the token within its sequence
  std::int32_t seq;       // row of the KV block table
  std::int32_t kv_slot;   // physical cache slot: block * block_size + offset in block
};

// What the engine does in one iteration, in the order free, preempt, alloc, run. Produced on the
// CPU; carries no device state.
struct StepPlan {
  struct Alloc {
    int seq;
    int n_blocks;  // blocks added to what the sequence already holds
  };
  struct Run {
    int seq;
    int n_tokens;  // next tokens of the sequence to process: a prompt chunk, or 1 when decoding
  };
  std::vector<int> free;  // finished sequences: their blocks return to the pool
  // Sequences that lose their blocks but keep their tokens. A later plan allocates blocks for
  // them again and runs every token from the first, which recomputes the cache.
  std::vector<int> preempt;
  std::vector<Alloc> alloc;
  std::vector<Run> run;

  bool empty() const noexcept {
    return free.empty() && preempt.empty() && alloc.empty() && run.empty();
  }
};

// Tokens sampled by one step: one per sequence whose known tokens were all processed.
struct StepResult {
  struct Token {
    int seq;
    std::int32_t token;
  };
  std::vector<Token> tokens;
};

}  // namespace kvb
