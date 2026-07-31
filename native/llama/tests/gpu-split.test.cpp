#include "clap/llama/gpu-split.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

int main() {
  using clap::llama_gpu::derive_split;
  using clap::llama_gpu::parse_split_mode;
  using clap::llama_gpu::parse_tensor_split;

  {
    llama_split_mode mode = LLAMA_SPLIT_MODE_NONE;
    assert(parse_split_mode("", &mode) && mode == LLAMA_SPLIT_MODE_LAYER);
    assert(parse_split_mode("layer", &mode) && mode == LLAMA_SPLIT_MODE_LAYER);
    assert(parse_split_mode("none", &mode) && mode == LLAMA_SPLIT_MODE_NONE);
    assert(parse_split_mode("row", &mode) && mode == LLAMA_SPLIT_MODE_ROW);
    assert(!parse_split_mode("tensor", &mode));
    assert(!parse_split_mode("LAYER", &mode));
  }

  {
    std::vector<float> split;
    assert(parse_tensor_split("", 4, &split) && split.empty());
    assert(parse_tensor_split("3,1", 4, &split));
    assert(split.size() == 2 && split[0] == 3.0f && split[1] == 1.0f);
    assert(parse_tensor_split("0.5,0.25,0.25", 4, &split) && split.size() == 3);
    assert(parse_tensor_split("0,1", 2, &split) && split[0] == 0.0f);
    // Malformed, negative, all-zero, and oversized lists are rejected.
    assert(!parse_tensor_split("1,x", 4, &split));
    assert(!parse_tensor_split("1,", 4, &split));
    assert(!parse_tensor_split("-1,2", 4, &split));
    assert(!parse_tensor_split("0,0", 4, &split));
    assert(!parse_tensor_split("1,1,1", 2, &split));
  }

  {
    // Defaults: layer split, main GPU 0, automatic distribution.
    const auto decision = derive_split("", 0, "", 8);
    assert(decision.valid);
    assert(decision.mode == LLAMA_SPLIT_MODE_LAYER);
    assert(decision.main_gpu == 0);
    assert(decision.tensor_split.empty());
  }

  {
    const auto decision = derive_split("row", 1, "3,1", 8);
    assert(decision.valid);
    assert(decision.mode == LLAMA_SPLIT_MODE_ROW);
    assert(decision.main_gpu == 1);
    assert(decision.tensor_split.size() == 2);
  }

  {
    // Misconfiguration produces a described failure, never a silent default.
    assert(!derive_split("diagonal", 0, "", 8).valid);
    assert(!derive_split("layer", -1, "", 8).valid);
    assert(!derive_split("layer", 8, "", 8).valid);
    assert(!derive_split("layer", 0, "1,oops", 8).valid);
    assert(!derive_split("", 0, "1,1,1", 2).valid);
    assert(!derive_split("diagonal", 0, "", 8).error.empty());
  }

  return 0;
}
