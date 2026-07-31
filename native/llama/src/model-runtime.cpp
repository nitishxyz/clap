#include "clap/llama/model-runtime.h"

#include "clap/llama/environment.h"
#include "clap/llama/kv-fit.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>

namespace clap::llama {

ModelRuntime::~ModelRuntime() {
  reset();
}

bool ModelRuntime::same_path(const std::string& model_path) const noexcept {
  return loaded() && model_path_ == model_path;
}

const llama_vocab* ModelRuntime::vocab() const noexcept {
  return model_ ? llama_model_get_vocab(model_) : nullptr;
}

void ModelRuntime::reset() noexcept {
  if (context_) {
    llama_free(context_);
    context_ = nullptr;
  }
  if (model_) {
    llama_model_free(model_);
    model_ = nullptr;
  }
  model_path_.clear();
  model_context_window_ = 0;
  backend_allocation_cap_ = 0;
  context_override_ = 0;
  max_output_tokens_ = 0;
  retained_max_ = 0;
  startup_available_bytes_ = 0;
  model_file_bytes_ = 0;
  hybrid_ = false;
  prompt_boundary_snapshots_ = false;
  has_encoder_ = false;
  cache_domain_.clear();
  kv_format_.clear();
  unified_kv_ = true;
}

bool ModelRuntime::load(const std::string& model_path) {
  if (same_path(model_path)) return false;
  reset();

  if (!std::filesystem::exists(model_path)) {
    throw std::runtime_error("GGUF model not found: " + model_path);
  }
  const uint64_t startup_available = available_memory_bytes();
  std::error_code file_size_error;
  const uint64_t model_file_bytes = std::filesystem::file_size(model_path, file_size_error);
  if (env_enabled("CLAP_LLAMA_KV_BUDGET_BYTES") ||
      env_enabled("CLAP_LLAMA_KV_BUDGET_PERCENT")) {
    throw std::runtime_error(
        "CLAP_LLAMA_KV_BUDGET_BYTES/PERCENT is unsupported by the pinned llama.cpp API: "
        "authoritative KV used/free cells and bytes are unavailable; use CLAP_LLAMA_CONTEXT");
  }

  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = env_int("CLAP_LLAMA_GPU_LAYERS", 999);
  model_ = llama_model_load_from_file(model_path.c_str(), model_params);
  if (!model_) throw std::runtime_error("failed to load GGUF model: " + model_path);

  llama_context_params context_params = llama_context_default_params();
  const int32_t env_context = env_int("CLAP_LLAMA_CONTEXT", 0);
  const int32_t train_context = llama_model_n_ctx_train(model_);
  int32_t context_size = train_context > 0 && env_context > 0
      ? std::min(train_context, env_context)
      : (env_context > 0 ? env_context : train_context);
  const char* kv_type_env = std::getenv("CLAP_LLAMA_KV_TYPE");
  const std::string kv_type_name = kv_type_env && *kv_type_env ? kv_type_env : "f16";
  if (env_context <= 0 && env_int("CLAP_LLAMA_KV_AUTOFIT", 1) != 0) {
    // Automatic context: shrink the unified KV pool allocation until its
    // estimated cost fits measured availability (scale plan T2.6). The
    // estimate is a scheduling input, never reported as measured KV bytes.
    const int32_t n_head = llama_model_n_head(model_);
    const int32_t head_dim = n_head > 0 ? llama_model_n_embd(model_) / n_head : 0;
    clap::llama_kv::FitInputs fit_inputs;
    fit_inputs.requested_context = 0;
    fit_inputs.train_context = context_size;
    fit_inputs.kv_bytes_per_token = clap::llama_kv::estimate_kv_bytes_per_token(
        llama_model_n_layer(model_), llama_model_n_head_kv(model_), head_dim,
        kv_type_name);
    fit_inputs.available_bytes = startup_available;
    fit_inputs.model_file_bytes = file_size_error ? 0 : model_file_bytes;
    const auto fit = clap::llama_kv::fit_context(fit_inputs);
    if (fit.clamped) {
      fprintf(stderr,
          "clap-llama: auto context reduced %d -> %d cells (%s; estimated KV pool "
          "%llu bytes, budget %llu bytes); set CLAP_LLAMA_CONTEXT or "
          "CLAP_LLAMA_KV_AUTOFIT=0 to override\n",
          context_size, fit.context_cells, fit.reason.c_str(),
          static_cast<unsigned long long>(fit.estimated_pool_bytes),
          static_cast<unsigned long long>(fit.budget_bytes));
      context_size = fit.context_cells;
    }
  }
  context_params.n_batch = env_int("CLAP_LLAMA_BATCH", 2048);
  context_params.n_ubatch = env_int("CLAP_LLAMA_UBATCH", 512);
  const int32_t retained_override = env_int("CLAP_LLAMA_RETAINED_MAX", 0);
  const int32_t derived_retained =
      std::min(128, std::max(32, std::max(context_size, 1) / 256));
  const int32_t retained_max = retained_override > 0 ? retained_override : derived_retained;
  context_params.n_seq_max = retained_max;
  context_params.kv_unified = env_int("CLAP_LLAMA_KV_UNIFIED", 1) != 0;
  if (const char* kv_type = std::getenv("CLAP_LLAMA_KV_TYPE"); kv_type && *kv_type) {
    const std::string requested(kv_type);
    if (requested == "q8_0") {
      context_params.type_k = GGML_TYPE_Q8_0;
      context_params.type_v = GGML_TYPE_Q8_0;
    } else if (requested == "q4_0") {
      context_params.type_k = GGML_TYPE_Q4_0;
      context_params.type_v = GGML_TYPE_Q4_0;
    } else if (requested != "f16") {
      fprintf(stderr, "clap-llama: unknown CLAP_LLAMA_KV_TYPE '%s'; using f16\n", kv_type);
    }
  }
  context_params.no_perf = true;
  context_params.n_ctx = std::max(context_size, 0);
  context_ = llama_init_from_model(model_, context_params);
  if (!context_) {
    llama_model_free(model_);
    model_ = nullptr;
    throw std::runtime_error("failed to create llama context for: " + model_path);
  }

  model_path_ = model_path;
  model_context_window_ = train_context > 0 ? train_context : 0;
  backend_allocation_cap_ = static_cast<int32_t>(llama_n_ctx(context_));
  context_override_ = env_context > 0 ? env_context : 0;
  max_output_tokens_ = std::max(0, env_int("CLAP_LLAMA_MAX_OUTPUT", 0));
  retained_max_ = retained_max;
  startup_available_bytes_ = startup_available;
  model_file_bytes_ = file_size_error ? 0 : model_file_bytes;
  hybrid_ = llama_model_is_recurrent(model_) || llama_model_is_hybrid(model_);
  char architecture[64] = {};
  const bool has_architecture = llama_model_meta_val_str(
      model_, "general.architecture", architecture, sizeof(architecture)) >= 0;
  // Gemma 4's shared KV layers cannot currently be cloned into another
  // sequence with llama_memory_seq_cp: the first suffix decode fails batch
  // initialization. Continue/branch remain valid, but anchor restore must be
  // rejected before planning until llama.cpp exposes copy-safe shared state.
  prompt_boundary_snapshots_ = !hybrid_ &&
      (!has_architecture || std::string(architecture) != "gemma4");
  has_encoder_ = llama_model_has_encoder(model_);
  kv_format_ = kv_type_name;
  unified_kv_ = context_params.kv_unified;
  cache_domain_ = model_path + "|llama|ctx=" + std::to_string(backend_allocation_cap_) +
      "|kv=" + kv_format_ +
      "|unified=" + (context_params.kv_unified ? "1" : "0") +
      "|layout=1";
  return true;
}

}  // namespace clap::llama
