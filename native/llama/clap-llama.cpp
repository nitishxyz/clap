#include "llama.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct LoadedLlama {
  std::string model_path;
  llama_model* model = nullptr;
  llama_context* ctx = nullptr;
};

std::string json_escape(const std::string& value) {
  std::ostringstream out;
  for (const char ch : value) {
    switch (ch) {
      case '\\': out << "\\\\"; break;
      case '"': out << "\\\""; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default: out << ch; break;
    }
  }
  return out.str();
}

std::string extract_string(const std::string& json, const std::string& key) {
  const std::string needle = "\"" + key + "\"";
  auto pos = json.find(needle);
  if (pos == std::string::npos) return "";
  pos = json.find(':', pos + needle.size());
  if (pos == std::string::npos) return "";
  pos = json.find('"', pos + 1);
  if (pos == std::string::npos) return "";

  std::string value;
  bool escape = false;
  for (std::size_t index = pos + 1; index < json.size(); ++index) {
    const char ch = json[index];
    if (escape) {
      switch (ch) {
        case 'n': value += '\n'; break;
        case 'r': value += '\r'; break;
        case 't': value += '\t'; break;
        case '"': value += '"'; break;
        case '\\': value += '\\'; break;
        default: value += ch; break;
      }
      escape = false;
    } else if (ch == '\\') {
      escape = true;
    } else if (ch == '"') {
      return value;
    } else {
      value += ch;
    }
  }
  return value;
}

double extract_double(const std::string& json, const std::string& key, double fallback) {
  const std::string needle = "\"" + key + "\"";
  auto pos = json.find(needle);
  if (pos == std::string::npos) return fallback;
  pos = json.find(':', pos + needle.size());
  if (pos == std::string::npos) return fallback;
  try {
    return std::stod(json.substr(pos + 1));
  } catch (...) {
    return fallback;
  }
}

int extract_int(const std::string& json, const std::string& key, int fallback) {
  const std::string needle = "\"" + key + "\"";
  auto pos = json.find(needle);
  if (pos == std::string::npos) return fallback;
  pos = json.find(':', pos + needle.size());
  if (pos == std::string::npos) return fallback;
  try {
    return std::stoi(json.substr(pos + 1));
  } catch (...) {
    return fallback;
  }
}

std::string prompt_from_messages(const std::string& json) {
  std::ostringstream prompt;
  std::size_t pos = 0;
  while ((pos = json.find("\"content\"", pos)) != std::string::npos) {
    const auto colon = json.find(':', pos + 9);
    if (colon == std::string::npos) break;
    const auto quote = json.find('"', colon + 1);
    if (quote == std::string::npos) {
      pos = colon + 1;
      continue;
    }
    std::string content;
    bool escape = false;
    for (std::size_t index = quote + 1; index < json.size(); ++index) {
      const char ch = json[index];
      if (escape) {
        switch (ch) {
          case 'n': content += '\n'; break;
          case 'r': content += '\r'; break;
          case 't': content += '\t'; break;
          case '"': content += '"'; break;
          case '\\': content += '\\'; break;
          default: content += ch; break;
        }
        escape = false;
      } else if (ch == '\\') {
        escape = true;
      } else if (ch == '"') {
        pos = index + 1;
        break;
      } else {
        content += ch;
      }
    }
    if (!content.empty()) prompt << content << "\n";
  }
  return prompt.str();
}

std::string prefix_for_id(const std::string& id) {
  return id.empty() ? "{" : "{\"id\":\"" + json_escape(id) + "\",";
}

void emit(const std::string& id, const std::string& fields) {
  std::cout << prefix_for_id(id) << fields << "}\n";
  std::cout.flush();
}

void emit_error(const std::string& id, const std::string& message) {
  emit(id, "\"error\":\"" + json_escape(message) + "\"");
}

void unload(LoadedLlama& loaded) {
  if (loaded.ctx) {
    llama_free(loaded.ctx);
    loaded.ctx = nullptr;
  }
  if (loaded.model) {
    llama_model_free(loaded.model);
    loaded.model = nullptr;
  }
  loaded.model_path.clear();
}

void load_model(LoadedLlama& loaded, const std::string& model_path) {
  if (loaded.model && loaded.ctx && loaded.model_path == model_path) return;
  unload(loaded);

  if (!std::filesystem::exists(model_path)) {
    throw std::runtime_error("GGUF model not found: " + model_path);
  }

  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = 999;
  loaded.model = llama_model_load_from_file(model_path.c_str(), model_params);
  if (!loaded.model) throw std::runtime_error("failed to load GGUF model: " + model_path);

  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 4096;
  ctx_params.n_batch = 512;
  ctx_params.no_perf = true;
  loaded.ctx = llama_init_from_model(loaded.model, ctx_params);
  if (!loaded.ctx) {
    llama_model_free(loaded.model);
    loaded.model = nullptr;
    throw std::runtime_error("failed to create llama context for: " + model_path);
  }
  loaded.model_path = model_path;
}

std::string token_to_piece(const llama_vocab* vocab, llama_token token) {
  char buffer[256];
  int n = llama_token_to_piece(vocab, token, buffer, sizeof(buffer), 0, true);
  if (n < 0) return "";
  return std::string(buffer, n);
}

void generate(LoadedLlama& loaded, const std::string& id, const std::string& request) {
  const std::string model_path = extract_string(request, "model");
  if (model_path.empty()) throw std::runtime_error("chat.model is required");
  load_model(loaded, model_path);

  const llama_vocab* vocab = llama_model_get_vocab(loaded.model);
  std::string prompt = prompt_from_messages(request);
  if (prompt.empty()) prompt = extract_string(request, "prompt");
  if (prompt.empty()) prompt = "Hello";

  const int max_tokens = extract_int(request, "max_tokens", 256);
  const double temperature = extract_double(request, "temperature", 0.7);
  const double top_p = extract_double(request, "top_p", 0.95);
  const int seed = extract_int(request, "seed", LLAMA_DEFAULT_SEED);

  const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
  if (n_prompt <= 0) throw std::runtime_error("failed to tokenize prompt");
  std::vector<llama_token> prompt_tokens(n_prompt);
  if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
    throw std::runtime_error("failed to tokenize prompt");
  }

  llama_memory_clear(llama_get_memory(loaded.ctx), true);

  llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
  if (llama_model_has_encoder(loaded.model)) {
    if (llama_encode(loaded.ctx, batch)) throw std::runtime_error("llama_encode failed");
    llama_token decoder_start_token_id = llama_model_decoder_start_token(loaded.model);
    if (decoder_start_token_id == LLAMA_TOKEN_NULL) decoder_start_token_id = llama_vocab_bos(vocab);
    batch = llama_batch_get_one(&decoder_start_token_id, 1);
  }

  auto sparams = llama_sampler_chain_default_params();
  sparams.no_perf = true;
  llama_sampler* sampler = llama_sampler_chain_init(sparams);
  if (temperature <= 0.0) {
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
  } else {
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(static_cast<float>(top_p), 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(static_cast<float>(temperature)));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(seed));
  }

  for (int n_pos = 0, n_decode = 0; n_decode < max_tokens; ) {
    if (llama_decode(loaded.ctx, batch)) {
      llama_sampler_free(sampler);
      throw std::runtime_error("llama_decode failed");
    }
    n_pos += batch.n_tokens;

    llama_token token = llama_sampler_sample(sampler, loaded.ctx, -1);
    if (llama_vocab_is_eog(vocab, token)) break;

    const std::string piece = token_to_piece(vocab, token);
    if (!piece.empty()) emit(id, "\"token\":\"" + json_escape(piece) + "\"");

    batch = llama_batch_get_one(&token, 1);
    n_decode += 1;
    if (n_pos >= 4095) break;
  }

  llama_sampler_free(sampler);
  emit(id, "\"done\":true");
}

}  // namespace

int main() {
  LoadedLlama loaded;
  try {
    ggml_backend_load_all();

    std::string request;
    while (std::getline(std::cin, request)) {
      if (request.empty()) continue;
      const std::string id = extract_string(request, "id");
      const std::string type = extract_string(request, "type");

      try {
        if (type == "shutdown") {
          emit(id, "\"done\":true");
          unload(loaded);
          return 0;
        }
        if (type == "unload") {
          unload(loaded);
          emit(id, "\"unloaded\":true,\"done\":true");
          continue;
        }
        if (type == "load") {
          const std::string model = extract_string(request, "model");
          if (model.empty()) throw std::runtime_error("load.model is required");
          load_model(loaded, model);
          emit(id, "\"loaded\":true,\"done\":true");
          continue;
        }
        generate(loaded, id, request);
      } catch (const std::exception& error) {
        emit_error(id, error.what());
      }
    }
    unload(loaded);
    return 0;
  } catch (const std::exception& error) {
    emit_error("", error.what());
    unload(loaded);
    return 1;
  }
}
