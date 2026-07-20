#pragma once

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>

namespace clap::llama_boundary {

struct StableBoundary {
  std::string token_hash;
  std::size_t token_count = 0;
  std::string kind;

  bool available() const {
    return token_count > 0 && !token_hash.empty() && !kind.empty();
  }
};

template <typename Tokens, typename Fingerprint>
StableBoundary exact(const Tokens& tokens, std::size_t count, std::string kind,
                     Fingerprint fingerprint) {
  if (count == 0 || count > tokens.size() || kind.empty()) return {};
  std::string hash = fingerprint(tokens, count);
  if (hash.empty()) return {};
  return {std::move(hash), count, std::move(kind)};
}

template <typename Tokens, typename IsTerminal>
std::optional<std::size_t> exact_template_boundary(const Tokens& prefix,
                                                   const Tokens& final,
                                                   IsTerminal is_terminal) {
  if (prefix.empty()) return std::nullopt;
  if (prefix.size() < final.size() &&
      std::equal(prefix.begin(), prefix.end(), final.begin())) return prefix.size();
  std::size_t shared = 0;
  while (shared < prefix.size() && shared < final.size() &&
         prefix[shared] == final[shared]) ++shared;
  if (shared == 0 || shared == prefix.size() || shared >= final.size()) return std::nullopt;
  if (!std::all_of(prefix.begin() + static_cast<std::ptrdiff_t>(shared), prefix.end(),
                   is_terminal)) return std::nullopt;
  return shared;
}

}  // namespace clap::llama_boundary
