#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace clap::llama::test {

class Sha256 {
 public:
  static std::string hex(const uint8_t* data, std::size_t size) {
    std::array<uint32_t, 8> state{
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    std::vector<uint8_t> padded(data, data + size);
    padded.push_back(0x80);
    while ((padded.size() % 64) != 56) padded.push_back(0);
    const uint64_t bits = static_cast<uint64_t>(size) * 8;
    for (int shift = 56; shift >= 0; shift -= 8) {
      padded.push_back(static_cast<uint8_t>(bits >> shift));
    }
    for (std::size_t offset = 0; offset < padded.size(); offset += 64) {
      compress(state, padded.data() + offset);
    }
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (uint32_t word : state) out << std::setw(8) << word;
    return out.str();
  }

  static std::string hex(const std::vector<uint8_t>& bytes) {
    static constexpr uint8_t empty = 0;
    return hex(bytes.empty() ? &empty : bytes.data(), bytes.size());
  }

 private:
  static uint32_t rotate(uint32_t value, uint32_t count) {
    return (value >> count) | (value << (32 - count));
  }

  static void compress(std::array<uint32_t, 8>& state, const uint8_t* block) {
    static constexpr uint32_t constants[64] = {
      0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
      0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
      0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
      0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
      0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
      0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
      0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
      0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};
    uint32_t words[64]{};
    for (int index = 0; index < 16; ++index) {
      const int base = index * 4;
      words[index] = (static_cast<uint32_t>(block[base]) << 24) |
          (static_cast<uint32_t>(block[base + 1]) << 16) |
          (static_cast<uint32_t>(block[base + 2]) << 8) | block[base + 3];
    }
    for (int index = 16; index < 64; ++index) {
      const uint32_t left = rotate(words[index - 15], 7) ^ rotate(words[index - 15], 18) ^
          (words[index - 15] >> 3);
      const uint32_t right = rotate(words[index - 2], 17) ^ rotate(words[index - 2], 19) ^
          (words[index - 2] >> 10);
      words[index] = words[index - 16] + left + words[index - 7] + right;
    }
    uint32_t a=state[0], b=state[1], c=state[2], d=state[3];
    uint32_t e=state[4], f=state[5], g=state[6], h=state[7];
    for (int index = 0; index < 64; ++index) {
      const uint32_t s1 = rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25);
      const uint32_t choice = (e & f) ^ (~e & g);
      const uint32_t temp1 = h + s1 + choice + constants[index] + words[index];
      const uint32_t s0 = rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22);
      const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t temp2 = s0 + majority;
      h=g; g=f; f=e; e=d+temp1; d=c; c=b; b=a; a=temp1+temp2;
    }
    state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
    state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
  }
};

inline void append_i32(std::vector<uint8_t>& bytes, int32_t value) {
  const uint32_t word = static_cast<uint32_t>(value);
  for (int shift = 0; shift < 32; shift += 8) bytes.push_back(word >> shift);
}

inline std::string token_fingerprint(const std::vector<int32_t>& tokens) {
  std::vector<uint8_t> bytes;
  bytes.reserve(tokens.size() * 4);
  for (int32_t token : tokens) append_i32(bytes, token);
  return Sha256::hex(bytes);
}

struct QuantizedLogit {
  int32_t token;
  int32_t value;
  bool operator==(const QuantizedLogit& other) const {
    return token == other.token && value == other.value;
  }
};

inline std::vector<QuantizedLogit> top_quantized_logits(
    const float* logits, std::size_t count, std::size_t limit = 16) {
  std::vector<QuantizedLogit> values;
  values.reserve(count);
  for (std::size_t token = 0; token < count; ++token) {
    values.push_back({static_cast<int32_t>(token),
      static_cast<int32_t>(std::llround(static_cast<double>(logits[token]) * 1024.0))});
  }
  std::sort(values.begin(), values.end(), [](const auto& left, const auto& right) {
    return left.value != right.value ? left.value > right.value : left.token < right.token;
  });
  values.resize(std::min(limit, values.size()));
  return values;
}

inline std::string logit_fingerprint(const std::vector<QuantizedLogit>& logits) {
  std::vector<uint8_t> bytes;
  bytes.reserve(logits.size() * 8);
  for (const auto& logit : logits) {
    append_i32(bytes, logit.token);
    append_i32(bytes, logit.value);
  }
  return Sha256::hex(bytes);
}

struct CacheProbeObservation {
  uint32_t operation = 0;
  uint64_t reused = 0;
  uint64_t generation = 0;
  std::string logical_token_sha256;
  std::string physical_state_sha256;
  int32_t selected_next_token = -1;
  std::string top16_quantized_logit_sha256;
  std::vector<QuantizedLogit> top16_quantized_logits;
};

}  // namespace clap::llama::test
