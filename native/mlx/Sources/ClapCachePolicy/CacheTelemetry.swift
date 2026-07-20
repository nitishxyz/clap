public func normalizedCacheReuseKind(operation: UInt32) -> String? {
  switch operation {
  case 1: return "slot"
  case 2: return "branch"
  case 3: return "anchor"
  default: return nil
  }
}

// A partial template render may append EOS only because it is the end of that
// render. Remove that suffix only when every differing terminal token is the
// tokenizer's authoritative EOS token and the retained sequence is an exact
// prefix of the final prompt. No arbitrary common-prefix offset is accepted.
public func exactTemplateBoundary(prefix: [Int], final: [Int], eosToken: Int?) -> Int? {
  guard !prefix.isEmpty else { return nil }
  if prefix.count < final.count, prefix.elementsEqual(final.prefix(prefix.count)) { return prefix.count }
  guard let eosToken else { return nil }
  let shared = zip(prefix, final).prefix { $0.0 == $0.1 }.count
  guard shared > 0, shared < prefix.count,
        prefix[shared...].allSatisfy({ $0 == eosToken }),
        shared < final.count,
        prefix.prefix(shared).elementsEqual(final.prefix(shared)) else { return nil }
  return shared
}
