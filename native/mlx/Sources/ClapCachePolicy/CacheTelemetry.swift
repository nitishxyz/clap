public func normalizedCacheReuseKind(operation: UInt32) -> String? {
  switch operation {
  case 1: return "slot"
  case 2: return "branch"
  case 3: return "anchor"
  default: return nil
  }
}
