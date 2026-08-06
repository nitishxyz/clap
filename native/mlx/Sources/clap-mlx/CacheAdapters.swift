import ClapMLXCache

extension CheckpointConfiguration {
  var coordinatorConfiguration: CoordinatorCheckpointConfiguration {
    CoordinatorCheckpointConfiguration(enabled: enabled,
      minimumTokens: coordinatorMinimumTokens,
      intervalTokens: coordinatorIntervalTokens,
      maximum: coordinatorMaximum,
      maximumPerSession: coordinatorMaximumPerSession,
      maximumAnchorsPerSession: maximumAnchorsPerSession,
      maximumAnchorBytesPerSession: maximumAnchorBytesPerSession,
      sessionIdleTTLMilliseconds: sessionIdleTTLMilliseconds,
      budgetBasisPoints: budgetBasisPoints,
      budgetBytes: budgetBytes)
  }
}
