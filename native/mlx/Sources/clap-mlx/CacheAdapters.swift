import ClapMLXCache

extension CheckpointConfiguration {
  var coordinatorConfiguration: CoordinatorCheckpointConfiguration {
    CoordinatorCheckpointConfiguration(enabled: enabled,
      minimumTokens: coordinatorMinimumTokens,
      intervalTokens: coordinatorIntervalTokens,
      maximum: coordinatorMaximum,
      maximumPerSession: coordinatorMaximumPerSession,
      maximumAnchorsPerSession: maximumAnchorsPerSession,
      budgetBasisPoints: budgetBasisPoints,
      budgetBytes: budgetBytes)
  }
}
