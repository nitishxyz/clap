import ClapMLXCache

extension CheckpointConfiguration {
  var coordinatorConfiguration: CoordinatorCheckpointConfiguration {
    CoordinatorCheckpointConfiguration(enabled: enabled,
      minimumTokens: coordinatorMinimumTokens,
      intervalTokens: coordinatorIntervalTokens,
      maximum: coordinatorMaximum,
      budgetBasisPoints: budgetBasisPoints,
      budgetBytes: budgetBytes)
  }
}
