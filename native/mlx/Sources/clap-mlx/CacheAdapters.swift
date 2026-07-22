import ClapMLXCache

extension CacheIntent {
  var identityInput: CacheIdentityInput {
    CacheIdentityInput(namespace: namespace, tenant: tenant, project: project,
      harness: harness, agent: agent, session: session, priority: priority,
      sideRequest: side_request ?? false)
  }
}

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
