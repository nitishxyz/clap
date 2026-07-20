#include "clap_cache.h"

#include <stddef.h>

_Static_assert(CLAP_CACHE_ABI_VERSION == 3u, "unexpected cache ABI version");
_Static_assert(sizeof(clap_cache_config_t) == 72, "clap_cache_config_t ABI size changed");
_Static_assert(offsetof(clap_cache_config_t, version) == 0, "version offset");
_Static_assert(offsetof(clap_cache_config_t, struct_size) == 4, "struct_size offset");
_Static_assert(offsetof(clap_cache_config_t, slot_count) == 8, "slot_count offset");
_Static_assert(offsetof(clap_cache_config_t, max_anchors) == 12, "max_anchors offset");
_Static_assert(offsetof(clap_cache_config_t, min_reuse_tokens) == 16, "min_reuse_tokens offset");
_Static_assert(offsetof(clap_cache_config_t, logical_token_capacity) == 24, "logical_token_capacity offset");
_Static_assert(offsetof(clap_cache_config_t, automatic_checkpoint_mode) == 32, "checkpoint mode offset");
_Static_assert(offsetof(clap_cache_config_t, automatic_checkpoint_max) == 36, "checkpoint max offset");
_Static_assert(offsetof(clap_cache_config_t, automatic_checkpoint_min_tokens) == 40, "checkpoint min offset");
_Static_assert(offsetof(clap_cache_config_t, automatic_checkpoint_interval_tokens) == 48, "checkpoint interval offset");
_Static_assert(offsetof(clap_cache_config_t, automatic_checkpoint_memory_basis_points) == 56, "checkpoint fraction offset");
_Static_assert(offsetof(clap_cache_config_t, reserved) == 60, "reserved offset");
_Static_assert(offsetof(clap_cache_config_t, automatic_checkpoint_memory_cap_bytes) == 64, "checkpoint cap offset");
_Static_assert(sizeof(clap_cache_retention_telemetry_t) == 112, "retention telemetry ABI size changed");
_Static_assert(offsetof(clap_cache_retention_telemetry_t, automatic_checkpoint_slots) == 48, "automatic slots offset");
_Static_assert(offsetof(clap_cache_retention_telemetry_t, automatic_checkpoint_bytes) == 56, "automatic bytes offset");
_Static_assert(offsetof(clap_cache_retention_telemetry_t, automatic_checkpoint_byte_budget) == 64, "automatic budget offset");
_Static_assert(offsetof(clap_cache_retention_telemetry_t, under_pressure) == 104, "pressure offset");

int main(void) {
  clap_cache_config_t config = {0};
  config.version = CLAP_CACHE_ABI_VERSION;
  config.struct_size = sizeof(config);
  return config.struct_size == sizeof(clap_cache_config_t) ? 0 : 1;
}
