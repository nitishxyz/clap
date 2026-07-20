# Policy traces

Backend-independent scenarios live in the core integration tests today because
they can assert leases and generations directly. `scenarios.json` is the stable,
content-free trace vocabulary intended for production decision replay. Native
adapter conformance can consume the same fields when C++ and Swift integration
is added.

Token values in checked-in traces are synthetic. Production trace export must
hash namespace and identity values and must not include raw prompt tokens.
