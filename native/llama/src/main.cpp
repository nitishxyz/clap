#include "clap/llama/worker.h"

#include "llama.h"

int main() {
  ggml_backend_load_all();
  clap::llama::Worker worker;
  return worker.run();
}
