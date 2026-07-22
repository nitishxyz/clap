import Darwin

func main() async {
  guard #available(macOS 14.0, *) else {
    emit(error: "clap-mlx requires macOS 14 or newer on Apple Silicon")
    exit(2)
  }
  #if !arch(arm64)
  emit(error: "clap-mlx requires Apple Silicon arm64")
  exit(2)
  #endif

  await WorkerApplication().run()
}

await main()
