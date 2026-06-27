export type PullCancellationHandlerOptions = {
  cancel: () => Promise<void>;
  write?: (message: string) => void;
  exit?: (code: number) => never;
};

export function createPullCancellationHandler(options: PullCancellationHandlerOptions): () => void {
  let cancelling = false;
  const write = options.write ?? ((message) => process.stderr.write(message));
  const exit = options.exit ?? ((code) => process.exit(code));
  return () => {
    if (cancelling) return;
    cancelling = true;
    void (async () => {
      try {
        await options.cancel();
        write("\nclap: pull cancelled\n");
      } finally {
        exit(130);
      }
    })();
  };
}

export function installPullCancellationHandler(cancel: () => Promise<void>): () => void {
  const handler = createPullCancellationHandler({ cancel });
  process.once("SIGINT", handler);
  return () => process.off("SIGINT", handler);
}
