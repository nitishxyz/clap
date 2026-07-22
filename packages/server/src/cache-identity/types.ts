export const INSTALLATION_SECRET_SCHEMA_VERSION = 1 as const;
export const INSTALLATION_SECRET_BYTE_LENGTH = 32;

export type InstallationSecret = {
  generation: string;
  key: Uint8Array;
};

export type InstallationSecretDocumentV1 = {
  version: typeof INSTALLATION_SECRET_SCHEMA_VERSION;
  generation: string;
  createdAt: string;
  secret: string;
};

export type InstallationSecretRotationOptions = {
  /** Test seam for exercising failure after the replacement has been persisted. */
  rename?: (oldPath: string, newPath: string) => Promise<void>;
};
