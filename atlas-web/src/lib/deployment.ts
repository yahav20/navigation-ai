// Deployment defaults: fall back to the local LangGraph dev server so the app
// connects out of the box even when no .env is present. Override via
// NEXT_PUBLIC_API_URL / NEXT_PUBLIC_ASSISTANT_ID (or ?apiUrl= / ?assistantId=).
export const DEFAULT_API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:2024";
export const DEFAULT_ASSISTANT_ID =
  process.env.NEXT_PUBLIC_ASSISTANT_ID || "agent";
