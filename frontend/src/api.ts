// Talks to the FastAPI backend (api.py), hosted on a free Hugging Face Space that sleeps when idle.

export type Verdict = { label: "fake" | "real"; confidence: number; explanation: string };

export const API_URL =
  import.meta.env.VITE_API_URL ?? (import.meta.env.DEV ? "http://localhost:8000" : "https://ramadhanzome-deepfake-rag.hf.space");

/** Ping the health endpoint so a sleeping Space starts waking before the upload. Resolves either way. */
export async function wake(timeoutMs = 90_000): Promise<boolean> {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${API_URL}/`, { signal: ctrl.signal });
    return res.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(t);
  }
}

export async function analyse(file: File): Promise<Verdict> {
  const body = new FormData();
  body.append("file", file);
  const res = await fetch(`${API_URL}/predict`, { method: "POST", body });
  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new Error(detail?.detail ?? `The analysis server hit an error (${res.status}). Try again in a little while.`);
  }
  return res.json();
}
