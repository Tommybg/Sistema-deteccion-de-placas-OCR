import type {
  DetectionList,
  DetectionResult,
  HealthResponse,
  ModelsResponse,
  SampleInfo,
} from "@/lib/types";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function jsonRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => jsonRequest<HealthResponse>("/api/health"),
  models: () => jsonRequest<ModelsResponse>("/api/models"),

  detect: async (
    file: Blob,
    options: { confidence?: number; source?: string; signal?: AbortSignal } = {},
  ): Promise<DetectionResult> => {
    const fd = new FormData();
    fd.append("image", file, "image.jpg");
    fd.append("confidence", String(options.confidence ?? 0.5));
    fd.append("source", options.source ?? "upload");
    const res = await fetch(`${API_URL}/api/detect`, {
      method: "POST",
      body: fd,
      signal: options.signal,
    });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`${res.status} ${res.statusText}: ${text}`);
    }
    return (await res.json()) as DetectionResult;
  },

  detectBatch: async (
    files: File[],
    confidence: number = 0.5,
  ): Promise<DetectionResult[]> => {
    const fd = new FormData();
    files.forEach((f) => fd.append("images", f, f.name));
    fd.append("confidence", String(confidence));
    const res = await fetch(`${API_URL}/api/detect/batch`, {
      method: "POST",
      body: fd,
    });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`${res.status} ${res.statusText}: ${text}`);
    }
    return (await res.json()) as DetectionResult[];
  },

  samples: () => jsonRequest<SampleInfo[]>("/api/samples"),
  sampleUrl: (name: string) => `${API_URL}/api/samples/${encodeURIComponent(name)}`,
  fetchSample: async (name: string): Promise<Blob> => {
    const res = await fetch(`${API_URL}/api/samples/${encodeURIComponent(name)}`);
    if (!res.ok) throw new Error(`Sample ${name} not found`);
    return await res.blob();
  },

  history: (params: {
    limit?: number;
    offset?: number;
    plate?: string;
    brand?: string;
    color?: string;
    from?: string;
    to?: string;
  } = {}): Promise<DetectionList> => {
    const qs = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== null && v !== "") qs.append(k, String(v));
    });
    return jsonRequest<DetectionList>(`/api/history?${qs.toString()}`);
  },

  exportHistoryUrl: (format: "csv" | "json", params: Record<string, string | undefined> = {}) => {
    const qs = new URLSearchParams({ format });
    Object.entries(params).forEach(([k, v]) => {
      if (v) qs.append(k, v);
    });
    return `${API_URL}/api/history/export?${qs.toString()}`;
  },

  deleteHistory: (id: string) =>
    jsonRequest<{ ok: boolean }>(`/api/history/${id}`, { method: "DELETE" }),
};

export const API_BASE = API_URL;
