"use client";

import { useCallback, useRef, useState } from "react";
import { toast } from "sonner";
import { api } from "@/lib/api";
import type { DetectionResult } from "@/lib/types";

export function useDetection(confidence: number) {
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [history, setHistory] = useState<DetectionResult[]>([]);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const lastUrlRef = useRef<string | null>(null);

  const run = useCallback(
    async (blob: Blob, source: "upload" | "webcam" | "sample") => {
      // Cancel any in-flight request so webcam latency stays bounded
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      setPending(true);
      const localUrl = URL.createObjectURL(blob);

      try {
        const detection = await api.detect(blob, {
          confidence,
          source,
          signal: controller.signal,
        });

        // Revoke the previous blob URL only after the new image loads
        if (lastUrlRef.current) URL.revokeObjectURL(lastUrlRef.current);
        lastUrlRef.current = localUrl;

        setImageUrl(localUrl);
        setResult(detection);
        setHistory((prev) => [detection, ...prev].slice(0, 20));
        return detection;
      } catch (err) {
        if ((err as Error).name === "AbortError") return null;
        URL.revokeObjectURL(localUrl);
        toast.error("Detection failed", { description: (err as Error).message });
        return null;
      } finally {
        setPending(false);
      }
    },
    [confidence],
  );

  return { run, result, history, imageUrl, pending };
}
