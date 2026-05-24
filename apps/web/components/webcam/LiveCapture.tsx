"use client";

import { useEffect, useRef, useState } from "react";
import { Power, Camera } from "lucide-react";
import { Button } from "@/components/ui/button";

interface Props {
  active: boolean;
  onToggle: (next: boolean) => void;
  onFrame: (blob: Blob) => void;
  intervalMs?: number;
}

export function LiveCapture({ active, onToggle, onFrame, intervalMs = 700 }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const intervalRef = useRef<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [streaming, setStreaming] = useState(false);

  useEffect(() => {
    if (!active) {
      stop();
      return;
    }
    let cancelled = false;

    (async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        const video = videoRef.current!;
        video.srcObject = stream;
        await video.play();
        setStreaming(true);
        setError(null);

        intervalRef.current = window.setInterval(captureFrame, intervalMs);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Camera access denied");
        onToggle(false);
      }
    })();

    return () => {
      cancelled = true;
      stop();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active]);

  function stop() {
    if (intervalRef.current) {
      window.clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    const video = videoRef.current;
    if (video?.srcObject instanceof MediaStream) {
      video.srcObject.getTracks().forEach((t) => t.stop());
      video.srcObject = null;
    }
    setStreaming(false);
  }

  function captureFrame() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.videoWidth === 0) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0);
    canvas.toBlob(
      (blob) => {
        if (blob) onFrame(blob);
      },
      "image/jpeg",
      0.85,
    );
  }

  return (
    <div className="space-y-3">
      <div className="relative aspect-video rounded-xl overflow-hidden bg-black border border-white/5">
        <video
          ref={videoRef}
          className="w-full h-full object-contain"
          playsInline
          muted
        />
        <canvas ref={canvasRef} className="hidden" />
        {!streaming && (
          <div className="absolute inset-0 grid place-items-center text-muted-foreground">
            <div className="flex flex-col items-center gap-2">
              <Camera className="w-8 h-8" />
              <p className="text-sm">Camera idle</p>
            </div>
          </div>
        )}
        {streaming && (
          <div className="absolute top-3 left-3 flex items-center gap-1.5 px-2 py-1 rounded-full bg-destructive/90 text-destructive-foreground text-xs font-medium">
            <span className="w-1.5 h-1.5 rounded-full bg-white animate-pulse" /> LIVE
          </div>
        )}
      </div>
      <div className="flex items-center justify-between">
        <Button
          onClick={() => onToggle(!active)}
          variant={active ? "destructive" : "default"}
          size="sm"
        >
          <Power className="w-4 h-4" />
          {active ? "Stop camera" : "Start camera"}
        </Button>
        {error && <span className="text-xs text-destructive">{error}</span>}
      </div>
    </div>
  );
}
