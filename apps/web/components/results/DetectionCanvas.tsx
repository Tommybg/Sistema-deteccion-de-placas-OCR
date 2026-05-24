"use client";

import { useEffect, useRef } from "react";
import { drawDetections } from "@/lib/draw";
import type { DetectionResult } from "@/lib/types";

interface Props {
  imageUrl: string;
  result: DetectionResult | null;
  showLabels?: boolean;
  showConfidence?: boolean;
}

export function DetectionCanvas({
  imageUrl,
  result,
  showLabels = true,
  showConfidence = true,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const img = new window.Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      imgRef.current = img;
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(img, 0, 0);
      if (result) drawDetections(ctx, result, { showLabels, showConfidence });
    };
    img.src = imageUrl;
  }, [imageUrl, result, showLabels, showConfidence]);

  return (
    <div className="relative w-full rounded-xl overflow-hidden border border-white/5 bg-black/40">
      <canvas ref={canvasRef} className="w-full h-auto block" />
    </div>
  );
}
