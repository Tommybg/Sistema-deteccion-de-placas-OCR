// Pure canvas drawing helpers for detection overlays.
// Colors mirror app_demo.py:162,175,209.

import type { DetectionResult } from "@/lib/types";

export interface DrawOptions {
  showLabels?: boolean;
  showConfidence?: boolean;
  lineWidth?: number;
  scale?: number;
}

const COLORS = {
  vehicle: "#3b82f6",  // blue-500
  plate: "#10b981",    // emerald-500
  brand: "#f97316",    // orange-500
} as const;

export function drawDetections(
  ctx: CanvasRenderingContext2D,
  result: DetectionResult,
  options: DrawOptions = {},
): void {
  const {
    showLabels = true,
    showConfidence = true,
    lineWidth = 3,
    scale = 1,
  } = options;

  ctx.lineWidth = lineWidth;
  ctx.font = `${Math.max(11, 14 * scale)}px ui-sans-serif, system-ui`;
  ctx.textBaseline = "top";

  for (const v of result.vehicles) {
    drawBox(ctx, v.bbox, COLORS.vehicle, scale, lineWidth);
    if (showLabels) {
      const parts: string[] = [v.type];
      if (showConfidence) parts.push(`${Math.round(v.type_confidence * 100)}%`);
      if (v.color) parts.push(v.color);
      if (v.brand) parts.push(v.brand);
      drawLabel(ctx, parts.join(" • "), v.bbox, COLORS.vehicle, "#fff");
    }
  }

  for (const p of result.plates) {
    drawBox(ctx, p.bbox, COLORS.plate, scale, lineWidth);
    if (showLabels) {
      const parts: string[] = [];
      if (p.text) parts.push(p.text);
      else parts.push("Placa");
      if (showConfidence) parts.push(`${Math.round(p.confidence * 100)}%`);
      drawLabel(ctx, parts.join(" • "), p.bbox, COLORS.plate, "#0a0a0a");
    }
  }
}

function drawBox(
  ctx: CanvasRenderingContext2D,
  bbox: [number, number, number, number],
  color: string,
  scale: number,
  lineWidth: number,
): void {
  const [x1, y1, x2, y2] = bbox;
  ctx.strokeStyle = color;
  ctx.shadowColor = color;
  ctx.shadowBlur = 8 * scale;
  ctx.lineWidth = lineWidth;
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  ctx.shadowBlur = 0;
}

function drawLabel(
  ctx: CanvasRenderingContext2D,
  text: string,
  bbox: [number, number, number, number],
  bg: string,
  fg: string,
): void {
  const [x1, y1] = bbox;
  const padX = 6;
  const padY = 4;
  const metrics = ctx.measureText(text);
  const w = metrics.width + padX * 2;
  const h = parseInt(ctx.font, 10) + padY * 2;
  const y = Math.max(0, y1 - h);
  ctx.fillStyle = bg;
  ctx.fillRect(x1, y, w, h);
  ctx.fillStyle = fg;
  ctx.fillText(text, x1 + padX, y + padY);
}

export function fitCanvas(
  canvas: HTMLCanvasElement,
  image: HTMLImageElement | HTMLVideoElement,
  maxWidth?: number,
): { scale: number; width: number; height: number } {
  const w = (image as HTMLImageElement).naturalWidth ?? (image as HTMLVideoElement).videoWidth;
  const h = (image as HTMLImageElement).naturalHeight ?? (image as HTMLVideoElement).videoHeight;
  const targetWidth = maxWidth && maxWidth < w ? maxWidth : w;
  const scale = targetWidth / w;
  canvas.width = w;
  canvas.height = h;
  return { scale, width: w, height: h };
}
