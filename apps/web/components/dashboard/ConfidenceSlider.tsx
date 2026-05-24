"use client";

import { Slider } from "@/components/ui/slider";
import { Gauge } from "lucide-react";

interface Props {
  value: number;
  onChange: (v: number) => void;
}

export function ConfidenceSlider({ value, onChange }: Props) {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <Gauge className="w-3.5 h-3.5" />
          <span>Confidence threshold</span>
        </div>
        <span className="text-sm font-medium tabular-nums">{Math.round(value * 100)}%</span>
      </div>
      <Slider
        value={[value]}
        min={0.1}
        max={0.95}
        step={0.05}
        onValueChange={(v) => onChange(v[0] ?? value)}
      />
    </div>
  );
}
