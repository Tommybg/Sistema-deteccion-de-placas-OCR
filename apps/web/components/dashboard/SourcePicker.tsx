"use client";

import { Upload, Camera, Images } from "lucide-react";
import { cn } from "@/lib/utils";
import type { DetectionSource } from "@/lib/types";

type Source = Exclude<DetectionSource, "batch">;

interface Props {
  value: Source;
  onChange: (next: Source) => void;
}

const SOURCES: { id: Source; label: string; icon: typeof Upload; description: string }[] = [
  { id: "upload", label: "Upload", icon: Upload, description: "Drop or pick a file" },
  { id: "webcam", label: "Webcam", icon: Camera, description: "Continuous live capture" },
  { id: "sample", label: "Samples", icon: Images, description: "Pre-loaded reference set" },
];

export function SourcePicker({ value, onChange }: Props) {
  return (
    <div className="grid grid-cols-3 gap-2">
      {SOURCES.map((s) => {
        const active = value === s.id;
        return (
          <button
            key={s.id}
            onClick={() => onChange(s.id)}
            className={cn(
              "flex flex-col items-center gap-1.5 rounded-lg border p-3 text-xs transition-all",
              active
                ? "border-brand/40 bg-brand/10 text-foreground"
                : "border-white/5 bg-secondary/20 text-muted-foreground hover:text-foreground hover:bg-white/5",
            )}
          >
            <s.icon className={cn("w-4 h-4", active && "text-brand")} />
            <span className="font-medium">{s.label}</span>
          </button>
        );
      })}
    </div>
  );
}
