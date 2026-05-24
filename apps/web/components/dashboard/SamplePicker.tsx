"use client";

import { useQuery } from "@tanstack/react-query";
import Image from "next/image";
import { api, API_BASE } from "@/lib/api";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

interface Props {
  onPick: (name: string) => void;
  selected?: string;
}

export function SamplePicker({ onPick, selected }: Props) {
  const { data, isLoading } = useQuery({
    queryKey: ["samples"],
    queryFn: api.samples,
  });

  if (isLoading) {
    return (
      <div className="grid grid-cols-3 gap-2">
        {Array.from({ length: 9 }).map((_, i) => (
          <Skeleton key={i} className="aspect-video" />
        ))}
      </div>
    );
  }

  const items = (data ?? []).slice(0, 12);

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 max-h-[420px] overflow-y-auto pr-1">
      {items.map((s) => (
        <button
          key={s.name}
          onClick={() => onPick(s.name)}
          className={cn(
            "relative aspect-video rounded-md overflow-hidden border transition-all",
            selected === s.name
              ? "border-brand ring-2 ring-brand/30"
              : "border-white/5 hover:border-white/20",
          )}
        >
          <Image
            src={`${API_BASE}${s.url}`}
            alt={s.name}
            fill
            sizes="200px"
            unoptimized
            className="object-cover"
          />
        </button>
      ))}
      {items.length === 0 && (
        <p className="col-span-full text-sm text-muted-foreground py-8 text-center">
          No samples available
        </p>
      )}
    </div>
  );
}
