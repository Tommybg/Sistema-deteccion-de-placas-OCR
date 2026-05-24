"use client";

import { motion } from "framer-motion";
import { Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { formatRelative } from "@/lib/utils";
import type { DetectionListItem } from "@/lib/types";

interface Props {
  items: DetectionListItem[];
  onDelete?: (id: string) => void;
}

export function HistoryTable({ items, onDelete }: Props) {
  if (items.length === 0) {
    return (
      <div className="rounded-xl border border-white/5 bg-card/40 backdrop-blur-xl p-12 text-center text-sm text-muted-foreground">
        No detections recorded yet.
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-white/5 bg-card/40 backdrop-blur-xl overflow-hidden">
      <div className="grid grid-cols-12 gap-2 px-4 py-3 text-[10px] uppercase tracking-widest text-muted-foreground border-b border-white/5 bg-secondary/20">
        <div className="col-span-3">When</div>
        <div className="col-span-2">Plate</div>
        <div className="col-span-2">Brand</div>
        <div className="col-span-1">Color</div>
        <div className="col-span-1">Source</div>
        <div className="col-span-2 text-right">Latency</div>
        <div className="col-span-1 text-right">Actions</div>
      </div>
      <ul>
        {items.map((row) => (
          <motion.li
            key={row.id}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="grid grid-cols-12 gap-2 items-center px-4 py-3 text-sm border-b border-white/5 last:border-b-0 hover:bg-white/[0.02]"
          >
            <div className="col-span-3">
              <div className="font-medium">{formatRelative(new Date(row.created_at))}</div>
              <div className="text-[11px] text-muted-foreground">
                {new Date(row.created_at).toLocaleString()}
              </div>
            </div>
            <div className="col-span-2 font-mono tabular-nums">
              {row.plate_text || <span className="text-muted-foreground">—</span>}
            </div>
            <div className="col-span-2">{row.brand || <span className="text-muted-foreground">—</span>}</div>
            <div className="col-span-1">{row.color || <span className="text-muted-foreground">—</span>}</div>
            <div className="col-span-1">
              <Badge variant="outline">{row.source}</Badge>
            </div>
            <div className="col-span-2 text-right tabular-nums text-muted-foreground">
              {row.latency_ms} ms
            </div>
            <div className="col-span-1 flex justify-end">
              {onDelete && (
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => onDelete(row.id)}
                  aria-label="Delete detection"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </Button>
              )}
            </div>
          </motion.li>
        ))}
      </ul>
    </div>
  );
}
