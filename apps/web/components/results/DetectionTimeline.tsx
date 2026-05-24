"use client";

import { motion, AnimatePresence } from "framer-motion";
import { Clock, Car } from "lucide-react";
import type { DetectionResult } from "@/lib/types";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

interface Props {
  entries: DetectionResult[];
}

export function DetectionTimeline({ entries }: Props) {
  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm">Session timeline</CardTitle>
          <span className="text-xs text-muted-foreground">{entries.length}</span>
        </div>
      </CardHeader>
      <CardContent>
        {entries.length === 0 ? (
          <p className="text-xs text-muted-foreground">Run some detections to populate the timeline.</p>
        ) : (
          <ol className="space-y-2 max-h-64 overflow-y-auto pr-1">
            <AnimatePresence initial={false}>
              {entries.map((e) => {
                const text = e.plates.find((p) => p.text)?.text;
                const brand = e.vehicles.find((v) => v.brand)?.brand;
                return (
                  <motion.li
                    key={e.id}
                    initial={{ opacity: 0, x: -8 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0 }}
                    className="flex items-center gap-3 text-xs"
                  >
                    <div className="w-7 h-7 rounded-md bg-secondary/40 grid place-items-center">
                      <Car className="w-3.5 h-3.5 text-muted-foreground" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="font-mono font-medium truncate">{text || "—"}</span>
                        {brand && <span className="text-muted-foreground">· {brand}</span>}
                      </div>
                      <div className="flex items-center gap-2 text-muted-foreground">
                        <Clock className="w-3 h-3" />
                        <time>{new Date(e.timestamp).toLocaleTimeString()}</time>
                        <span>· {e.latency_ms} ms</span>
                      </div>
                    </div>
                  </motion.li>
                );
              })}
            </AnimatePresence>
          </ol>
        )}
      </CardContent>
    </Card>
  );
}
