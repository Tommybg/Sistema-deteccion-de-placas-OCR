"use client";

import { motion } from "framer-motion";
import { Car, Palette, Tag, ScanLine } from "lucide-react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import type { DetectionResult } from "@/lib/types";

interface Props {
  result: DetectionResult | null;
}

export function ResultsPanel({ result }: Props) {
  if (!result) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-sm text-muted-foreground">No detection yet</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-xs text-muted-foreground">
            Upload an image, start the webcam, or pick a sample to see live results here.
          </p>
        </CardContent>
      </Card>
    );
  }

  const plates = result.plates;
  const noDetections = plates.length === 0 && result.vehicles.length === 0;

  return (
    <div className="space-y-3">
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm">Detection summary</CardTitle>
            <Badge variant="outline">{result.latency_ms} ms</Badge>
          </div>
        </CardHeader>
        <CardContent className="grid grid-cols-2 gap-3 text-sm">
          <Stat label="Vehicles" value={result.vehicles.length} />
          <Stat label="Plates" value={plates.length} />
          <Stat label="Resolution" value={`${result.image_width}×${result.image_height}`} small />
          <Stat label="Confidence" value={`≥ ${Math.round(result.confidence_used * 100)}%`} small />
        </CardContent>
      </Card>

      {noDetections && (
        <Card>
          <CardContent className="p-5 text-sm text-muted-foreground">
            No detections in this frame. Try lowering the confidence threshold.
          </CardContent>
        </Card>
      )}

      {plates.map((plate, idx) => (
        <motion.div
          key={idx}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: idx * 0.05 }}
        >
          <Card>
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="text-xs uppercase tracking-widest text-muted-foreground">
                  Detection #{idx + 1}
                </CardTitle>
                <Badge variant="success">{Math.round(plate.confidence * 100)}%</Badge>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex justify-center">
                <span className="plate-pill">{plate.text || "—"}</span>
              </div>
              <Detail icon={ScanLine} label="Plate confidence">
                <Progress value={plate.confidence * 100} className="w-24" />
                <span className="text-xs font-medium tabular-nums w-10 text-right">
                  {Math.round(plate.confidence * 100)}%
                </span>
              </Detail>
              {plate.vehicle_type && (
                <Detail icon={Car} label="Vehicle type">
                  <span className="text-sm font-medium">{plate.vehicle_type}</span>
                </Detail>
              )}
              {plate.vehicle_color && (
                <Detail icon={Palette} label="Color">
                  <span className="text-sm font-medium">{plate.vehicle_color}</span>
                  {plate.vehicle_color_confidence !== null && (
                    <span className="text-xs text-muted-foreground tabular-nums">
                      {Math.round((plate.vehicle_color_confidence ?? 0) * 100)}%
                    </span>
                  )}
                </Detail>
              )}
              {plate.brand && (
                <Detail icon={Tag} label="Brand">
                  <span className="text-sm font-medium">{plate.brand}</span>
                </Detail>
              )}
            </CardContent>
          </Card>
        </motion.div>
      ))}
    </div>
  );
}

function Stat({ label, value, small }: { label: string; value: React.ReactNode; small?: boolean }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-widest text-muted-foreground">{label}</div>
      <div className={small ? "text-sm font-medium" : "text-2xl font-semibold tabular-nums"}>
        {value}
      </div>
    </div>
  );
}

function Detail({
  icon: Icon,
  label,
  children,
}: {
  icon: typeof Car;
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex items-center justify-between gap-3">
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <Icon className="w-3.5 h-3.5" />
        <span>{label}</span>
      </div>
      <div className="flex items-center gap-2 min-w-0">{children}</div>
    </div>
  );
}
