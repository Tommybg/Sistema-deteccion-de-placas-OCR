"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { toast } from "sonner";
import { AppShell } from "@/components/shell/AppShell";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { SourcePicker } from "@/components/dashboard/SourcePicker";
import { ConfidenceSlider } from "@/components/dashboard/ConfidenceSlider";
import { ModelInfoCard } from "@/components/dashboard/ModelInfoCard";
import { SamplePicker } from "@/components/dashboard/SamplePicker";
import { DropZone } from "@/components/upload/DropZone";
import { LiveCapture } from "@/components/webcam/LiveCapture";
import { DetectionCanvas } from "@/components/results/DetectionCanvas";
import { ResultsPanel } from "@/components/results/ResultsPanel";
import { DetectionTimeline } from "@/components/results/DetectionTimeline";
import { api } from "@/lib/api";
import { useDetection } from "@/hooks/useDetection";
import type { DetectionSource } from "@/lib/types";

type Source = Exclude<DetectionSource, "batch">;

export default function DashboardPage() {
  const [source, setSource] = useState<Source>("upload");
  const [confidence, setConfidence] = useState(0.5);
  const [webcamActive, setWebcamActive] = useState(false);
  const [selectedSample, setSelectedSample] = useState<string | null>(null);

  const { run, result, history, imageUrl, pending } = useDetection(confidence);

  const handleFile = useCallback(
    async (file: File) => {
      const r = await run(file, "upload");
      if (r) toast.success(`Detected ${r.plates.length} plate(s) in ${r.latency_ms} ms`);
    },
    [run],
  );

  const handleSample = useCallback(
    async (name: string) => {
      setSelectedSample(name);
      try {
        const blob = await api.fetchSample(name);
        const r = await run(blob, "sample");
        if (r) toast.success(`Detected ${r.plates.length} plate(s) in ${r.latency_ms} ms`);
      } catch (err) {
        toast.error("Could not load sample", { description: (err as Error).message });
      }
    },
    [run],
  );

  const handleFrame = useCallback(
    async (blob: Blob) => {
      await run(blob, "webcam");
    },
    [run],
  );

  // Switching away from webcam should stop the camera
  useEffect(() => {
    if (source !== "webcam") setWebcamActive(false);
  }, [source]);

  // Paste from clipboard
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      if (!e.clipboardData) return;
      for (const item of Array.from(e.clipboardData.items)) {
        if (item.type.startsWith("image/")) {
          const file = item.getAsFile();
          if (file) {
            setSource("upload");
            handleFile(file);
            return;
          }
        }
      }
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [handleFile]);

  return (
    <AppShell
      title="Detection Dashboard"
      subtitle="Upload, stream, or pick a sample to run the full inference pipeline"
      actions={pending ? <Badge variant="warning">Inferring…</Badge> : null}
    >
      <div className="grid grid-cols-12 gap-6">
        {/* Left rail */}
        <aside className="col-span-12 lg:col-span-3 space-y-4">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-xs uppercase tracking-widest text-muted-foreground">
                Source
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-5">
              <SourcePicker value={source} onChange={setSource} />
              <ConfidenceSlider value={confidence} onChange={setConfidence} />
            </CardContent>
          </Card>
          <ModelInfoCard />
        </aside>

        {/* Center: input + result canvas */}
        <section className="col-span-12 lg:col-span-6 space-y-4">
          <Card>
            <CardContent className="p-5">
              {source === "upload" && <DropZone onFile={handleFile} disabled={pending} />}
              {source === "webcam" && (
                <LiveCapture
                  active={webcamActive}
                  onToggle={setWebcamActive}
                  onFrame={handleFrame}
                />
              )}
              {source === "sample" && (
                <SamplePicker
                  onPick={handleSample}
                  selected={selectedSample ?? undefined}
                />
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm">Annotated frame</CardTitle>
                {result && (
                  <Badge variant="outline">
                    {result.vehicles.length} vehicle(s) · {result.plates.length} plate(s)
                  </Badge>
                )}
              </div>
            </CardHeader>
            <CardContent>
              {imageUrl ? (
                <motion.div
                  key={imageUrl + (result?.id ?? "")}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                >
                  <DetectionCanvas imageUrl={imageUrl} result={result} />
                </motion.div>
              ) : (
                <Skeleton className="aspect-video w-full" />
              )}
            </CardContent>
          </Card>
        </section>

        {/* Right rail: results */}
        <aside className="col-span-12 lg:col-span-3 space-y-4">
          <ResultsPanel result={result} />
          <DetectionTimeline entries={history} />
        </aside>
      </div>
    </AppShell>
  );
}
