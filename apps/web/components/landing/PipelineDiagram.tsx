"use client";

import { motion } from "framer-motion";
import { ArrowRight, Camera, Car, Tag, Palette, ScanLine, FileJson } from "lucide-react";

const stages = [
  { icon: Camera, label: "Frame" },
  { icon: Car, label: "Vehicles" },
  { icon: Tag, label: "Brand" },
  { icon: ScanLine, label: "Plate" },
  { icon: Palette, label: "Color" },
  { icon: FileJson, label: "JSON" },
];

export function PipelineDiagram() {
  return (
    <section className="container mx-auto px-6 pb-24">
      <div className="rounded-2xl border border-white/5 bg-card/40 backdrop-blur-xl p-6 md:p-10">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-xl font-semibold tracking-tight">Single-pass inference pipeline</h2>
            <p className="text-sm text-muted-foreground mt-1">
              All five models execute in sequence on a single Railway CPU instance — under one second per frame.
            </p>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          {stages.map((s, i) => (
            <div key={s.label} className="flex items-center gap-3">
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.08 }}
                className="flex items-center gap-2 px-4 py-2 rounded-full border border-white/10 bg-secondary/40 text-sm"
              >
                <s.icon className="w-4 h-4 text-brand" />
                <span>{s.label}</span>
              </motion.div>
              {i < stages.length - 1 && (
                <ArrowRight className="w-4 h-4 text-muted-foreground" />
              )}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
