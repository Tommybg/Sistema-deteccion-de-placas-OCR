"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowRight, ScanLine, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";

export function Hero() {
  return (
    <section className="relative overflow-hidden">
      <div className="absolute inset-0 grid-bg opacity-30 pointer-events-none" />
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background:
            "radial-gradient(800px circle at 30% 20%, hsl(263 70% 60% / 0.15), transparent 60%), radial-gradient(600px circle at 70% 80%, hsl(192 70% 50% / 0.10), transparent 60%)",
        }}
      />

      <div className="relative container mx-auto px-6 pt-28 pb-20 lg:pt-36 lg:pb-28">
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: "easeOut" }}
          className="max-w-3xl"
        >
          <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium border border-white/10 bg-white/5 text-muted-foreground">
            <Sparkles className="w-3.5 h-3.5 text-brand" />
            Multi-model vehicle intelligence
          </span>
          <h1 className="mt-6 text-4xl md:text-6xl font-semibold tracking-tight leading-[1.05]">
            Intelligent vehicle recognition for{" "}
            <span className="bg-gradient-to-br from-brand via-violet-400 to-cyan-400 bg-clip-text text-transparent">
              smart cities
            </span>
            .
          </h1>
          <p className="mt-6 text-lg text-muted-foreground max-w-2xl">
            Detect, identify, and analyze every vehicle in frame — plate text,
            brand, color, and type — through a single ONNX-accelerated
            pipeline. Built for traffic agencies, security operators, and
            researchers.
          </p>

          <div className="mt-8 flex flex-wrap items-center gap-3">
            <Button asChild size="lg" className="animate-pulse-glow">
              <Link href="/dashboard">
                Open dashboard <ArrowRight className="w-4 h-4" />
              </Link>
            </Button>
            <Button asChild size="lg" variant="outline">
              <Link href="/history">
                <ScanLine className="w-4 h-4" />
                Browse detections
              </Link>
            </Button>
          </div>

          <dl className="mt-12 grid grid-cols-3 gap-6 max-w-lg text-sm">
            <Stat value="5" label="AI models" />
            <Stat value="30" label="Brand classes" />
            <Stat value="15" label="Color classes" />
          </dl>
        </motion.div>
      </div>
    </section>
  );
}

function Stat({ value, label }: { value: string; label: string }) {
  return (
    <div>
      <div className="text-3xl font-semibold tracking-tight text-foreground">{value}</div>
      <div className="text-xs uppercase tracking-widest text-muted-foreground">{label}</div>
    </div>
  );
}
