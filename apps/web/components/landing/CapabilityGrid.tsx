"use client";

import { motion } from "framer-motion";
import { ScanLine, Tag, Palette, Car } from "lucide-react";

const items = [
  {
    icon: ScanLine,
    title: "License plate OCR",
    description:
      "Custom-trained YOLOv11n locates plates; fast-plate-ocr reads them on CPU in milliseconds.",
    accent: "from-emerald-500/20 to-emerald-500/0",
    color: "text-emerald-400",
  },
  {
    icon: Tag,
    title: "Brand classification",
    description: "30-class brand detector — Audi to Volvo — with Colombia-aware filtering.",
    accent: "from-orange-500/20 to-orange-500/0",
    color: "text-orange-400",
  },
  {
    icon: Palette,
    title: "Color classification",
    description: "EfficientNetB0 quantized to INT8 TFLite for 15 color classes per vehicle crop.",
    accent: "from-pink-500/20 to-pink-500/0",
    color: "text-pink-400",
  },
  {
    icon: Car,
    title: "Vehicle type",
    description: "COCO-pretrained YOLOv11n distinguishes cars, motorcycles, buses, and trucks.",
    accent: "from-blue-500/20 to-blue-500/0",
    color: "text-blue-400",
  },
];

export function CapabilityGrid() {
  return (
    <section className="container mx-auto px-6 pb-24">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {items.map((it, i) => (
          <motion.div
            key={it.title}
            initial={{ opacity: 0, y: 12 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: i * 0.07 }}
            className="relative overflow-hidden rounded-xl border border-white/5 bg-card/40 backdrop-blur-xl p-5 group"
          >
            <div
              className={`absolute -inset-px rounded-xl bg-gradient-to-br ${it.accent} opacity-60 pointer-events-none`}
            />
            <div className="relative">
              <div className="w-10 h-10 rounded-lg bg-white/5 grid place-items-center mb-4">
                <it.icon className={`w-5 h-5 ${it.color}`} />
              </div>
              <h3 className="font-semibold tracking-tight">{it.title}</h3>
              <p className="text-sm text-muted-foreground mt-1">{it.description}</p>
            </div>
          </motion.div>
        ))}
      </div>
    </section>
  );
}
