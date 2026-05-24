import Link from "next/link";
import { Hero } from "@/components/landing/Hero";
import { CapabilityGrid } from "@/components/landing/CapabilityGrid";
import { PipelineDiagram } from "@/components/landing/PipelineDiagram";

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-background">
      <nav className="fixed top-0 inset-x-0 z-40 border-b border-white/5 bg-background/60 backdrop-blur-xl">
        <div className="container mx-auto px-6 h-14 flex items-center justify-between">
          <Link href="/" className="font-semibold tracking-tight">
            ANPR <span className="text-muted-foreground">Vision</span>
          </Link>
          <div className="flex items-center gap-4 text-sm">
            <Link href="/dashboard" className="text-muted-foreground hover:text-foreground transition-colors">
              Dashboard
            </Link>
            <Link href="/history" className="text-muted-foreground hover:text-foreground transition-colors">
              History
            </Link>
            <Link
              href="/dashboard"
              className="px-3 py-1.5 rounded-md bg-brand text-brand-foreground text-xs font-medium hover:bg-brand/90 transition-colors"
            >
              Try the demo
            </Link>
          </div>
        </div>
      </nav>

      <Hero />
      <CapabilityGrid />
      <PipelineDiagram />

      <footer className="border-t border-white/5 py-6">
        <div className="container mx-auto px-6 text-xs text-muted-foreground flex items-center justify-between">
          <span>© ANPR Vision · Built on top of the Universidad de La Sabana AI Lab pipeline.</span>
          <span>Powered by Next.js 15 · FastAPI · YOLOv11n</span>
        </div>
      </footer>
    </div>
  );
}
