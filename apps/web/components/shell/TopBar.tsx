"use client";

import { useQuery } from "@tanstack/react-query";
import { Activity, CheckCircle2, XCircle } from "lucide-react";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";

interface Props {
  title: string;
  subtitle?: string;
  actions?: React.ReactNode;
}

export function TopBar({ title, subtitle, actions }: Props) {
  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 15_000,
  });

  const ok = health?.status === "ok" && health.models_ready;

  return (
    <header className="h-16 px-4 md:px-8 flex items-center justify-between border-b border-white/5 bg-background/60 backdrop-blur-xl sticky top-0 z-30">
      <div>
        <h1 className="text-lg font-semibold tracking-tight leading-tight">{title}</h1>
        {subtitle && (
          <p className="text-xs text-muted-foreground leading-tight">{subtitle}</p>
        )}
      </div>
      <div className="flex items-center gap-3">
        {actions}
        <div
          className={cn(
            "flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs border",
            ok
              ? "border-success/30 bg-success/10 text-success"
              : "border-destructive/30 bg-destructive/10 text-destructive",
          )}
        >
          {ok ? <CheckCircle2 className="w-3.5 h-3.5" /> : <XCircle className="w-3.5 h-3.5" />}
          <span className="font-medium">{ok ? "Models online" : "Models offline"}</span>
          <Activity className={cn("w-3 h-3", ok && "animate-pulse")} />
        </div>
      </div>
    </header>
  );
}
