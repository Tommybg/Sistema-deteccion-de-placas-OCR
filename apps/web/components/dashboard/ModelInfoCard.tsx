"use client";

import { useQuery } from "@tanstack/react-query";
import { Cpu, CheckCircle2 } from "lucide-react";
import { api } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

export function ModelInfoCard() {
  const { data, isLoading } = useQuery({
    queryKey: ["models"],
    queryFn: api.models,
  });

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-xs uppercase tracking-widest text-muted-foreground flex items-center gap-2">
          <Cpu className="w-3.5 h-3.5" /> Inference stack
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        {isLoading && (
          <>
            <Skeleton className="h-4 w-3/4" />
            <Skeleton className="h-4 w-2/3" />
            <Skeleton className="h-4 w-3/4" />
          </>
        )}
        {data?.models.map((m) => (
          <div key={m.name} className="flex items-center justify-between text-xs">
            <div>
              <div className="font-medium">{m.name}</div>
              <div className="text-muted-foreground">{m.version}</div>
            </div>
            {m.loaded && <CheckCircle2 className="w-3.5 h-3.5 text-success" />}
          </div>
        ))}
        {data && (
          <div className="pt-2 mt-2 border-t border-white/5 text-[10px] uppercase tracking-widest text-muted-foreground">
            Device: {data.device}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
