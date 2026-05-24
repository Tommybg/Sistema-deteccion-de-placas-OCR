"use client";

import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Download, RefreshCw, Search } from "lucide-react";
import { toast } from "sonner";
import { AppShell } from "@/components/shell/AppShell";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { HistoryTable } from "@/components/history/HistoryTable";
import { api } from "@/lib/api";

export default function HistoryPage() {
  const queryClient = useQueryClient();
  const [plate, setPlate] = useState("");
  const [brand, setBrand] = useState("");
  const [color, setColor] = useState("");

  const filters = useMemo(() => ({ plate, brand, color }), [plate, brand, color]);

  const { data, isLoading, refetch } = useQuery({
    queryKey: ["history", filters],
    queryFn: () => api.history({ limit: 100, ...filters }),
  });

  const del = useMutation({
    mutationFn: (id: string) => api.deleteHistory(id),
    onSuccess: () => {
      toast.success("Detection removed");
      queryClient.invalidateQueries({ queryKey: ["history"] });
    },
  });

  function exportFile(format: "csv" | "json") {
    const url = api.exportHistoryUrl(format, {
      plate: plate || undefined,
      brand: brand || undefined,
      color: color || undefined,
    });
    window.open(url, "_blank");
  }

  return (
    <AppShell
      title="Detection history"
      subtitle="Full audit trail of every detection run by this instance"
      actions={
        <>
          <Button size="sm" variant="outline" onClick={() => refetch()}>
            <RefreshCw className="w-3.5 h-3.5" /> Refresh
          </Button>
          <Button size="sm" variant="outline" onClick={() => exportFile("csv")}>
            <Download className="w-3.5 h-3.5" /> CSV
          </Button>
          <Button size="sm" onClick={() => exportFile("json")}>
            <Download className="w-3.5 h-3.5" /> JSON
          </Button>
        </>
      }
    >
      <div className="space-y-4">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm flex items-center gap-2">
              <Search className="w-4 h-4 text-muted-foreground" /> Filters
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
              <Input
                placeholder="Plate contains…"
                value={plate}
                onChange={(e) => setPlate(e.target.value)}
              />
              <Input
                placeholder="Brand exact match"
                value={brand}
                onChange={(e) => setBrand(e.target.value)}
              />
              <Input
                placeholder="Color exact match"
                value={color}
                onChange={(e) => setColor(e.target.value)}
              />
              <div className="flex items-center text-xs text-muted-foreground">
                {data && (
                  <Badge variant="outline">
                    {data.items.length} / {data.total} detections
                  </Badge>
                )}
              </div>
            </div>
          </CardContent>
        </Card>

        {isLoading ? (
          <div className="rounded-xl border border-white/5 bg-card/40 p-12 text-center text-sm text-muted-foreground">
            Loading detections…
          </div>
        ) : (
          <HistoryTable items={data?.items ?? []} onDelete={(id) => del.mutate(id)} />
        )}
      </div>
    </AppShell>
  );
}
