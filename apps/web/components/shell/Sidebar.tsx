"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Car, LayoutDashboard, History, Github, Cpu } from "lucide-react";
import { cn } from "@/lib/utils";

const nav = [
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/history", label: "History", icon: History },
];

export function Sidebar() {
  const path = usePathname();
  return (
    <aside className="hidden md:flex flex-col w-60 shrink-0 border-r border-white/5 bg-card/30 backdrop-blur-xl">
      <div className="h-16 flex items-center gap-2 px-5 border-b border-white/5">
        <div className="w-9 h-9 rounded-lg bg-brand/15 text-brand grid place-items-center ring-1 ring-brand/30">
          <Car className="w-5 h-5" />
        </div>
        <div>
          <div className="font-semibold tracking-tight leading-none">ANPR Vision</div>
          <div className="text-[10px] uppercase tracking-widest text-muted-foreground">Vehicle AI</div>
        </div>
      </div>

      <nav className="flex-1 p-3 space-y-1">
        {nav.map((item) => {
          const active = path === item.href || path?.startsWith(item.href + "/");
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors",
                active
                  ? "bg-brand/15 text-brand"
                  : "text-muted-foreground hover:text-foreground hover:bg-white/5",
              )}
            >
              <item.icon className="w-4 h-4" />
              {item.label}
            </Link>
          );
        })}
      </nav>

      <div className="p-3 border-t border-white/5 space-y-2">
        <div className="flex items-center gap-2 px-3 py-2 text-xs text-muted-foreground">
          <Cpu className="w-3.5 h-3.5" />
          <span>CPU inference</span>
        </div>
        <Link
          href="https://github.com"
          target="_blank"
          rel="noreferrer"
          className="flex items-center gap-2 px-3 py-2 text-xs text-muted-foreground hover:text-foreground"
        >
          <Github className="w-3.5 h-3.5" />
          Source code
        </Link>
      </div>
    </aside>
  );
}
