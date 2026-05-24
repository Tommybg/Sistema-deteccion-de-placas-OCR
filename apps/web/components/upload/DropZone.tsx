"use client";

import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { UploadCloud, ImageIcon } from "lucide-react";
import { cn } from "@/lib/utils";

interface Props {
  onFile: (file: File) => void;
  disabled?: boolean;
}

export function DropZone({ onFile, disabled }: Props) {
  const onDrop = useCallback(
    (accepted: File[]) => {
      const file = accepted[0];
      if (file) onFile(file);
    },
    [onFile],
  );

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: { "image/*": [".jpg", ".jpeg", ".png", ".bmp", ".webp"] },
    multiple: false,
    disabled,
    maxSize: 8 * 1024 * 1024,
  });

  return (
    <div
      {...getRootProps()}
      className={cn(
        "relative w-full aspect-video rounded-xl border-2 border-dashed flex flex-col items-center justify-center text-center p-6 transition-all cursor-pointer overflow-hidden",
        isDragActive && !isDragReject && "border-brand bg-brand/5",
        isDragReject && "border-destructive bg-destructive/10",
        !isDragActive && "border-white/10 bg-secondary/20 hover:border-white/20 hover:bg-secondary/30",
        disabled && "opacity-50 cursor-not-allowed",
      )}
    >
      <input {...getInputProps()} />
      <div className="absolute inset-0 grid-bg opacity-30 pointer-events-none" />
      <div className="relative w-12 h-12 mb-4 rounded-xl bg-brand/15 text-brand grid place-items-center ring-1 ring-brand/30">
        {isDragActive ? <ImageIcon className="w-6 h-6" /> : <UploadCloud className="w-6 h-6" />}
      </div>
      <p className="relative text-sm font-medium">
        {isDragActive ? "Drop image to detect" : "Drop or click to upload"}
      </p>
      <p className="relative text-xs text-muted-foreground mt-1">
        JPG, PNG, BMP, WebP • up to 8 MB
      </p>
      <p className="relative text-[10px] text-muted-foreground/70 mt-3 uppercase tracking-widest">
        Or paste from clipboard
      </p>
    </div>
  );
}
