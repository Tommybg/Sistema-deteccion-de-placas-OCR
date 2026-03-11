#!/usr/bin/env python3
"""
================================================================================
CORAL SIMULATOR
================================================================================
Capa de simulación Python para Google Coral Edge TPU.

Provee una API idéntica a `pycoral` para que el código de inferencia sea
intercambiable entre Coral real y simulación CPU (TFLite fallback).

Uso básico:
    from scripts.coral_simulator import CoralInterpreter, CoralBackend

    interp = CoralInterpreter("model_edgetpu.tflite")  # auto-detecta Coral
    interp.set_input_from_image(image_rgb)
    result = interp.invoke()
    print(result.latency_ms, result.estimated_tpu_ms)

Comparativa de backends:
    ┌─────────────┬──────────────────────────────────────────────────┐
    │ Backend     │ Descripción                                      │
    ├─────────────┼──────────────────────────────────────────────────┤
    │ coral       │ Coral real vía pycoral + libedgetpu              │
    │ cpu         │ TFLite intérprete estándar (simulación funcional)│
    │ auto        │ Coral si está conectado, CPU si no               │
    └─────────────┴──────────────────────────────────────────────────┘

Sobre los simuladores de instrucciones (RISC-V):
    Google también ofrece simuladores de bajo nivel para el procesador
    RISC-V del Coral NPU:
      • MPACT-CoralNPU (behavioral): github.com/google-coral/coralnpu-mpact
      • Verilator cycle-accurate:    github.com/google-coral/coralnpu
    Esos simuladores aceptan archivos .elf/.bin y son para desarrollo de
    hardware/compiladores, NO para desarrollo ML en Python. CoralSimulator
    usa TFLite CPU fallback, que da resultados idénticos con menor complejidad.

================================================================================
"""

from __future__ import annotations

import time
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from enum import Enum, auto

import numpy as np


# ─── Backend enum ──────────────────────────────────────────────────────────────

class CoralBackend(Enum):
    AUTO   = auto()
    CORAL  = auto()
    CPU    = auto()


# ─── Resultado de inferencia ───────────────────────────────────────────────────

@dataclass
class InferenceResult:
    """Resultado de una invocación de inferencia."""

    raw_output: list[np.ndarray]
    """Tensores de salida crudos del modelo."""

    latency_ms: float = 0.0
    """Latencia real medida en CPU (ms)."""

    backend: str = "cpu"
    """Backend utilizado: 'coral' o 'cpu'."""

    # Factores de escala estimados basados en benchmarks públicos:
    #   YOLOn  640px INT8:  CPU ARM ~400ms, Coral ~35ms → factor ~11x
    #   MobileNet INT8:     CPU ARM ~100ms, Coral ~2ms  → factor ~50x
    _SCALE_FACTORS: dict = field(default_factory=lambda: {
        "yolo":    11.0,   # YOLO nano-class
        "mobile":   8.0,   # MobileNet-class
        "efficient": 6.0,  # EfficientNet-class
        "default":  10.0,
    }, repr=False)

    model_hint: str = "default"
    """Pista del tipo de modelo para calcular speedup."""

    @property
    def estimated_tpu_ms(self) -> float:
        """Latencia estimada si corriera en Coral Edge TPU real."""
        scale = {
            "yolo":     11.0,
            "mobile":    8.0,
            "efficient": 6.0,
        }.get(self.model_hint, 10.0)
        return self.latency_ms / scale

    @property
    def speedup_factor(self) -> float:
        """Factor de aceleración estimado vs Coral real."""
        return {
            "yolo":     11.0,
            "mobile":    8.0,
            "efficient": 6.0,
        }.get(self.model_hint, 10.0)

    def __str__(self) -> str:
        backend_label = "🪸 Coral real" if self.backend == "coral" else "🖥️  CPU (sim)"
        lines = [
            f"  Backend      : {backend_label}",
            f"  Latencia real: {self.latency_ms:.1f} ms",
        ]
        if self.backend == "cpu":
            lines += [
                f"  Est. TPU     : {self.estimated_tpu_ms:.1f} ms  (~{self.speedup_factor:.0f}x más rápido en Coral)",
            ]
        lines.append(f"  Outputs      : {len(self.raw_output)} tensor(s)")
        return "\n".join(lines)


# ─── CoralInterpreter ─────────────────────────────────────────────────────────

class CoralInterpreter:
    """
    Intérprete unificado Coral / TFLite CPU.

    Imita la API de pycoral para que el código sea intercambiable:

        interp = CoralInterpreter("model_edgetpu.tflite", device="auto")
        interp.set_input_from_array(img_array_nhwc)
        result = interp.invoke()
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str | CoralBackend = "auto",
        num_threads: int = 4,
        verbose: bool = False,
    ):
        self.model_path = Path(model_path)
        self.num_threads = num_threads
        self.verbose = verbose
        self._backend: CoralBackend
        self._interp = None
        self._pycoral_interp = None

        # Normalizar backend
        if isinstance(device, str):
            device = {"auto": CoralBackend.AUTO,
                      "coral": CoralBackend.CORAL,
                      "cpu": CoralBackend.CPU}.get(device.lower(), CoralBackend.AUTO)
        self._requested_backend = device

        self._init_interpreter()

    # ── Inicialización ────────────────────────────────────────────────────────

    def _init_interpreter(self):
        """Inicializa el intérprete según backend solicitado."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {self.model_path}")

        if self._requested_backend in (CoralBackend.CORAL, CoralBackend.AUTO):
            coral_ok = self._try_init_coral()
            if coral_ok:
                self._backend = CoralBackend.CORAL
                return
            if self._requested_backend == CoralBackend.CORAL:
                raise RuntimeError(
                    "Se solicitó backend Coral pero no hay hardware disponible.\n"
                    "Usa device='auto' o device='cpu' para fallback a simulación."
                )

        # Fallback a CPU (TFLite estándar)
        self._init_cpu()
        self._backend = CoralBackend.CPU

    def _try_init_coral(self) -> bool:
        """Intenta inicializar el backend Coral real. Retorna True si exitoso."""
        try:
            from pycoral.utils.edgetpu import make_interpreter as _make
            self._pycoral_interp = _make(str(self.model_path))
            self._pycoral_interp.allocate_tensors()
            self._interp = self._pycoral_interp
            if self.verbose:
                print("🪸 Backend: Coral Edge TPU real (pycoral)")
            return True
        except ImportError:
            if self.verbose:
                print("⚠️  pycoral no instalado — usando simulación CPU")
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Coral no disponible ({e}) — usando simulación CPU")
        return False

    def _init_cpu(self):
        """Inicializa intérprete TFLite estándar (simulación funcional)."""
        import tensorflow as tf
        self._interp = tf.lite.Interpreter(
            model_path=str(self.model_path),
            num_threads=self.num_threads,
        )
        self._interp.allocate_tensors()
        if self.verbose:
            print(f"🖥️  Backend: TFLite CPU (simulación — {self.num_threads} threads)")

    # ── Propiedades del modelo ────────────────────────────────────────────────

    @property
    def backend(self) -> str:
        return "coral" if self._backend == CoralBackend.CORAL else "cpu"

    @property
    def input_details(self) -> list[dict]:
        return self._interp.get_input_details()

    @property
    def output_details(self) -> list[dict]:
        return self._interp.get_output_details()

    @property
    def input_shape(self) -> tuple:
        return tuple(self.input_details[0]["shape"])

    @property
    def input_dtype(self):
        return self.input_details[0]["dtype"]

    # ── Configurar entrada ────────────────────────────────────────────────────

    def set_input_tensor(self, tensor: np.ndarray):
        """
        Asigna tensor de entrada crudo.
        Shape debe ser (1, H, W, C) o (H, W, C) → se expande a batch.
        """
        if tensor.ndim == 3:
            tensor = np.expand_dims(tensor, axis=0)
        inp = self.input_details[0]
        # Convertir dtype si necesario
        if tensor.dtype != inp["dtype"]:
            if inp["dtype"] == np.uint8:
                tensor = np.clip(tensor, 0, 255).astype(np.uint8)
            elif inp["dtype"] == np.int8:
                tensor = (np.clip(tensor, 0, 255) - 128).astype(np.int8)
            else:
                tensor = tensor.astype(inp["dtype"])
        self._interp.set_tensor(inp["index"], tensor)

    def set_input_from_image(
        self,
        image_rgb: np.ndarray,
        target_size: tuple[int, int] | None = None,
    ):
        """
        Preprocesa y asigna una imagen RGB (H×W×3, uint8 0-255).

        Args:
            image_rgb   : Array numpy uint8 en formato RGB
            target_size : (width, height) opcional; si None usa input_shape del modelo
        """
        import cv2

        h, w = self.input_shape[1], self.input_shape[2]
        if target_size:
            w, h = target_size

        img = cv2.resize(image_rgb, (w, h))

        dtype = self.input_dtype
        if dtype in (np.float32,):
            img = img.astype(np.float32) / 255.0
        elif dtype == np.uint8:
            img = img.astype(np.uint8)
        elif dtype == np.int8:
            img = (img.astype(np.int16) - 128).astype(np.int8)

        self.set_input_tensor(img)

    # ── Inferencia ────────────────────────────────────────────────────────────

    def invoke(self, model_hint: str = "default") -> InferenceResult:
        """
        Ejecuta inferencia y retorna InferenceResult con outputs y métricas.

        Args:
            model_hint : 'yolo' | 'mobile' | 'efficient' | 'default'
                         Usado para calcular speedup estimado.
        """
        t0 = time.perf_counter()
        self._interp.invoke()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        outputs = [
            self._interp.get_tensor(d["index"])
            for d in self.output_details
        ]

        return InferenceResult(
            raw_output=outputs,
            latency_ms=elapsed_ms,
            backend=self.backend,
            model_hint=model_hint,
        )

    # ── Información del modelo ────────────────────────────────────────────────

    def info(self) -> str:
        """Devuelve descripción del modelo cargado."""
        inp = self.input_details[0]
        out = self.output_details[0]
        lines = [
            f"  Modelo   : {self.model_path.name}",
            f"  Backend  : {self.backend}",
            f"  Entrada  : dtype={inp['dtype'].__name__}  shape={inp['shape'].tolist()}",
            f"  Salida(s): {len(self.output_details)} tensor(s)",
            f"             shape={out['shape'].tolist()}  dtype={out['dtype'].__name__}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"CoralInterpreter(model={self.model_path.name!r}, backend={self.backend!r})"


# ─── Benchmark rápido ──────────────────────────────────────────────────────────

def benchmark(
    model_path: str | Path,
    n_runs: int = 50,
    device: str = "auto",
    verbose: bool = True,
) -> dict:
    """
    Borra n_runs de inferencia dummy y devuelve estadísticas de latencia.

    Returns:
        {"mean_ms": float, "min_ms": float, "max_ms": float, "p95_ms": float,
         "estimated_tpu_mean_ms": float, "backend": str}
    """
    interp = CoralInterpreter(model_path, device=device, verbose=verbose)
    inp = interp.input_details[0]
    dummy = np.zeros(inp["shape"], dtype=inp["dtype"])

    if verbose:
        print(interp.info())
        print(f"\n  Calentando ({min(5, n_runs)} runs)...")

    # Warmup
    for _ in range(min(5, n_runs)):
        interp.set_input_tensor(dummy.copy())
        interp.invoke()

    if verbose:
        print(f"  Benchmarking {n_runs} runs...")

    latencies = []
    for _ in range(n_runs):
        interp.set_input_tensor(dummy.copy())
        r = interp.invoke()
        latencies.append(r.latency_ms)

    arr = np.array(latencies)
    stats = {
        "mean_ms":              float(arr.mean()),
        "min_ms":               float(arr.min()),
        "max_ms":               float(arr.max()),
        "p95_ms":               float(np.percentile(arr, 95)),
        "estimated_tpu_mean_ms": float(arr.mean() / 10.0),
        "backend":              interp.backend,
    }

    if verbose:
        print(f"\n  ──── Resultados ({interp.backend}) ────")
        print(f"  Media   : {stats['mean_ms']:.1f} ms")
        print(f"  Mín     : {stats['min_ms']:.1f} ms")
        print(f"  Máx     : {stats['max_ms']:.1f} ms")
        print(f"  P95     : {stats['p95_ms']:.1f} ms")
        if interp.backend == "cpu":
            print(f"  Est. TPU: {stats['estimated_tpu_mean_ms']:.1f} ms  (~10x aceleración)")

    return stats


# ─── CLI rápida ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Benchmark de modelo TFLite vía CoralSimulator")
    p.add_argument("modelo", help="Ruta al .tflite")
    p.add_argument("--runs", type=int, default=30, help="Número de runs (default: 30)")
    p.add_argument("--device", default="auto", choices=["auto", "coral", "cpu"])
    a = p.parse_args()

    print("\n" + "="*60)
    print("  CORAL SIMULATOR — Benchmark")
    print("="*60)
    benchmark(a.modelo, n_runs=a.runs, device=a.device, verbose=True)
