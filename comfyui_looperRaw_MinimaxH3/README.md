# Looper-Minimax-H3 (Nodos para ComfyUI)

Nodos personalizados para **ComfyUI** diseñados para crear **loops de video infinitos y sin cortes visibles** utilizando el modelo **MiniMax H3 (Hailuo-01)**.

Permite crear loops perfectos tanto a partir de un **archivo de Video (MP4)** como directamente desde un **`LATENT`** generado en ComfyUI, sin subgrafos complicados.

---

## ¿Con qué quieres hacer el Loop?

### 1. Con un Video (Flujo de Video / Píxeles)
Ideal cuando cargas un archivo MP4 desde tu disco. **Solo se usan 2 nodos**:

```
[ ModelLoader] ──► model (opcional) ┐
[ CLIPLoader ] ──► clip  ───────────┤
[ VAELoader  ] ──► vae   ───────────┼──► [ Pre Looper (MiniMax H3) ]
[ Tu Prompt  ] ──► prompt ──────────┘      │
                                           ├──► model    ──────► [ BasicGuider ] / [ SamplerCustomAdvanced ]
                                           ├──► positive ──────► [ BasicGuider ] ──► [ SamplerCustomAdvanced ]
                                           ├──► latent   ──────────────────────────► [ SamplerCustomAdvanced ]
                                           ├──► chunk_a  ────────┐
                                           └──► chunk_b  ──┐     │
                                                           ▼     ▼
[ VAEDecode  ] ──► trans_image ────────────────────► [ Post Looper Video (MiniMax H3) ] ──► [ VHS_VideoCombine ]
```

### 2. Con un Latent (Flujo Nativo en VRAM sin perder calidad)
Ideal cuando el video viene directamente de un Sampler en ComfyUI. No decodifica el video intermedio y ahorra hasta un 75% de tiempo en extracción de referencias:

```
[ In Latent  ] ──► [ Pre Looper Latent Preview (MiniMax H3) ] (Opcional: vista previa interactiva)
                         │
[ ModelLoader] ──► model (opcional) ┐
[ CLIPLoader ] ──► clip  ───────────┤
[ VAELoader  ] ──► vae   ───────────┼──► [ Pre Looper Latent (MiniMax H3) ]
[ Tu Prompt  ] ──► prompt ──────────┘      │
                                           ├──► model          ──────► [ BasicGuider ] / [ SamplerCustomAdvanced ]
                                           ├──► positive       ──────► [ BasicGuider ] ──► [ SamplerCustomAdvanced ]
                                           ├──► latent         ──────────────────────────► [ SamplerCustomAdvanced ]
                                           ├──► chunk_latent_a ────────┐
                                           └──► chunk_latent_b ──┐     │
                                                                 ▼     ▼
[ Sampler    ] ──► trans_latent ─────────────────────────► [ Post Looper Latent (MiniMax H3) ] ──► [ VAEDecode ] ──► [ VHS_VideoCombine ]
```

---

## Los 5 Nodos de la Suite

1. **`Pre Looper (MiniMax H3)`** (`PreLooperMinimax`): Carga el video con reproductor interactivo, recorta extremos, escala con Lanczos en bloques de 32 frames, aplica `prompt_assist`, inyecta las guías y parchea opcionalmente el `model`.
2. **`Post Looper Video (MiniMax H3)`** (`PostLooperVideoMinimax`): Elimina el solapamiento de frames guía, corrige el salto de iluminación en ambas costuras, interpola fotogramas de forma circular (`1x`, `2x`, `4x`) y une el audio con micro-crossfade.
3. **`Pre Looper Latent (MiniMax H3)`** (`PreLooperLatentMinimax`): Corta el `LATENT` respetando la regla $17k+5$, decodifica solo 1 clip causal para `<Picture 1>`, inyecta keyframes latentes en `positive` y parchea opcionalmente el `model`.
4. **`Post Looper Latent (MiniMax H3)`** (`PostLooperLatentMinimax`): Recorta automáticamente los tokens guía del puente, corrige derivas en ambas uniones y ensambla el loop completo en VRAM antes del `VAEDecode`.
5. **`Pre Looper Latent Preview (MiniMax H3)`** (`PreLooperLatentPreviewMinimax`): Reproductor interactivo ligero para visualizar y ajustar los puntos de corte (`head_a`, `tail_b`, `seam_frame`) trabajando con latentes.

---

## Licencia

Distribuido bajo la licencia **Apache License 2.0**.
