# Looper-Minimax-H3

Extensión para **ComfyUI** diseñada para crear **loops de video infinitos y sin costuras (seamless)** usando el modelo de video y audio **MiniMax H3 (Hailuo-01)**.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Extension-black.svg)](https://github.com/comfyanonymous/ComfyUI)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

---

## ¿Qué hace este proyecto?

Crear un loop perfecto (donde el final de un video se conecte de forma invisible con su principio) normalmente requiere cortar el video en programas externos, hacer subgrafos gigantes en ComfyUI y lidiar con saltos bruscos de color o audio.

**Looper-Minimax-H3** automatiza todo el proceso:
1. Toma tu video (o tu `LATENT` generado en ComfyUI), lo divide en dos mitades (**Parte A** y **Parte B**) e invierte su orden para que el final y el principio queden frente a frente en el centro.
2. Usa la IA de **MiniMax H3** para generar un puente de transición fluido entre el final del video y el principio, siguiendo tu **prompt**.
3. Une todo automáticamente (`[Parte B] + [Puente Generado] + [Parte A]`) con audio sincronizado, de modo que cuando el reproductor llega al final, vuelve al fotograma 1 sin ningún corte visible.

---

## ¿Cómo se usa? (2 Formas de Trabajar)

Puedes usar estos nodos de **dos maneras distintas** según lo que tengas en tu workflow:

### Opción 1: Hacer Loop desde un Archivo de Video (MP4 / Video Cargado)
Usa esta opción cuando ya tienes un video guardado en tu computadora y quieres convertirlo en un loop infinito. **Solo necesitas 2 nodos**:

```
[ ModelLoader] ──► model (opcional) ┐
[ CLIPLoader ] ──► clip  ───────────┤
[ VAELoader  ] ──► vae   ───────────┼──► [ 1. Pre Looper (MiniMax H3) ]
[ Tu Prompt  ] ──► prompt ──────────┘      │
                                           ├──► model    ──────► [ BasicGuider ] / [ SamplerCustomAdvanced ]
                                           ├──► positive ──────► [ BasicGuider ] ──► [ SamplerCustomAdvanced ]
                                           ├──► latent   ──────────────────────────► [ SamplerCustomAdvanced ]
                                           ├──► chunk_a  ────────┐
                                           └──► chunk_b  ──┐     │
                                                           ▼     ▼
[ VAEDecode  ] ──► trans_image ────────────────────► [ 2. Post Looper Video (MiniMax H3) ] ──► [ VHS_VideoCombine ]
```

1. **`Pre Looper (MiniMax H3)`**: Cargas tu video directamente dentro del nodo (tiene un reproductor interactivo integrado para recortar el inicio `head_a`, el final `tail_b` o mover el punto de corte `seam_frame`). Prepara las guías para el Sampler.
2. **`Post Looper Video (MiniMax H3)`**: Recibe las dos partes del video (`chunk_a` y `chunk_b`) más la transición decodificada (`trans_image`) y entrega el loop final listo para guardar.

---

### Opción 2: Hacer Loop directamente desde un `LATENT` (Sin perder calidad)
Usa esta opción cuando acabas de generar un video dentro de ComfyUI y todavía está en formato **`LATENT`**. Trabaja 100% en la memoria VRAM sin decodificar y volver a codificar el video completo, manteniendo la máxima calidad y velocidad. **Solo necesitas 2 o 3 nodos**:

```
[ In Latent  ] ──► [ Pre Looper Latent Preview (MiniMax H3) ] (Opcional: para ver y ajustar los cortes visualmente)
                         │
[ ModelLoader] ──► model (opcional) ┐
[ CLIPLoader ] ──► clip  ───────────┤
[ VAELoader  ] ──► vae   ───────────┼──► [ 1. Pre Looper Latent (MiniMax H3) ]
[ Tu Prompt  ] ──► prompt ──────────┘      │
                                           ├──► model          ──────► [ BasicGuider ] / [ SamplerCustomAdvanced ]
                                           ├──► positive       ──────► [ BasicGuider ] ──► [ SamplerCustomAdvanced ]
                                           ├──► latent         ──────────────────────────► [ SamplerCustomAdvanced ]
                                           ├──► chunk_latent_a ────────┐
                                           └──► chunk_latent_b ──┐     │
                                                                 ▼     ▼
[ Sampler    ] ──► trans_latent ─────────────────────────► [ 2. Post Looper Latent (MiniMax H3) ] ──► [ VAEDecode ] ──► [ VHS_VideoCombine ]
```

1. **`Pre Looper Latent Preview (MiniMax H3)` *(Opcional)***: Si quieres ver el video antes de cortarlo y mover las barras de recorte con el ratón, conéctalo antes del Pre Looper Latent.
2. **`Pre Looper Latent (MiniMax H3)`**: Corta los tokens latentes directamente en VRAM, extrae únicamente el fotograma de referencia necesario (un 75% más rápido) e inyecta las guías de continuación nativas de MiniMax H3.
3. **`Post Looper Latent (MiniMax H3)`**: Une los latentes en orden circular, corrige cualquier diferencia de brillo/color en las uniones y entrega un único `LATENT` para que hagas un solo `VAEDecode` al final.

---

## Los 5 Nodos Incluidos

Al buscar en la categoría **`LooperMinimaxH3`** de ComfyUI encontrarás únicamente los 5 nodos esenciales:

| Nodo | ¿Para qué flujo es? | Función principal |
| :--- | :--- | :--- |
| **1. Pre Looper (MiniMax H3)** | Video (MP4 / Imagen) | Carga el video con reproductor interactivo, reescala por bloques sin saturar la memoria, prepara el prompt, las guías y (opcionalmente) parchea el `model`. |
| **2. Post Looper Video (MiniMax H3)** | Video (MP4 / Imagen) | Recorta el solapamiento de la transición, corrige saltos de color en ambas costuras, une el audio con micro-crossfade y arma el loop final. |
| **3. Pre Looper Latent (MiniMax H3)** | Latent (VRAM) | Corta el `LATENT` sin pérdida de calidad, inyecta keyframes latentes en `positive` y parchea opcionalmente el `model`. |
| **4. Post Looper Latent (MiniMax H3)** | Latent (VRAM) | Ensambla el loop completo en espacio latente (`chunk_latent_b + trans_latent + chunk_latent_a`) antes del `VAEDecode` final. |
| **5. Pre Looper Latent Preview (MiniMax H3)** | Latent (VRAM) | Genera una vista previa ligera e interactiva para elegir visualmente dónde cortar (`head_a`, `tail_b`, `seam_frame`) cuando trabajas con latentes. |

---

## Parámetros Principales Explicados de Forma Sencilla

En **`Pre Looper`** y **`Pre Looper Latent`** encontrarás estos ajustes clave:

* **`trans_duration`** *(Por defecto: `56`)*: Cuántos fotogramas durará el puente generado por la IA. Se ajusta automáticamente a la regla matemática de MiniMax H3 ($17k + 5$ fotogramas, como `39`, `56`, `73`, `90`). Si tu prompt pide una acción compleja, súbelo a `73` o `90`.
* **`head_a` / `tail_b` / `seam_frame`**: Controles para recortar fotogramas del inicio (`head_a`), del final (`tail_b`) o elegir el fotograma exacto donde se parte el video (`seam_frame`, `0` = mitad exacta).
* **`prompt_assist`** *(Por defecto: `True`)*: Formatea automáticamente tu prompt agregando `<Picture 1>`, indicaciones de movimiento continuo y etiquetas de sonido de MiniMax H3 para que el modelo obedezca mejor tu instrucción sin chocar entre el inicio y el final.
* **`guide_tokens` / `guide_frames`** *(Por defecto: `5 tokens` / `22 frames`)*: Cantidad de contexto del video original que se inyecta a cada lado del puente. Un valor de `5` tokens deja más espacio libre en el centro para que la IA haga caso a tu prompt.
* **`model` *(Opcional)* + `continuity_strength` + `pad_clips`**:
  * Si conectas tu modelo en la entrada `model` y lo sacas por la salida `model` hacia el Sampler, puedes activar **`continuity_strength`** (ej. `0.25`) para forzar que los bordes del puente no cambien de luz ni color durante el muestreo (se libera suavemente al final gracias a `sigma_schedule`), o activar **`pad_clips`** (ej. `1`) para envolver el inicio con el final de forma periódica.

En **`Post Looper Video`** y **`Post Looper Latent`**:

* **`interpolate_multiplier`** *(En Post Looper Video: `1`, `2`, `4`)*: Permite duplicar o cuadruplicar la fluidez de fotogramas del loop completo (incluyendo el cierre entre el último y el primer fotograma).
* **`circular_pad_clips`** *(En Post Looper Latent)*: Envuelve tokens latentes del final al principio antes de pasar por el `VAEDecode` para que el decodificador mantenga continuidad circular perfecta.

---

## Instalación

Abre una terminal dentro de la carpeta `ComfyUI/custom_nodes` y ejecuta:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/zze1l/Looper-Minimax-H3.git
```

Instala las dependencias en el entorno Python de tu ComfyUI:

```bash
# Si usas ComfyUI Portable (Windows):
ruta/a/python_embeded/python.exe -m pip install -r Looper-Minimax-H3/requirements.txt

# O si usas un entorno virtual activado:
pip install -r Looper-Minimax-H3/requirements.txt
```

Reinicia ComfyUI. Encontrarás todos los nodos dentro del menú **`LooperMinimaxH3`**.

---

## Créditos y Agradecimientos

* **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** por comfyanonymous.
* **[MiniMax AI](https://www.minimax.io/)**: Creadores del modelo base MiniMax H3 (Hailuo-01).
* **[ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)** por Kosinkadink.

---

## Licencia

Distribuido bajo la licencia **Apache License 2.0**. Consulta el archivo [LICENSE](LICENSE) para más detalles.
