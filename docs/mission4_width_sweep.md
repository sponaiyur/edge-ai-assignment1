# Mission 4: Width sweep for 5 → w → w → 1

How big can a two-hidden-layer BatteryNet get before its weights stop fitting on the ESP32-S3?

**Memory budgets** (same constants as `/model_info` in `app/main.py`):
- SRAM free for the program: 300 KB = 300 × 1024 = **307,200 bytes**
- PSRAM: 8 MB = 8 × 1024 × 1024 = **8,388,608 bytes**

The "fits" columns count **weights and biases only**. They leave out activations, the
runtime and the rest of the firmware, which is how the README frames the question.

## Formulas

For 5 → w → w → 1:

| Layer | Params (weights + biases) | MACs |
|---|---|---|
| Linear(5, w) | 5w + w | 5w |
| Linear(w, w) | w² + w | w² |
| Linear(w, 1) | w + 1 | w |
| **Total** | **w² + 8w + 1** | **w² + 6w** |

float32 = 4 bytes per param, int8 = 1 byte per param.

**Checked against the real network.** For w = 3, 32, 100, 273, 274, 550 and 551, I built
`BatteryNet((w, w))` from `app/model.py`. I counted `sum(p.numel())` and the
`in × out` MACs of every `nn.Linear`, and both matched the formulas exactly. For example,
w = 32 gives 1,281 params and 1,216 MACs, the same as the README's example `/model_info`.

## Table

| w | Params | float32 bytes | int8 bytes | Fits 300 KB SRAM (f32) | Fits 300 KB SRAM (int8) | Fits 8 MB PSRAM (f32) | Fits 8 MB PSRAM (int8) | MACs |
|---:|---:|---:|---:|:---:|:---:|:---:|:---:|---:|
| 32 | 1,281 | 5,124 | 1,281 | yes | yes | yes | yes | 1,216 |
| 128 | 17,409 | 69,636 | 17,409 | yes | yes | yes | yes | 17,152 |
| 256 | 67,585 | 270,340 | 67,585 | yes | yes | yes | yes | 67,072 |
| 512 | 266,241 | 1,064,964 | 266,241 | **no** | yes | yes | yes | 265,216 |
| 2048 | 4,210,689 | 16,842,756 | 4,210,689 | **no** | **no** | **no** | yes | 4,206,592 |

## Exact crossover widths (300 KB SRAM)

| Precision | Last width that fits | Bytes | First width that doesn't | Bytes |
|---|---:|---:|---:|---:|
| float32 | **w = 273** | 306,856 | **w = 274** | 309,076 |
| int8 | **w = 550** | 306,901 | **w = 551** | 308,010 |

- **float32:** 4(w² + 8w + 1) ≤ 307,200 gives w ≤ 273.1, so the weights stop fitting at **w = 274**.
- **int8:** w² + 8w + 1 ≤ 307,200 gives w ≤ 550.3, so the weights stop fitting at **w = 551**.

Going from float32 to int8 makes the weights 4× smaller, which lets the width roughly double
(273 → 550). Params grow with w², so 4× the memory buys about √4 = 2× the width.

## Where our real model sits

Our deployed model is 5 → 32 → 32 → 32 → 16 → 1 (from `/model_info`): **2,849 params,
11,396 bytes float32, 2,849 bytes int8, 2,736 MACs**. That is about 3.7% of the 300 KB SRAM
budget at float32 and about 0.9% at int8.
