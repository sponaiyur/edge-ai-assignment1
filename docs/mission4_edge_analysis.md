# Mission 4: Would it fit inside the doorbell?

**Target:** ESP32-S3: about 512 KB SRAM (roughly 300 KB free for our program), 8 MB PSRAM,
16 MB flash, 240 MHz.

**Our model:** 5 → 32 → 32 → 32 → 16 → 1, ReLU (`model/config.json`, `hidden = [32, 32, 32, 16]`).
Figures from `GET /model_info`:

| Layer | in → out | Params | bytes_f32 | MACs |
|---|---|---:|---:|---:|
| linear1 | 5 → 32 | 192 | 768 | 160 |
| linear2 | 32 → 32 | 1,056 | 4,224 | 1,024 |
| linear3 | 32 → 32 | 1,056 | 4,224 | 1,024 |
| linear4 | 32 → 16 | 528 | 2,112 | 512 |
| linear5 | 16 → 1 | 17 | 68 | 16 |
| **Total** | | **2,849** | **11,396** | **2,736** |

`total_bytes_int8` = **2,849**, `fits_300kb_sram_f32` = **true**, `fits_300kb_sram_int8` = **true**.

---

## 1. Weights and biases

| Precision | Bytes | Share of 300 KB (307,200 B) |
|---|---:|---:|
| float32 | 11,396 | 3.7% |
| int8 | 2,849 | 0.9% |

The int8 figure counts 1 byte per parameter, which is the README's counting rule. Real int8
quantisation usually keeps the 113 biases as int32, which adds 113 × 3 = 339 bytes (3,188 bytes
in total). It still fits easily.

## 2. Largest activation memory needed at once

A layer has to hold its input vector and its output vector at the same time. For each layer
of the real architecture:

| Layer | Input values | Output values | Held at once | float32 bytes | int8 bytes |
|---|---:|---:|---:|---:|---:|
| linear1 | 5 | 32 | 37 | 148 | 37 |
| **linear2** | 32 | 32 | **64** | **256** | **64** |
| **linear3** | 32 | 32 | **64** | **256** | **64** |
| linear4 | 32 | 16 | 48 | 192 | 48 |
| linear5 | 16 | 1 | 17 | 68 | 17 |

**Largest: 64 values, in linear2 and linear3 (32 → 32). That is 256 bytes at float32 and
64 bytes at int8.** ReLU works in place, so it needs no extra buffer. Two 32-value buffers that
take turns being input and output are enough for the whole network.

Total RAM for inference is therefore about 11,396 + 256 = **11,652 bytes at float32**, or
2,849 + 64 = **2,913 bytes at int8**. That is less than 4% of the 300 KB budget either way.

## 3. Inference time on the ESP32-S3

### Source

Espressif's ESP-NN library (optimised int8 kernels for ESP32 chips) publishes per-kernel cycle
counts in `docs/performance.md`, measured on an **ESP32-S3 at 240 MHz** (QPI PSRAM at 80 MHz,
64 KB data cache). The ESP32-S3 rows used here are quoted exactly from that page
(read 21 Nov 2026 from the `master` branch; the latest tag then was v1.5.0):

| Kernel (ESP32-S3) | Shape | ANSI C cycles | Optimised cycles | Memory |
|---|---|---:|---:|---|
| Fully connected | len 271, ch 3 | 8,439 | 1,543 | Internal |
| Convolution 1x1 | input (10,10), filter 64x1x1x64 | 4,712,500 | 331,008 | External |
| Convolution 1x1 | input (8,8), filter 16x1x1x16 | 312,754 | 39,022 | External |

### Method

1. Divide each row's cycle count by its MAC count to get cycles per MAC.
2. Multiply by our model's **2,736 MACs** (`total_macs` from `/model_info`).
3. Divide by 240 MHz (240 cycles per µs).

| Basis | MACs in row | Cycles / MAC | Cycles for 2,736 MACs | Time at 240 MHz |
|---|---:|---:|---:|---:|
| **Fully connected, optimised (main estimate)** | 271 × 3 = 813 | 1.90 | 5,193 | **≈ 22 µs** |
| Fully connected, ANSI C (slow-case fallback) | 813 | 10.38 | 28,400 | ≈ 118 µs |
| Cross-check: 1x1 conv 64→64 on 10x10, optimised | 10·10·64·64 = 409,600 | 0.81 | 2,211 | ≈ 9 µs |
| Cross-check: 1x1 conv 16→16 on 8x8, optimised | 8·8·16·16 = 16,384 | 2.38 | 6,516 | ≈ 27 µs |

A 1x1 convolution is the same arithmetic as a fully connected layer applied at every pixel,
so the conv rows are a fair cross-check. The large conv amortises its per-call overhead over
many MACs and gives the fastest rate. The small conv is closer to our layer sizes and lands
near the FC estimate.

**Estimate: about 22 µs per prediction (int8, ESP-NN optimised kernels, 240 MHz). The
cross-checks give 9–27 µs. Even the slow case, with unoptimised ANSI C kernels, is about
0.12 ms.**

Caveats:
- The FC row is a single layer. Our model makes 5 layer calls plus 4 ReLUs, and its layers are
  smaller than 271 inputs, so per-call overhead weighs more. Treat 22 µs as "tens of µs", not
  as a precise figure.
- ESP-NN's optimised kernels are **int8**. A float32 model would use the chip's FPU through
  generic code, closer to the slow-case figure. It would still be well under a millisecond.
- **Correction to the earlier estimate.** An earlier draft said "~7 µs for the 1216-MAC
  (5 → 32 → 32 → 1) model, from the ESP-NN readme v1.2.3". I couldn't confirm it. ESP-NN has
  **no v1.2.3 release or tag** (v1.2.0 is followed by v1.3.0). The README table gives speedup
  ratios, not times. Applying the cycle counts above to 1,216 MACs gives 4–12 µs (FC row:
  9.6 µs), not 7 µs. Scaling the old 7 µs by 2,736 / 1,216 = 2.25 would give about 16 µs,
  which falls inside the range above. The estimate in this document is the one derived from
  the cycle counts.

### Sanity check: person detection

The ESP-NN README reports the TFLite Micro **person-detection** model (Visual Wake Words,
INT8) on an **ESP32-S3 at 240 MHz** taking **54 ms** per `invoke()` with ESP-NN (2,300 ms
without), and **47 ms** with internal RAM. That is a complete image CNN. Our whole model is
about 2,000× faster than that by the main estimate (47 ms / 22 µs), and still about 400×
faster by the slow-case figure (47 ms / 118 µs), so a run time of tens of microseconds is
plausible. The README gives no MAC count for that model, so it's only used as
an upper bound here, not to scale our estimate.

---

## 4. Server or device? The doorbell should run the model itself.

**Evidence:** the model is **11,396 bytes float32 / 2,849 bytes int8** (3.7% / 0.9% of the free
SRAM). It needs **256 bytes** of activations at most and runs in **about 22 µs** per prediction
on the ESP32-S3 (at most about 0.12 ms).

- **Latency.** On the device, one prediction costs about 22 µs. Calling the server costs a full
  HTTP round trip. Even on localhost with no real network, that measured 0.73 ms (p50) with
  keep-alive and 1.57 ms without (`docs/mission3_benchmark_notes.md`). Over home Wi-Fi and the
  internet to a Codespace it will be much more, plus the time to wake the radio and connect.
  The server's own model time (≈0.04 ms) is already tiny. Almost all of the cost is network
  and framework, which on-device inference avoids completely.
- **Privacy.** The inputs include `rings_per_day`, which reveals when people are home and how
  many visitors come. Running locally means those readings never leave the house.
- **Cost.** A server has to be hosted and kept running for every doorbell sold, and the cost
  grows with the number of devices and requests. On the device, the model costs nothing extra:
  2.9 KB of storage on a chip already in the product, and no cloud bill.
- **Power.** The doorbell runs on a battery, and the battery is what we're predicting.
  Switching on Wi-Fi to send a request and wait for a reply uses far more energy than a few
  thousand int8 MACs taking about 22 µs of CPU time. Running locally saves the very resource
  the model is meant to protect.
- **When the Wi-Fi is down.** A server-based doorbell can't produce an estimate without Wi-Fi.
  An on-device model keeps working. That matters most when the battery is low, the Wi-Fi is
  weak (low `wifi_rssi_dbm`) and the user most needs a warning.

**Conclusion:** run the model on the doorbell. It fits many times over, it runs in
microseconds, and it keeps working offline, privately and at no running cost. The server is
still useful for training, for collecting opt-in data to improve the model, and for pushing
updated weights to doorbells.

---

## Sources

- Espressif ESP-NN, kernel cycle counts (ESP32-S3 at 240 MHz):
  <https://github.com/espressif/esp-nn/blob/master/docs/performance.md>
- Espressif ESP-NN README (speedup table, person-detection model benchmark):
  <https://github.com/espressif/esp-nn/blob/master/README.md>
- ESP-NN tags (shows there is no v1.2.3): <https://github.com/espressif/esp-nn/tags>
- Our model's sizes and MACs: live `GET /model_info` from this repo's `app/main.py`.
- ESP32-S3 memory budget (300 KB free SRAM, 8 MB PSRAM): the assignment README, Mission 4.

All web sources were read on 21 Nov 2026.
