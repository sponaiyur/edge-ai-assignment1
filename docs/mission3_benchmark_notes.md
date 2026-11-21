# Mission 3: Benchmark notes

> **Placeholder: these are localhost numbers.** Every measurement below was taken with the
> client and the server on the **same Mac** (`http://localhost:8000`, `uvicorn` with 1 worker,
> Python 3.13.9) on 21 Nov 2026. There is no real network in them. The Mission 3 numbers that
> count are **laptop → Codespace** over the public `*.app.github.dev` URL, plus one run
> **inside the Codespace** against `localhost`. Those still need to be captured once the
> Codespace exists. Replace or extend this file with them.

## Before testing: which stage would be slowest?

> **This was not logged before the benchmark.** The README asks for an LLM's answer to be
> logged before Mission 3, and that step was missed. The paragraph below is the answer an LLM
> gives to that question, but it was written on 21 Nov 2026, **after** the localhost runs
> below had been seen. It is not a blind prediction, so don't present it as one.

**Prediction:** the model itself will be the smallest cost. Inference is 2,736
multiply-accumulates (`/model_info`), which a CPU does in a few microseconds, and PyTorch
dispatch overhead raises that to tens of microseconds at most. Parsing a five-field JSON body,
normalising five numbers and building a two-key reply should each take microseconds too. The
slowest part by far should be **network and framework overhead**: TCP/TLS connection setup,
HTTP parsing, routing through FastAPI/Starlette and the ASGI server, and the trip over the
network. Expect that to be 90%+ of the round trip. Over the internet to a Codespace (tens of
ms) it will dwarf everything else, and turning keep-alive off should make it clearly worse
because every request pays for a new connection.

## After testing: localhost `bench.py` results

Commands (`client/bench.py` unchanged):

```
python client/bench.py --url http://localhost:8000 --n 200
python client/bench.py --url http://localhost:8000 --n 200 --no-keepalive
```

Percentiles are over requests 2..200. All times are in ms.

### Keep-alive (default)

First request: **2.6 ms** round trip.

| Stage | p50 | p95 | p99 |
|---|---:|---:|---:|
| round_trip | 0.733 | 1.110 | 2.581 |
| parse | 0.009 | 0.017 | 0.028 |
| normalise | 0.006 | 0.012 | 0.021 |
| inference | 0.044 | 0.077 | 0.140 |
| build | 0.004 | 0.006 | 0.007 |
| server_total | 0.065 | 0.115 | 0.162 |
| network_and_framework | 0.665 | 1.002 | 2.454 |

### `--no-keepalive`

First request: **2.6 ms** round trip.

| Stage | p50 | p95 | p99 |
|---|---:|---:|---:|
| round_trip | 1.572 | 2.842 | 3.471 |
| parse | 0.008 | 0.018 | 0.021 |
| normalise | 0.005 | 0.014 | 0.016 |
| inference | 0.042 | 0.090 | 0.136 |
| build | 0.003 | 0.007 | 0.007 |
| server_total | 0.060 | 0.132 | 0.202 |
| network_and_framework | 1.513 | 2.597 | 3.340 |

### Where the time goes (keep-alive, p50)

| Part | ms | Share of round trip |
|---|---:|---:|
| network_and_framework | 0.665 | 91% |
| inference | 0.044 | 6% |
| parse + normalise + build | 0.019 | 3% |
| **round_trip** | **0.733** | 100% |

Even with no real network, about 91% of a request is outside our code: the HTTP client, the
loopback socket, uvicorn and the FastAPI routing. Inference is the largest of our own stages,
but it's still only about 0.04 ms.

**Keep-alive vs no keep-alive:** the server's own time doesn't change (server_total p50 is
0.065 vs 0.060 ms). The round trip roughly doubles (0.73 → 1.57 ms p50) because every request
now opens and closes a new TCP connection. Over the internet to a Codespace, with TLS and real
latency, a new connection costs several round trips, so expect this gap to grow from about
0.8 ms to tens of ms.

## 400 single calls vs one batch of 400

The same 400 valid rows from `data/team_15_train.csv` (glitch rows removed), one
`requests.Session` (keep-alive), localhost, 3 repeats:

| | Repeat 1 | Repeat 2 | Repeat 3 |
|---|---:|---:|---:|
| 400 × `POST /predict` | 271 ms | 245 ms | 248 ms |
| 1 × `POST /predict_batch` (400 rows) | 10.9 ms | 10.7 ms | 10.5 ms |

That is about **24× faster as a batch**: about 0.6 ms per single request against about 0.03 ms
per row in a batch. Server-side timings for the batch were parse 0.31, normalise 0.17,
inference 8.6, build 0.09, server_total 9.2 ms. The batch pays the HTTP and framework overhead
**once** instead of 400 times. Almost all of the batch's time is inference, because
`/predict_batch` deliberately runs the rows one at a time (shape `[1, 5]`) so every answer
matches a single `/predict` exactly (0 mismatches in 500- and 1000-row checks).

An earlier, cruder measurement on the same machine (500 rows, single timing) gave about 322 ms
for 500 singles vs about 20 ms for one 500-row batch. Same conclusion.

## Was the prediction right?

**Yes.** Network and framework overhead was the slowest part, at about 91% of the round trip
even on localhost. Inference was about 0.04 ms (about 6%), and parse/normalise/build were each
around 0.01 ms or less. Turning off keep-alive doubled the round trip without changing server
time, and batching 400 rows was about 24× faster than 400 single calls. Both results confirm
that per-request overhead, not the model, dominates.

The reason is scale. The model is 2,849 parameters and 2,736 MACs, microseconds of arithmetic,
while every HTTP request has a fixed cost of hundreds of microseconds even over loopback.

**Caveat:** the prediction was written after these results were seen (see above), so this
"right" carries no weight as a test. The real check is the laptop → Codespace run, where the
network part should be larger still.

## Still to do (needs the Codespace)

- [ ] `python client/bench.py --url https://<name>-8000.app.github.dev --n 200` from a laptop (port 8000 set to Public)
- [ ] The same with `--no-keepalive`
- [ ] Inside the Codespace terminal: `python client/bench.py --url http://localhost:8000 --n 200`
- [ ] 400 singles vs one 400-row batch, laptop → Codespace
- [ ] Explain the differences (laptop→Codespace vs in-Codespace vs this Mac localhost)
