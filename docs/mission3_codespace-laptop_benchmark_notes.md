# Mission 3: Serve it and time every stage

## Initial latency expectation

Our qualitative expectation was that network and framework overhead
would dominate laptop-to-Codespace round-trip latency, while model
inference would account for a small fraction. This was based on the
selected network's small size (2,849 parameters and 2,736 MACs) and our
earlier training experience: 1,500 epochs took approximately 4–5
seconds. Training time is not a direct measure of inference latency, but
it suggested modest computational requirements. We did not record an
exact numerical prediction beforehand.

## Laptop-to-Codespace benchmark

`client/bench.py` was run unchanged from a Windows laptop against the
public Codespaces URL. Each run used `--n 200`; percentiles are over
requests 2–200. All times are milliseconds.

### Keep-alive enabled

First request: **828.3 ms**.

| Stage                      |    p50 |    p95 |     p99 |
|----------------------------|-------:|-------:|--------:|
| Round trip                 | 38.636 | 46.516 | 119.821 |
| Parse                      |  0.033 |  0.050 |   0.057 |
| Normalise                  |  0.031 |  0.053 |   0.069 |
| Inference                  |  0.198 |  0.322 |   0.365 |
| Build                      |  0.014 |  0.020 |   0.042 |
| Server total               |  0.285 |  0.454 |   0.514 |
| Network/framework residual | 38.337 | 46.289 | 119.395 |

The residual is calculated as round-trip time minus measured server
time. It approximates time outside the measured server stages; it is not
a pure network measurement. At p50, it accounts for approximately
**99.2%** of round-trip time, while inference accounts for approximately
**0.5%**.

### Keep-alive disabled

First request: **873.2 ms**.

| Stage                      |     p50 |     p95 |      p99 |
|----------------------------|--------:|--------:|---------:|
| Round trip                 | 756.963 | 886.305 | 1031.620 |
| Parse                      |   0.043 |   0.052 |    0.072 |
| Normalise                  |   0.044 |   0.071 |    0.083 |
| Inference                  |   0.255 |   0.342 |    0.398 |
| Build                      |   0.016 |   0.021 |    0.041 |
| Server total               |   0.386 |   0.498 |    0.547 |
| Network/framework residual | 756.601 | 885.940 | 1031.195 |

Disabling keep-alive increased p50 round-trip latency from **38.636 ms
to 756.963 ms**, approximately **19.6×**. Server processing remained
small; repeatedly establishing connections added substantial overhead.

## Benchmark inside the Codespace

The same benchmark was run against `http://localhost:8000` from inside
the Codespace.

First request: **2.1 ms**.

| Stage                      |   p50 |   p95 |   p99 |
|----------------------------|------:|------:|------:|
| Round trip                 | 2.127 | 3.654 | 5.050 |
| Parse                      | 0.024 | 0.185 | 0.610 |
| Normalise                  | 0.022 | 0.026 | 0.123 |
| Inference                  | 0.177 | 0.229 | 1.107 |
| Build                      | 0.014 | 0.015 | 0.041 |
| Server total               | 0.268 | 0.536 | 1.179 |
| Network/framework residual | 1.838 | 3.206 | 3.780 |

### Remote versus local

| Metric (p50)               | Laptop → Codespace | Codespace localhost |
|----------------------------|-------------------:|--------------------:|
| Round trip                 |          38.636 ms |            2.127 ms |
| Server total               |           0.285 ms |            0.268 ms |
| Network/framework residual |          38.337 ms |            1.838 ms |

Server-side processing was similar, but the laptop request traversed the
network to the Codespace. The localhost request stayed within the
Codespace environment.

## 400 individual requests versus one batch

A separate script loaded 400 valid readings from
`data/400_rows_batch.csv` on the laptop and sent the same readings to
the public Codespaces endpoint as (1) 400 individual `POST /predict`
requests using a persistent `requests.Session`, and (2) one
`POST /predict_batch` request.

| Operation                    | Total elapsed time |
|------------------------------|-------------------:|
| 400 individual requests      |      19,681.049 ms |
| One batch request (400 rows) |          93.892 ms |

Batching was approximately **209.61× faster**. There were **zero
prediction mismatches**, with a maximum absolute prediction difference
of **0.000000**.

Batch server timings:

| Stage        | Time (ms) |
|--------------|----------:|
| Parse        |     9.407 |
| Normalise    |     0.817 |
| Inference    |    22.743 |
| Build        |     0.217 |
| Server total |    33.202 |

Batching pays the HTTP/framework request overhead once rather than 400
times. The batch endpoint uses the same single-reading inference path,
preserving equivalence with `/predict`.

## Interpretation and conclusion

1.  Remote request latency is dominated by overhead outside the measured
    model stages: keep-alive p50 round trip was 38.636 ms, compared with
    0.198 ms for inference.
2.  Keep-alive is important for remote calls: disabling it increased p50
    round-trip latency by approximately 19.6×.
3.  The laptop-to-Codespace p50 round trip was 38.636 ms, compared with
    2.127 ms from inside the Codespace using localhost.
4.  One batch of 400 readings was approximately 209.61× faster than 400
    individual requests, with identical predictions.

The results support our qualitative expectation that communication
overhead would dominate inference for this small model. We did not
record an exact numerical prediction beforehand, so this should not be
presented as a blind numerical prediction.

## Reproducibility

``` bash
python client/bench.py --url <CODESPACE_URL> --n 200
python client/bench.py --url <CODESPACE_URL> --n 200 --no-keepalive
python client/bench.py --url http://localhost:8000 --n 200
python client/bench_batch.py --url <CODESPACE_URL> data/400_rows_batch.csv
```

`client/bench.py` was not modified. The 400-row CSV was created by sampling 400 valid rows from the cleaned training dataset and was used only as input for benchmarking the single-request and batch prediction endpoints. These rows were already part of the model's training/validation dataset; no new data was introduced or used for additional training during the benchmark.
