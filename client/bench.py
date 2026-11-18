"""
client/bench.py -- time your API from the outside. Run it from your LAPTOP, not the Codespace.

    python client/bench.py --url https://<your-codespace>-8000.app.github.dev --n 200
    python client/bench.py --url http://localhost:8000 --n 200          # inside the Codespace
    python client/bench.py --url ... --n 200 --no-keepalive             # a new connection every time

It sends n readings to /predict, one at a time, and prints:
  - the very first request on its own (the "cold" one)
  - median, 95th and 99th percentile of the round trip
  - the same for every stage your server reports in "timings_ms"
  - network + framework time = round trip - server_total
It also saves every number to bench_<timestamp>.csv.
Needs: pip install requests numpy
"""
import argparse
import csv
import time

import numpy as np
import requests


def reading(r):
    return {"battery_v": round(r.uniform(3.4, 4.2), 3), "temp_c": round(r.uniform(10, 45), 1),
            "rings_per_day": int(r.integers(0, 30)), "wifi_rssi_dbm": int(r.integers(-88, -45)),
            "age_months": int(r.integers(0, 37))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--no-keepalive", action="store_true")
    a = ap.parse_args()
    url = a.url.rstrip("/")
    r = np.random.default_rng(1)
    print("health:", requests.get(url + "/health", timeout=30).json())
    s = requests.Session()
    rows = []
    for i in range(a.n):
        t0 = time.perf_counter()
        if a.no_keepalive:
            resp = requests.post(url + "/predict", json=reading(r), timeout=30, headers={"Connection": "close"})
        else:
            resp = s.post(url + "/predict", json=reading(r), timeout=30)
        rt = (time.perf_counter() - t0) * 1000
        if resp.status_code != 200:
            print("request", i, "->", resp.status_code, resp.text[:200]); continue
        tm = resp.json().get("timings_ms", {})
        row = {"i": i, "round_trip": rt, **tm}
        if "server_total" in tm:
            row["network_and_framework"] = rt - tm["server_total"]
        rows.append(row)
    if not rows:
        return
    keys = [k for k in rows[0] if k != "i"]
    print(f"\nfirst request: {rows[0]['round_trip']:.1f} ms round trip")
    print(f"{'stage':24s} {'p50 ms':>9} {'p95 ms':>9} {'p99 ms':>9}   (requests 2..{len(rows)})")
    for k in keys:
        v = np.array([x[k] for x in rows[1:] if k in x])
        if v.size:
            print(f"{k:24s} {np.percentile(v, 50):9.3f} {np.percentile(v, 95):9.3f} {np.percentile(v, 99):9.3f}")
    name = f"bench_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["i"] + keys); w.writeheader(); w.writerows(rows)
    print("saved", name)


if __name__ == "__main__":
    main()
