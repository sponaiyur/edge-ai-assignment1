# Module 1: How Long Will the Battery Last?

**AI255TBD Edge AI · Module 1 assignment: Neural Networks from Scratch to Deployment**
Team of four · runs entirely in GitHub Codespaces (no board this time)

The doorbell company wants a number instead of "Low": **how many days of battery are left?**
You train a small neural network on your team's log and serve it as a live API from a GitHub
Codespace. The API also reports where its time goes, and whether the model would fit inside
the doorbell itself.

**You hand in one thing: a working endpoint.** Your instructor checks it with a script that
sends it readings. There is no report and no viva, so the endpoint has to speak for you.

This README is everything you need: what to do, in what order, and what is checked.
**Find your team number in `TEAMS.csv`.** Your data file is `data/team_NN_train.csv` with that
number. Use only your own team's file: every team's data, and the right answer, is different.

---

## 1. What is in this repository

| Path | What it is | Your job |
|---|---|---|
| `TEAMS.csv` | Team numbers and names | Find your number |
| `data/` | Every team's `team_NN_train.csv` | Use only yours |
| `train.py` | Loads the CSV, splits off validation rows, trains the network, saves it to `model/` | TODOs: glitch rule, linear baseline, normalisation |
| `app/model.py` | The network (5 → 32 → 32 → 1, ReLU), shared by training and the server | Change the layer sizes if you choose to |
| `app/main.py` | The API: `GET /health`, `GET /model_info`, `POST /predict`, `POST /predict_batch` | TODOs: input ranges, stage timing, `/model_info` |
| `client/bench.py` | Times your API from outside: first request, p50/p95/p99 per stage | Run it, don't change it |
| `.devcontainer/` | Makes Codespaces install everything on start | Leave it alone |
| `model/` | Created by `train.py`: `model.pt` and `config.json` | **Commit it** |

---

## 2. Set up (about 20 minutes)

1. One member creates a new repository on GitHub for your team and uploads **everything** in
   this folder, keeping the folders as they are. Include the hidden `.devcontainer` folder
   (on a Mac press Cmd+Shift+. in Finder to see it). Add the other three as collaborators.
2. On your repository: **Code → Codespaces → Create codespace on main**. The first start
   installs PyTorch and takes a few minutes. If nothing installs by itself, run
   `pip install -r requirements.txt` in the terminal.
3. Check that `data/team_NN_train.csv` (your number) is there.
4. In the Codespace terminal, from the top folder:
   ```
   python train.py --team NN
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
   The starter runs before you fill in any TODO. Its error will just be large.
5. **Ports** tab → right-click **8000** → **Port Visibility → Public**. Copy the address
   (`https://<name>-8000.app.github.dev`).
6. On your laptop, open `<address>/health`, then `<address>/docs` to try `/predict` by hand.
7. On your laptop: `pip install requests numpy`, then
   `python client/bench.py --url <address> --n 20`.

Stop the Codespace when you are not using it (your free hours are limited). Commit and push
after every mission.

**Every time the Codespace restarts, port 8000 goes back to Private.** After a restart, run
`uvicorn` again and set the port to **Public** again, or nobody else can reach your endpoint.

**Before Mission 3:** ask an LLM which stage of a web request will be the slowest. Log its answer.

---

## 3. What you need to do

Every mission follows the same loop: **write a prediction with a number → measure → explain the
gap.** Rotate the roles (Builder, Measurer, Skeptic, Scribe) so everyone builds every part.

### Mission 1: Meet the data and set the bar
- [ ] Predict the error of always guessing the average.
- [ ] Plot a histogram of every column, and `days_left` against each input. Which relationships are not straight lines?
- [ ] **Code (`train.py`):** write the glitch rule for each kind of impossible reading you find, and count the rows removed.
- [ ] Report the validation MAE (in days) of **baseline 1**, always predicting the mean.
- [ ] **Code (`train.py`):** report the validation MAE of **baseline 2**, linear regression with `np.linalg.lstsq`. Plot predicted against true. Where is it most wrong?
- [ ] Run `train.py` as it is and record the network's MAE.

### Mission 2: A small network, opened up
- [ ] Predict, by hand, the parameters, float32 bytes and MACs of 5 → 32 → 32 → 1.
- [ ] **Code (`train.py`):** normalise the inputs using the training rows only. Compare training with and without normalisation.
- [ ] Train with hidden widths 4, 8, 16, 32 and 64, with one and two hidden layers (`--hidden 16` or `--hidden 32,32`). Plot validation MAE against parameters.
- [ ] Compare training MAE with validation MAE for your chosen size. Is it overfitting?
- [ ] Work out the per-layer table (parameters, bytes, MACs) by hand, then **code (`app/main.py`):** make `/model_info` compute it from the network itself.
- [ ] Choose your final model and write down the MAE you expect on the company's 400 unseen readings.

### Mission 3: Serve it, and time every stage
- [ ] Predict the inference time and the laptop-to-Codespace round trip.
- [ ] **Code (`app/main.py`):** give every input a range, so glitches get HTTP 422. Base the ranges on what a real sensor can report, not on your data's smallest and largest values: a real reading just outside your data must still get an answer. Test three glitches at `/docs`.
- [ ] **Code (`app/main.py`):** time `normalise`, `inference` and `build` the same way `parse` is timed.
- [ ] From your **laptop**: `python client/bench.py --url <address> --n 200`. Record the first request, then p50/p95/p99 of the round trip and every stage.
- [ ] Draw where the time goes in a typical request (server stages + network and framework). Was the LLM right?
- [ ] Run it again with `--no-keepalive`, and once inside the Codespace against `http://localhost:8000`. Explain both differences.
- [ ] Time 400 single `/predict` calls against one `/predict_batch` call of 400 rows. Explain the difference.

### Mission 4: Would it fit inside the doorbell?
ESP32-S3: about 512 KB SRAM (roughly 300 KB free for your program), 8 MB PSRAM, 16 MB flash.
- [ ] For your model: bytes for weights and biases as float32 and int8, and the largest activation memory needed at once.
- [ ] **Code (`app/main.py`):** add `total_bytes_int8`, `fits_300kb_sram_f32` and `fits_300kb_sram_int8` to `/model_info`.
- [ ] Use the same function to fill in the table for 5 → w → w → 1 at w = 32, 128, 256, 512 and 2048: parameters, float32 size, int8 size, fits in 300 KB SRAM (float32 and int8), fits in 8 MB PSRAM, MACs.
- [ ] Find the exact width where the weights stop fitting in 300 KB, at float32 and at int8.
- [ ] Estimate one prediction's time on the board. State the speed you assume and where it came from.
- [ ] Argue: should the doorbell call your server or run the model itself? Use latency, privacy, cost, power, and what happens when the Wi-Fi is down.

---

## 4. What to hand in, and how it is checked

**Last date of submission: Monday, 23 November 2026.** How to submit will be announced. You will need:
1. **Your endpoint address**, e.g. `https://<name>-8000.app.github.dev`
2. **Your repository link**, with every TODO done and the `model/` folder committed

**Keep the endpoint running during the check window** your instructor announces: Codespace
started, `uvicorn` running, port 8000 set to **Public** (check it again after any restart). A Codespace stops itself when idle, so
in GitHub **Settings → Codespaces → Default idle timeout**, set it as long as it goes before
the window, and keep the Codespace open in a browser tab while the window is on. If the script
cannot reach your endpoint, it cannot give you marks.

The script checks:

| Check | What it does | To pass |
|---|---|---|
| Health | `GET /health` | `"status": "ok"` and **your** team number |
| Accuracy | Sends the company's 400 hidden readings to `/predict` one at a time | Error (MAE) close to what a good small network gets on your data |
| Per-layer table | `GET /model_info` | Every layer's params, bytes and MACs right; layers connect 5 → … → 1; totals, int8 bytes and both "fits" answers right; `total_params` equals `/health`'s `params` |
| Broken readings | Sends five broken readings | All five refused with a 4xx error, none answered |
| Stage timing | Reads `timings_ms` on every reply | `parse`, `normalise`, `inference`, `build` and `server_total` on every reply |
| Batch | Sends 50 readings to `/predict_batch` | Same answers as 50 single `/predict` calls |

Before you submit, test it yourself from your laptop:
```
python client/bench.py --url <your address> --n 50
```
and open `<your address>/health` and `<your address>/model_info` in a browser.

You don't hand in the missions. They are how you get to a good endpoint, and every number
the script checks comes out of them.

## 5. Quick reference

**The API (do not change the names or shapes):**
```
GET  /health          -> {"status": "ok", "team": 7, "params": 1281, "val_mae": 3.9}
GET  /model_info      -> {"layers": [{"name": "linear1", "in": 5, "out": 32, "params": 192,
                                      "bytes_f32": 768, "macs": 160}, ...],
                          "total_params": 1281, "total_bytes_f32": 5124, "total_bytes_int8": 1281,
                          "total_macs": 1216, "fits_300kb_sram_f32": true, "fits_300kb_sram_int8": true}
POST /predict         {"battery_v": 3.86, "temp_c": 28.0, "rings_per_day": 5,
                       "wifi_rssi_dbm": -66, "age_months": 22}
                      -> {"days_left": 97.4, "timings_ms": {"parse": .., "normalise": ..,
                          "inference": .., "build": .., "server_total": ..}}
POST /predict_batch   {"rows": [reading, ...]} -> {"days_left": [...], "timings_ms": {...}}
broken reading        -> HTTP 422 with the reason
```

**Columns:** `battery_v` (V), `temp_c` (°C), `rings_per_day`, `wifi_rssi_dbm` (dBm, closer to 0 is stronger), `age_months`, `days_left` (the answer).

**Useful commands**
```
python train.py --team NN --hidden 32,32 --epochs 1500
uvicorn app.main:app --host 0.0.0.0 --port 8000
python client/bench.py --url <address> --n 200 [--no-keepalive]
```

**Counting a `Linear(a, b)` layer:** a×b weights + b biases; a×b MACs per prediction; float32 = 4 bytes, int8 = 1 byte.

---

## 6. Rules for using an LLM

- Use it to explain, to check your reasoning, to explain error messages, and for hints. Do not use it for finished code.
- Your measurements are the judge. If an LLM makes a claim you can test, test it.
- Everyone must be able to explain every line you submit.
- Everyone in the team should understand how each part of the endpoint works.

## 7. When things go wrong

| Problem | Try |
|---|---|
| The Codespace is still installing | The first start installs PyTorch. Wait; don't rebuild |
| `No module named torch` (or fastapi) | The `.devcontainer` folder was not uploaded. Run `pip install -r requirements.txt` |
| `/health` works in the Codespace but not from your laptop | Port 8000 is still Private. Set it to Public |
| The address stopped working | The Codespace went idle. Start it, rerun `uvicorn`, and set port 8000 to Public again |
| `ModuleNotFoundError: app` | Run commands from the top folder, not from inside `app/` |
| `model/config.json` not found | Run `train.py` first, and commit `model/` |
| `/model_info` is wrong after changing the layer sizes | Compute it from `net.net`, not by hand |
| A result looks impossible | Good: that is the Skeptic's job. Find another explanation and test it |

Still stuck? Ask a teammate, then an LLM (and check what it says), then your instructor.

*Predict honestly. Measure carefully. Explain everything.*
