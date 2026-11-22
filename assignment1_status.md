# Module 1: How Long Will the Battery Last?

## Mission completion cross-reference — Team 15

This section maps the README's mission expectations to the work
completed. The missions are milestones; the final deliverable remains a
working endpoint and repository.

### Mission 1: Meet the data and set the bar

| README expectation                                                                           | Status / result                                                                                            |
|----------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| Predict mean-baseline error                                                                  | Completed; mean-baseline validation MAE: **60.63 days**.                                                   |
| Plot histograms and target-versus-input relationships                                        | Completed in the EDA notebook.                                                                             |
| Identify and count impossible/glitch readings                                                | Completed; **24 of 2,000** rows removed using `battery_v == 0`, `temp_c == -127`, or `wifi_rssi_dbm == 0`. |
| Report baseline 1 validation MAE                                                             | **60.63 days**.                                                                                            |
| Implement `np.linalg.lstsq` linear regression, report MAE and plot predictions against truth | Completed; validation MAE: **27.70 days**.                                                                 |
| Run the neural network and record its MAE                                                    | Completed; further architecture experiments were conducted in Mission 2.                                   |

### Mission 2: A small network, opened up

| README expectation                                                                    | Status / result                                                                                                                                                                                                        |
|---------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Hand-calculate parameters, float32 bytes and MACs for 5 → 32 → 32 → 1                 | Completed as part of model-size analysis.                                                                                                                                                                              |
| Normalise using training rows only; compare with and without normalisation            | Completed; training-set mean and standard deviation are reused for validation and inference.                                                                                                                           |
| Train required hidden-width configurations and plot validation MAE against parameters | Completed, with additional architecture experiments.                                                                                                                                                                   |
| Compare training and validation MAE for the chosen model                              | Completed in the experiment notebook.                                                                                                                                                                                  |
| Produce per-layer parameter/byte/MAC calculations and implement `/model_info`         | Completed; `/model_info` derives the table from the deployed network.                                                                                                                                                  |
| Choose final model and record expected MAE on the company's 400 unseen readings       | Completed. Selected architecture: **5 → 32 → 32 → 32 → 16 → 1**, **2,849 parameters**, validation MAE **5.60 days**. The company's 400 hidden readings are evaluation data, not the locally generated batch benchmark. |

### Mission 3: Serve it, and time every stage

| README expectation                                                                 | Status / result                                                                                                                                                                |
|------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Predict inference time and laptop-to-Codespace round trip                          | Qualitative expectation documented in `mission3_benchmark_notes.md`: network/framework overhead was expected to dominate. No exact numerical estimate was recorded beforehand. |
| Add realistic input ranges and return HTTP 422 for invalid readings; test glitches | Implemented in `app/main.py` using Pydantic constraints.                                                                                                                       |
| Time normalise, inference and build, in addition to parse                          | Implemented; responses include `parse`, `normalise`, `inference`, `build`, and `server_total`.                                                                                 |
| Run laptop-to-Codespace benchmark; record first request and p50/p95/p99            | Completed from Windows laptop. Keep-alive p50 round trip: **38.636 ms**; p50 inference: **0.198 ms**.                                                                          |
| Explain time distribution and compare with LLM expectation                         | Completed in `mission3_benchmark_notes.md`; network/framework residual was approximately **99.2%** of p50 round-trip time with keep-alive.                                     |
| Repeat with `--no-keepalive` and run inside Codespace against localhost            | Completed. No-keepalive p50 round trip: **756.963 ms**; Codespace localhost p50: **2.127 ms**.                                                                                 |
| Compare 400 individual `/predict` calls with one 400-row `/predict_batch` call     | Completed laptop-to-Codespace: **19,681.049 ms** versus **93.892 ms**; **209.61×** speedup and zero prediction mismatches.                                                     |

See `mission3_benchmark_notes.md` for full tables, interpretation, and
reproduction commands.

### Mission 4: Would it fit inside the doorbell?

| README expectation                                                                                 | Status / result                                                                                                                                                  |
|----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Calculate float32/int8 weight sizes and largest activation memory for selected model               | Completed in `mission4_edge_analysis.md`: **11,396 bytes FP32**, **2,849 bytes theoretical INT8**; largest activation buffer **256 bytes FP32 / 64 bytes INT8**. |
| Implement int8 size and SRAM-fit fields in `/model_info`                                           | Completed in `app/main.py`.                                                                                                                                      |
| Compare widths 32, 128, 256, 512 and 2048 for memory and MACs                                      | Completed in `mission4_width_sweep.md`.                                                                                                                          |
| Find exact widths at which weights stop fitting in 300 KB SRAM                                     | Completed: maximum width **273** for FP32 and **550** for theoretical INT8 under the stated formula.                                                             |
| Estimate prediction time on ESP32-S3 and state assumptions/source                                  | Completed in `mission4_edge_analysis.md`; approximately **22 µs** for the optimised INT8 estimate, with caveats.                                                 |
| Argue server versus on-device inference using latency, privacy, cost, power and Wi-Fi availability | Completed; recommendation: run inference on the doorbell itself.                                                                                                 |

See `mission4_width_sweep.md` and `mission4_edge_analysis.md` for
calculations, assumptions and sources.

## Final submission readiness

The assignment README says the final hand-in is the working endpoint
address and repository link, with the `model/` folder committed. Before
the instructor's check window:

- Confirm `model/model.pt` and `model/config.json` are committed and
  correspond to the selected architecture.
- Start the Codespace and run Uvicorn from the repository root.
- Set port 8000 to **Public** after every Codespace restart.
- Verify `/health` and `/model_info` from outside the Codespace.
- Keep the endpoint running during the announced check window.

The benchmark CSVs and local 400-row batch input are supporting
experiment artifacts. The required final deliverables remain the
endpoint and repository.
