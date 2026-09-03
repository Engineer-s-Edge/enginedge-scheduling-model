# Scheduling-model k6 results

**Run date:** 2026-09-03
**Source revision:** `489e60fd684098bf9810b14dd41cc774bb75eb77`

Run commands:

```bash
k6 run --summary-export load-tests/results/health-2026-09-03.json \
  load-tests/k6/health.js | tee load-tests/results/health-2026-09-03.txt
k6 run --summary-export load-tests/results/predict-slots-2026-09-03.json \
  load-tests/k6/predict-slots.js | tee load-tests/results/predict-slots-2026-09-03.txt
```

## Measured endpoints

### `GET /health`

The reported HTTP request rate is `1122.3223196973547` requests per second:

```bash
jq '.metrics.http_reqs.rate' load-tests/results/health-2026-09-03.json
```

The p95 request duration is `1.4770235999999994` milliseconds:

```bash
jq '.metrics.http_req_duration["p(95)"]' load-tests/results/health-2026-09-03.json
```

The HTTP failure rate is `0`:

```bash
jq '.metrics.http_req_failed.value' load-tests/results/health-2026-09-03.json
```

### `POST /predict-slots`

The reported HTTP request rate is `442.91250863097855` requests per second:

```bash
jq '.metrics.http_reqs.rate' load-tests/results/predict-slots-2026-09-03.json
```

The p95 request duration is `3.0442377499999997` milliseconds:

```bash
jq '.metrics.http_req_duration["p(95)"]' load-tests/results/predict-slots-2026-09-03.json
```

The HTTP failure rate is `0`:

```bash
jq '.metrics.http_req_failed.value' load-tests/results/predict-slots-2026-09-03.json
```

## Headline

`POST /predict-slots` reported `442.91250863097855` requests per second under
the committed one-virtual-user k6 profile. This is a single-host container
measurement of the standalone scheduling-model service, not a full-platform or
production-cluster measurement.

## Not measured

- Kafka delivery or consumer lag: `[NEEDS MEASUREMENT]`
- Worker-to-model end-to-end latency: `[NEEDS MEASUREMENT]`
- Frontend request behavior: `[NEEDS MEASUREMENT]`
- Model cold-start latency and external model-download time: `[NEEDS MEASUREMENT]`
- Kubernetes or multi-replica throughput: `[NEEDS MEASUREMENT]`

## Authorship

The commit history for this standalone repository was collected with:

```bash
git shortlog -sne --all
```

The output at the measured revision names Chris Alexander Pop, Chris Pop, and
`google-labs-jules[bot]`. This result describes this repository's standalone
service; it does not establish authorship or performance claims for the
deprecated monorepo or other split EnginEdge repositories.
