# Scheduling-model k6 profile

**Run date:** 2026-09-03
**Source revision:** `489e60fd684098bf9810b14dd41cc774bb75eb77`

## Host and toolchain

The following commands produced the profile facts:

```bash
nproc
lscpu | rg 'Model name'
free -h
k6 version
```

- Logical CPUs: `12`
- CPU: `Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz`
- Memory: `62Gi` total, `45Gi` available at collection time
- k6: `v2.2.0` (`go1.26.5`, `linux/amd64`)

## Service image and invocation

The service was built from this repository's `Dockerfile`:

```bash
docker build --tag enginedge-scheduling-model:k6-20260903 .
docker run --detach --name enginedge-scheduling-model-k6 \
  --publish 127.0.0.1:8000:3009 \
  --env KAFKA_BROKERS=127.0.0.1:1 \
  enginedge-scheduling-model:k6-20260903
```

The locally built image digest was
`localhost/enginedge-scheduling-model@sha256:fbd1d13b04bc352333fc7381ce589c4ead34894527f25a9605991c6dabc7c2ea`.
Its `python:3.9-slim` base image digest was
`docker.io/library/python@sha256:2d97f6910b16bd338d3060f261f53f144965f755599aab1acda1e13cf1731b1b`.

`KAFKA_BROKERS` intentionally points to an unused local port. Kafka logging is
best-effort and the tested API paths do not require a Kafka broker.

## Driven endpoints

| Endpoint | Container port | Host address | k6 script |
| --- | --- | --- | --- |
| `GET /health` | `3009` | `http://127.0.0.1:8000/health` | `k6/health.js` |
| `POST /predict-slots` | `3009` | `http://127.0.0.1:8000/predict-slots` | `k6/predict-slots.js` |

The prediction request body is the tracked
[`k6/fixtures/predict-slots.json`](k6/fixtures/predict-slots.json) fixture.
Each script uses one virtual user with a ten-second ramp up, a thirty-second
steady interval, and a ten-second ramp down.

## Scope

This is a single-host container measurement of the standalone scheduling-model
service, not a Kubernetes or full-platform benchmark. It does not establish
throughput for the former monorepo, worker integration, Kafka, frontend paths,
or a production deployment.
