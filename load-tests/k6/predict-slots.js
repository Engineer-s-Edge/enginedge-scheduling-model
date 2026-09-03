import http from "k6/http";
import { check } from "k6";

const requestBody = open("./fixtures/predict-slots.json");

export const options = {
  stages: [
    { duration: "10s", target: 1 },
    { duration: "30s", target: 1 },
    { duration: "10s", target: 0 },
  ],
  thresholds: {
    checks: ["rate>0.99"],
    http_req_failed: ["rate<0.01"],
  },
};

export default function () {
  const response = http.post(
    "http://127.0.0.1:8000/predict-slots",
    requestBody,
    { headers: { "Content-Type": "application/json" } },
  );

  check(response, {
    "returns HTTP 200": (result) => result.status === 200,
  });
}
