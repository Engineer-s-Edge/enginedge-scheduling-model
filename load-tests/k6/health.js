import http from "k6/http";
import { check } from "k6";

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
  const response = http.get("http://127.0.0.1:8000/health");

  check(response, {
    "returns HTTP 200": (result) => result.status === 200,
  });
}
