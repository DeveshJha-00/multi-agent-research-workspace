"""Run the reference-based Chat benchmark against a live API."""

import argparse
import csv
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import requests


def _load_cases(path: Path) -> list[dict]:
    cases = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            cases.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return cases


def _request(response: requests.Response) -> dict:
    response.raise_for_status()
    return response.json()


def _upload_fixtures(base_url: str, session_id: str, cases: list[dict]) -> None:
    uploaded = set()
    for case in cases:
        fixture = Path(case["fixture"])
        if fixture in uploaded:
            continue
        with fixture.open("rb") as handle:
            _request(
                requests.post(
                    f"{base_url}/rag/documents/upload",
                    files={"file": (fixture.name, handle, "text/plain")},
                    headers={
                        "X-Session-ID": session_id,
                        "X-Description": case.get("description", "Evaluation fixture"),
                    },
                    timeout=300,
                )
            )
        uploaded.add(fixture)


def _wait_for_evaluation(base_url: str, session_id: str, evaluation_id: str) -> dict:
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        job = _request(
            requests.get(
                f"{base_url}/rag/evaluations/{evaluation_id}",
                headers={"X-Session-ID": session_id},
                timeout=60,
            )
        )
        if job["status"] in {"completed", "failed"}:
            return job
        time.sleep(1)
    raise TimeoutError(f"Evaluation {evaluation_id} did not finish within 15 minutes")


def run_benchmark(base_url: str, dataset: Path, output_dir: Path) -> dict:
    cases = _load_cases(dataset)
    session_id = f"ragas-benchmark-{uuid4().hex}"
    _upload_fixtures(base_url, session_id, cases)
    results = []
    for case in cases:
        response_started = time.monotonic()
        answer = _request(
            requests.post(
                f"{base_url}/rag/query",
                json={"query": case["query"], "session_id": session_id},
                timeout=300,
            )
        )
        response_duration = round(time.monotonic() - response_started, 3)
        created = _request(
            requests.post(
                f"{base_url}/rag/evaluations",
                json={"response_id": answer["response_id"], "reference": case["reference"]},
                headers={
                    "X-Session-ID": session_id,
                    "Idempotency-Key": f"benchmark-{case['case_id']}-{uuid4()}",
                },
                timeout=60,
            )
        )
        evaluation = _wait_for_evaluation(
            base_url, session_id, created["evaluation_id"]
        )
        results.append(
            {
                "case_id": case["case_id"],
                "query": case["query"],
                "reference": case["reference"],
                "answer": answer["content"],
                "route": answer["route"],
                "expected_route": case["expected_route"],
                "route_correct": answer["route"] == case["expected_route"],
                "response_duration_seconds": response_duration,
                "evaluation_duration_seconds": evaluation.get("duration_seconds"),
                "evaluation_status": evaluation["status"],
                "metrics": evaluation.get("metrics", {}),
            }
        )

    metric_values: dict[str, list[float]] = defaultdict(list)
    for result in results:
        for name, metric in result["metrics"].items():
            if metric.get("status") == "completed" and metric.get("score") is not None:
                metric_values[name].append(float(metric["score"]))
    summary = {
        "cases": len(results),
        "route_accuracy": sum(item["route_correct"] for item in results) / len(results),
        "metric_averages": {
            name: sum(values) / len(values) for name, values in sorted(metric_values.items())
        },
        "failures": sum(item["evaluation_status"] != "completed" for item in results),
    }
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "session_id": session_id,
        "summary": summary,
        "results": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"ragas-benchmark-{timestamp}.json"
    csv_path = output_dir / f"ragas-benchmark-{timestamp}.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    metric_names = sorted(metric_values)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "route",
                "expected_route",
                "route_correct",
                "response_duration_seconds",
                "evaluation_duration_seconds",
                "evaluation_status",
                *metric_names,
            ],
        )
        writer.writeheader()
        for result in results:
            row = {key: result.get(key) for key in writer.fieldnames}
            for name in metric_names:
                row[name] = result["metrics"].get(name, {}).get("score")
            writer.writerow(row)
    print(json.dumps({"summary": summary, "json": str(json_path), "csv": str(csv_path)}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--dataset", type=Path, default=Path("evals/rag_chat.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("evals/results"))
    args = parser.parse_args()
    run_benchmark(args.base_url.rstrip("/"), args.dataset, args.output_dir)


if __name__ == "__main__":
    main()
