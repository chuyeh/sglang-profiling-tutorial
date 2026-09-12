#!/usr/bin/env python3
"""Build compact JSON/CSV summaries for an AgentX sweep campaign."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def nested(data: dict[str, Any], *keys: str) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows.append(
                {
                    "point_id": row["point_id"],
                    "tp": int(row["tp"]),
                    "conc": int(row["conc"]),
                    "kv_mode": row["kv_mode"],
                    "selected_in_last_run": row["selected_in_last_run"] == "yes",
                }
            )
    return rows


def summarize_point(
    campaign: Path, point: dict[str, Any], gpu_hour_price: float
) -> dict[str, Any]:
    point_dir = campaign / "points" / point["point_id"]
    result_path = point_dir / "agentx_result.json"
    marker_path = point_dir / "SUCCESS"
    row = dict(point)
    row["result_dir"] = str(point_dir)

    if not result_path.is_file():
        row["status"] = "incomplete" if point_dir.exists() else "pending"
        return row

    try:
        with result_path.open(encoding="utf-8") as handle:
            result = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        row.update(status="invalid", error=str(exc))
        return row

    per_gpu_tput = nested(
        result, "request_metrics", "throughput", "per_gpu", "total_tput_tps"
    )
    row.update(
        status="complete" if marker_path.is_file() else "unverified",
        successful_requests=result.get("num_requests_successful"),
        error_requests=nested(
            result, "request_accounting", "records_error_dropped"
        ),
        measured_seconds=nested(
            result, "request_metrics", "throughput", "duration_seconds"
        ),
        total_tokens_per_second=nested(
            result, "request_metrics", "throughput", "total", "tokens_per_second"
        ),
        tokens_per_second_per_gpu=per_gpu_tput,
        output_tokens_per_second_per_gpu=nested(
            result, "request_metrics", "throughput", "per_gpu", "output_tput_tps"
        ),
        p90_decode_interactivity=nested(
            result, "request_metrics", "latency", "intvty", "p90"
        ),
        p90_e2e_normalized_interactivity=nested(
            result, "request_metrics", "latency", "e2e_norm_intvty", "p90"
        ),
        p90_ttft_seconds=nested(
            result, "request_metrics", "latency", "ttft", "p90"
        ),
        gpu_kv_usage=nested(result, "server_metrics", "kv_cache", "gpu_usage_pct"),
        cpu_kv_usage=nested(result, "server_metrics", "kv_cache", "cpu_usage_pct"),
    )
    if isinstance(per_gpu_tput, (int, float)):
        row["input_output_tokens_per_dollar"] = (
            per_gpu_tput * 3600 / gpu_hour_price
        )
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--gpu-hour-price", type=float, default=1.50)
    args = parser.parse_args()

    campaign = args.campaign.resolve()
    manifest = campaign / "manifest.tsv"
    if not manifest.is_file():
        parser.error(f"missing manifest: {manifest}")
    if args.gpu_hour_price <= 0:
        parser.error("--gpu-hour-price must be positive")

    rows = [
        summarize_point(campaign, point, args.gpu_hour_price)
        for point in read_manifest(manifest)
    ]
    complete = sum(row["status"] == "complete" for row in rows)
    incomplete = sum(
        row["status"] in {"incomplete", "invalid", "unverified"} for row in rows
    )
    pending = sum(row["status"] == "pending" for row in rows)
    selected = sum(bool(row["selected_in_last_run"]) for row in rows)
    selected_complete = sum(
        row["selected_in_last_run"] and row["status"] == "complete" for row in rows
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "campaign_dir": str(campaign),
        "gpu_hour_list_price_usd": args.gpu_hour_price,
        "planned_points": len(rows),
        "selected_in_last_run": selected,
        "selected_completed_points": selected_complete,
        "completed_points": complete,
        "incomplete_points": incomplete,
        "pending_points": pending,
        "points": rows,
    }
    with (campaign / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    fieldnames = [
        "point_id",
        "tp",
        "conc",
        "kv_mode",
        "selected_in_last_run",
        "status",
        "successful_requests",
        "error_requests",
        "measured_seconds",
        "total_tokens_per_second",
        "tokens_per_second_per_gpu",
        "output_tokens_per_second_per_gpu",
        "p90_decode_interactivity",
        "p90_e2e_normalized_interactivity",
        "p90_ttft_seconds",
        "gpu_kv_usage",
        "cpu_kv_usage",
        "input_output_tokens_per_dollar",
        "result_dir",
        "error",
    ]
    with (campaign / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Summary: selected {selected_complete}/{selected}; "
        f"full matrix {complete}/{len(rows)}; {incomplete} incomplete; "
        f"{campaign / 'summary.csv'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
