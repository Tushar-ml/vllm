# SPDX-License-Identifier: Apache-2.0
"""Small-model Megakernel benchmark matrix helper.

This script compares latency across:
- vanilla eager
- torch compile default
- Megakernel eager
- Megakernel FULL_AND_PIECEWISE
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.request


def wait_health(timeout_s: int) -> bool:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def request_latency_ms(model: str, prompt: str, max_tokens: int = 32) -> float:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:8000/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as r:
        r.read()
    return (time.perf_counter() - t0) * 1000


def summarize(values: list[float]) -> dict[str, float | int]:
    vals = sorted(values)
    return {
        "avg_ms": round(sum(vals) / len(vals), 3),
        "p50_ms": round(vals[len(vals) // 2], 3),
        "p95_ms": round(vals[int(len(vals) * 0.95) - 1], 3),
        "runs": len(vals),
    }


def run_mode(
    model: str,
    prompt: str,
    args_extra: list[str],
    env_extra: dict[str, str],
    warmup_runs: int,
    measured_runs: int,
    timeout_s: int,
    log_path: str,
) -> dict[str, float | int | str]:
    base_env = os.environ.copy()
    base_env["PYTHONPATH"] = "/home/ubuntu/vllm:" + base_env.get("PYTHONPATH", "")
    base_env["LD_LIBRARY_PATH"] = (
        "/home/ubuntu/miniconda3/envs/vllm_env/lib:"
        + base_env.get("LD_LIBRARY_PATH", "")
    )
    base_env.update(env_extra)
    args = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "1",
    ] + args_extra
    subprocess.run("pkill -f 'vllm.entrypoints.openai.api_server' || true", shell=True)
    subprocess.run("pkill -f 'vllm.v1.engine.core' || true", shell=True)
    time.sleep(2)
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            args,
            cwd="/home/ubuntu/vllm",
            env=base_env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        try:
            if not wait_health(timeout_s):
                raise RuntimeError(f"health check failed for mode: {log_path}")
            for _ in range(warmup_runs):
                request_latency_ms(model, prompt)
            vals = [request_latency_ms(model, prompt) for _ in range(measured_runs)]
            out = summarize(vals)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except Exception:
                proc.kill()
                proc.wait(timeout=10)
    out["log_path"] = log_path
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--prompt", default="Write one short sentence about Paris.")
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--measured-runs", type=int, default=15)
    parser.add_argument("--timeout-s", type=int, default=420)
    parser.add_argument(
        "--output-json", default="/tmp/megakernel_small_model_matrix.json"
    )
    args = parser.parse_args()

    mk_env = {
        "VLLM_MEGAKERNEL_ON": "1",
        "VLLM_MEGAKERNEL_FAMILIES": "llama_small,qwen_small,mistral_small",
        "VLLM_MEGAKERNEL_MK_LLAMA_PATH": "/home/ubuntu/Megakernels/demos/low-latency-llama",
        "VLLM_MEGAKERNEL_ROOT": "/home/ubuntu/Megakernels",
        "THUNDERKITTENS_ROOT": "/home/ubuntu/Megakernels/ThunderKittens",
        "MEGAKERNELS_ROOT": "/home/ubuntu/Megakernels",
    }

    results = {
        "vanilla_eager": run_mode(
            args.model,
            args.prompt,
            ["--enforce-eager"],
            {"VLLM_MEGAKERNEL_ON": "0"},
            args.warmup_runs,
            args.measured_runs,
            args.timeout_s,
            "/tmp/vanilla_eager.log",
        ),
        "torch_compile_default": run_mode(
            args.model,
            args.prompt,
            [],
            {"VLLM_MEGAKERNEL_ON": "0"},
            args.warmup_runs,
            args.measured_runs,
            args.timeout_s,
            "/tmp/torch_compile_default.log",
        ),
        "megakernel_eager": run_mode(
            args.model,
            args.prompt,
            ["--enforce-eager"],
            mk_env,
            args.warmup_runs,
            args.measured_runs,
            args.timeout_s,
            "/tmp/megakernel_eager.log",
        ),
        "megakernel_full_and_piecewise": run_mode(
            args.model,
            args.prompt,
            ["-cc.cudagraph_mode=FULL_AND_PIECEWISE"],
            mk_env,
            args.warmup_runs,
            args.measured_runs,
            args.timeout_s,
            "/tmp/megakernel_full_and_piecewise.log",
        ),
    }
    out = {"model": args.model, "prompt": args.prompt, "results": results}
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
