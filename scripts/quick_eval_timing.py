#!/usr/bin/env python3
import os
import time
import json
import random
import statistics
import argparse
from typing import List, Tuple, Optional

import pandas as pd
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def sample_models(models_csv: str, k: int, seed: int) -> List[str]:
    df = pd.read_csv(models_csv)
    if "name" not in df.columns:
        raise ValueError("Expected column 'name' in models CSV")
    rng = random.Random(seed)
    names = df["name"].dropna().unique().tolist()
    if len(names) < k:
        k = len(names)
    return rng.sample(names, k)


def sample_prompts(prompts_csv: str, k: int, seed: int) -> List[str]:
    df = pd.read_csv(prompts_csv)
    if "prompt" not in df.columns:
        raise ValueError("Expected column 'prompt' in prompts CSV")
    prompts = [p for p in df["prompt"].dropna().tolist() if isinstance(p, str) and p.strip()]
    rng = random.Random(seed)
    if len(prompts) < k:
        k = len(prompts)
    return rng.sample(prompts, k)


def hf_generate(model: str, prompt: str, token: str, timeout: float, max_new_tokens: int) -> Tuple[bool, float, str]:
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "return_full_text": False,
        },
        "options": {
            "wait_for_model": True
        }
    }
    start = time.perf_counter()
    try:
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
        elapsed = time.perf_counter() - start
        ok = resp.status_code == 200
        text = ""
        if ok:
            try:
                data = resp.json()
                # API may return list of dicts with 'generated_text'
                if isinstance(data, list) and data and isinstance(data[0], dict):
                    text = data[0].get("generated_text", "")
                else:
                    text = str(data)
            except Exception:
                text = resp.text
        else:
            text = resp.text
        return ok, elapsed, text
    except Exception as e:
        elapsed = time.perf_counter() - start
        return False, elapsed, str(e)


def load_local_model(model_name: str,
                     dtype: str,
                     device_map: str,
                     revision: str | None = None):
    torch_dtype = None
    if dtype:
        d = dtype.lower()
        if d in ["float16", "fp16", "torch.float16"]:
            torch_dtype = torch.float16
        elif d in ["bfloat16", "bf16", "torch.bfloat16"]:
            torch_dtype = torch.bfloat16
        elif d in ["float32", "fp32", "torch.float32"]:
            torch_dtype = torch.float32

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, revision=revision)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        revision=revision,
    )
    return tok, mdl


def local_generate(tok: AutoTokenizer,
                   mdl: AutoModelForCausalLM,
                   prompt: str,
                   max_new_tokens: int,
                   device: str) -> Tuple[bool, float, str]:
    try:
        inputs = tok(prompt, return_tensors="pt")
        if device == "cuda" and torch.cuda.is_available():
            inputs = {k: v.to(mdl.device) for k, v in inputs.items()}
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            out = mdl.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tok.eos_token_id,
            )
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        gen_text = tok.decode(out[0], skip_special_tokens=True)
        return True, elapsed, gen_text
    except Exception as e:
        return False, 0.0, str(e)


def run_benchmark(models_csv: str,
                  prompts_csv: str,
                  num_models: int,
                  num_prompts: int,
                  seed: int,
                  timeout: float,
                  max_new_tokens: int,
                  sleep_between: float,
                  output_csv: str,
                  hf_token: str,
                  use_local: bool,
                  device_map: str,
                  dtype: str,
                  revision: Optional[str]) -> None:
    models = sample_models(models_csv, num_models, seed)
    prompts = sample_prompts(prompts_csv, num_prompts, seed + 1)

    token = hf_token or os.environ.get("HF_TOKEN", "")
    durations: List[float] = []
    records = []

    for model in models:
        if use_local:
            try:
                tok, mdl = load_local_model(model, dtype=dtype, device_map=device_map, revision=revision)
                current_device = "cuda" if any(p.is_cuda for p in mdl.parameters()) else "cpu"
            except Exception as e:
                records.append({"model": model, "elapsed_seconds": float("nan"), "ok": False, "error": f"load_failed: {e}"})
                continue
            for prompt in prompts:
                ok, elapsed, gen_text = local_generate(tok, mdl, prompt, max_new_tokens, device=current_device)
                durations.append(elapsed)
                records.append({
                    "model": model,
                    "elapsed_seconds": elapsed,
                    "ok": ok,
                    "gen_text": gen_text
                })
                if sleep_between > 0:
                    time.sleep(sleep_between)
            try:
                del tok
                del mdl
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        else:
            for prompt in prompts:
                ok, elapsed, gen_text = hf_generate(model, prompt, token, timeout, max_new_tokens)
                durations.append(elapsed)
                records.append({
                    "model": model,
                    "elapsed_seconds": elapsed,
                    "ok": ok,
                    "gen_text": gen_text
                })
                if sleep_between > 0:
                    time.sleep(sleep_between)

    mean_s = statistics.mean(durations) if durations else float("nan")
    std_s = statistics.pstdev(durations) if len(durations) > 1 else 0.0

    print(f"Total requests: {len(durations)}")
    print(f"Average time per prompt: {mean_s:.3f} s")
    print(f"Std dev: {std_s:.3f} s")

    if output_csv:
        out_df = pd.DataFrame.from_records(records)
        out_df.to_csv(output_csv, index=False)
        print(f"Wrote per-call timings to {output_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_csv", type=str, required=True,
                        help="Path to open-llm-leaderboard.csv")
    parser.add_argument("--prompts_csv", type=str, required=True,
                        help="Path to prompts CSV with a 'prompt' column")
    parser.add_argument("--num_models", type=int, default=10)
    parser.add_argument("--num_prompts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max_new_tokens", type=int, default=16,
                        help="Small number to approximate a single forward pass")
    parser.add_argument("--sleep_between", type=float, default=0.5,
                        help="Seconds to sleep between requests to avoid rate limits")
    parser.add_argument("--output_csv", type=str, default="",
                        help="Optional path to save per-call timings")
    parser.add_argument("--hf_token_file", type=str, default="",
                        help="Path to a file containing the HF API token")
    parser.add_argument("--use_local", action="store_true",
                        help="Use local transformers models instead of HF Inference API")
    parser.add_argument("--device_map", type=str, default="auto",
                        help="transformers device_map (e.g., 'auto', 'cpu')")
    parser.add_argument("--dtype", type=str, default="float16",
                        help="torch dtype for model weights: float16|bfloat16|float32")
    parser.add_argument("--revision", type=str, default=None,
                        help="Optional model revision/sha to pin")

    args = parser.parse_args()
    hf_token = ""
    if args.hf_token_file:
        try:
            with open(args.hf_token_file, "r") as f:
                hf_token = f.read().strip()
        except Exception as e:
            raise RuntimeError(f"Failed to read HF token from file: {e}")
    run_benchmark(
        models_csv=args.models_csv,
        prompts_csv=args.prompts_csv,
        num_models=args.num_models,
        num_prompts=args.num_prompts,
        seed=args.seed,
        timeout=args.timeout,
        max_new_tokens=args.max_new_tokens,
        sleep_between=args.sleep_between,
        output_csv=args.output_csv,
        hf_token=hf_token,
        use_local=args.use_local,
        device_map=args.device_map,
        dtype=args.dtype,
        revision=args.revision,
    )


if __name__ == "__main__":
    main()


