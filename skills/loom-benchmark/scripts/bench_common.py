"""Shared helpers for the Loom benchmark scripts.

A *package* is a directory with model.json + tokenizer.json + llm_config.json
(written by loom-port-llm/starter/build.py). llm_config.json's "chat" block is
the single source of truth for prompt rendering, so llama.cpp (golden, fed raw
token ids) and Loom (fed chat messages) see byte-identical prompts.
"""
import json
import os
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")

PROMPTS = [
    "What is the capital of France? Answer in one word.",
    "Write a Python function that returns the n-th Fibonacci number.",
    "Explain what a hash map is in two sentences.",
    "What is 17 * 23?",
    "List three prime numbers greater than 50.",
    "Translate 'good morning' into Spanish and German.",
    "Write a haiku about compilers.",
    "What does the Unix command `grep -r` do?",
    "Why is the sky blue? Keep it short.",
    "Give me a SQL query that counts rows in a table named users.",
    "What is the difference between a process and a thread?",
    "Name the planets of the solar system in order.",
    "Fix the bug: def add(a, b): return a - b",
    "What is the time complexity of binary search?",
    "Summarize the plot of Romeo and Juliet in one sentence.",
    "Write a bash one-liner to count lines in all .py files.",
    "What year did the first moon landing happen?",
    "Convert 100 degrees Fahrenheit to Celsius.",
    "What is a mixture-of-experts language model?",
    "Write a regular expression that matches an email address.",
]


def doc(n):
    """Real-text document of ~n tokens (Loom docs markdown): 1000 2000 8000 16000 32000."""
    return open(os.path.join(DATA, f"doc_{n}.txt")).read()


class Package:
    def __init__(self, path):
        from tokenizers import Tokenizer
        self.path = path
        self.cfg = json.load(open(os.path.join(path, "llm_config.json")))
        self.chat = self.cfg["chat"]
        self.model_id = self.cfg["model_id"]
        self.tok = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))

    def render(self, user):
        """Exactly what control.loom renders for one user message + generation prompt."""
        c = self.chat
        bos = self.tok.id_to_token(self.cfg["bos_id"]) if self.cfg.get("bos_id") is not None else ""
        return bos + c.get("default_system", "") + c["user_prefix"] + user + c["suffix"] + c["generation_prompt"]

    def ids(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def raw_from_chat(self, message, finished):
        """Rebuild the model's raw output text from Loom's parsed channels so it can be
        re-tokenized and compared token-by-token with the golden ids."""
        c = self.chat
        reasoning = message.get("reasoning_content")
        content = message.get("content") or ""
        if not c.get("starts_in_reasoning"):
            return content
        end = c.get("think_end", "</think>")
        after = c["think_close"].split(end, 1)[1] if end in c["think_close"] else ""
        text = reasoning or ""
        if content or finished:
            text += end + after + content
        return text


def stream(url, model, messages, max_tokens, extra=None):
    """Streams one chat completion; returns client-side TTFT, decode tok/s and usage."""
    t0 = time.perf_counter()
    first = None
    n = pt = 0
    body = {"messages": messages, "max_tokens": max_tokens, "stream": True, "temperature": 0,
            "stream_options": {"include_usage": True}}
    if model:
        body["model"] = model
    body.update(extra or {})
    with requests.post(url + "/v1/chat/completions", json=body, stream=True,
                       headers={"Content-Type": "application/json"}) as r:
        for line in r.iter_lines():
            if not line.startswith(b"data: {"):
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                n = d["usage"]["completion_tokens"]
                pt = d["usage"]["prompt_tokens"]
            for ch in d.get("choices", []):
                dl = ch.get("delta", {})
                if (dl.get("content") or dl.get("reasoning_content")) and first is None:
                    first = time.perf_counter() - t0
    total = time.perf_counter() - t0
    return dict(ttft=first, total=total, tokens=n, prompt_tokens=pt,
                decode_tps=(n - 1) / (total - first) if n > 1 and first else None)
