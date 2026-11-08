"""
Which Gemini models are FAST on your key?   python test_llm.py
Needs GEMINI_API_KEY (same terminal). Prints time per model; the fastest working one is what you want.
"""
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from google import genai
from google.genai import types

key = "".join(os.environ.get("GEMINI_API_KEY", "").split())
if not key:
    sys.exit("Set GEMINI_API_KEY first.")
client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=40_000))

names = []
for m in client.models.list():
    n = (m.name or "").replace("models/", "")
    if "flash" in n and "generateContent" in (m.supported_actions or ["generateContent"]) \
            and not any(b in n for b in ("image", "tts", "live", "audio", "embed", "native", "robotics", "computer")):
        names.append(n)
print("Flash models on your key:", names)

PROMPT = 'Return ONLY JSON {"distance_km": number, "quoted_fare": number}. Text: "majestic se indiranagar 11 km, driver 450 maang raha hai"'


def test(model, mode):
    kw = dict(temperature=0, response_mime_type="application/json")
    if mode == "minimal-thinking":
        kw["thinking_config"] = (types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL)
                                 if "gemini-3" in model else types.ThinkingConfig(thinking_budget=0))
    t = time.perf_counter()
    try:
        out = client.models.generate_content(model=model, contents=PROMPT,
                                             config=types.GenerateContentConfig(**kw)).text
        return model, mode, time.perf_counter() - t, "OK " + (out or "")[:50].replace("\n", " ")
    except Exception as e:  # noqa: BLE001
        return model, mode, time.perf_counter() - t, "FAIL " + str(e)[:90].replace("\n", " ")


jobs = [(n, m) for n in names for m in ("default", "minimal-thinking")]
with ThreadPoolExecutor(max_workers=6) as ex:
    results = list(ex.map(lambda a: test(*a), jobs))
print(f"\n{'model':32} {'mode':17} {'secs':>5}  result")
for model, mode, secs, res in sorted(results, key=lambda r: (not r[3].startswith("OK"), r[2])):
    print(f"{model:32} {mode:17} {secs:5.1f}  {res}")
ok = [r for r in results if r[3].startswith("OK")]
if ok:
    best = min(ok, key=lambda r: r[2])
    print(f"\nFASTEST: {best[0]} ({best[1]}, {best[2]:.1f}s)\nUse it:  $env:GEMINI_MODEL=\"{best[0]}\"")
else:
    print("\nNo model answered. Check the key / quota / region in Google AI Studio.")
