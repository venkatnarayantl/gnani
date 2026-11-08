"""
METER SACH - voice fare checker for auto-rickshaws (Bengaluru / Chennai / Delhi)

Voice in  : Gnani Prisma STT (language_code="indic-auto") - handles Kanglish / Hinglish / Tanglish
LLM       : ONLY turns the messy transcript into strict JSON (no keyword matching)
Maps      : looks up the road distance between the spoken start / destination (OpenStreetMap, or Google Maps if key set)
Python    : computes the official meter fare from a tariff table (exact math, no LLM guessing)
Voice out : Gnani Timbre TTS speaks the verdict back in the user's own language

Setup
-----
    pip install -r requirements.txt     # (needs: requests, numpy, gradio, gnani sdk, google-genai / anthropic)
    export GNANI_API_KEY="..."
    export GEMINI_API_KEY="..."        # or: export ANTHROPIC_API_KEY="..."
    export GOOGLE_MAPS_API_KEY="..."   # OPTIONAL: more accurate place lookup. Without it, free OpenStreetMap is used.
    python app.py

Optional env vars: LLM_PROVIDER (gemini|claude), GEMINI_MODEL, CLAUDE_MODEL, TTS_VOICE,
                   DENOISE (set 0 to turn noise removal off), MIC_RAW, SHARE
"""

import inspect
import json
import math
import os
import re
import tempfile
import time
import wave
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from functools import lru_cache
from zoneinfo import ZoneInfo

def _load_env():
    """Read KEY=VALUE lines from .env (next to app.py or in the current folder). No extra package needed."""
    here = os.path.dirname(os.path.abspath(__file__))
    for folder in (here, os.getcwd()):
        for name in (".env", ".env.txt"):  # Notepad sometimes saves .env as .env.txt
            path = os.path.join(folder, name)
            if os.path.isfile(path):
                for line in open(path, encoding="utf-8-sig").read().splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.replace("export ", "").strip()
                    v = v.strip().strip('"').strip("'")
                    if k and v:
                        os.environ[k] = v
                print(f"ENV      : loaded {path}")
                return
    print(f"ENV      : no .env file found in {here}")


_load_env()
print("ENV      : GNANI_API_KEY", "FOUND" if os.getenv("GNANI_API_KEY") else "MISSING",
      "| GEMINI_API_KEY", "FOUND" if os.getenv("GEMINI_API_KEY") else "MISSING")
import gradio as gr
import numpy as np
import requests
try:  # optional extra denoiser:  pip install noisereduce
    import noisereduce as _nr
except Exception:  # noqa: BLE001
    _nr = None
from gnani.stt import GnaniSTTClient
from gnani.tts import AudioConfig, GnaniTTSClient

# --------------------------------------------------------------------------- #
# 1. TARIFF TABLE  (the "source of truth" - Python does the math, not the LLM)
# --------------------------------------------------------------------------- #
# !! VERIFY every number below against the CURRENT official fare notification
# !! (transport department / RTO) before your final demo. Tariffs get revised.
TARIFFS = {
    "bengaluru": dict(
        label="Bengaluru",
        base_fare=36, base_km=2.0, per_km=18,
        night_start="22:00", night_end="05:00", night_pct=0.50,
        wait_free_min=5, wait_per_min=5 / 15,
    ),
    "chennai": dict(
        label="Chennai",
        base_fare=50, base_km=1.8, per_km=18,
        night_start="23:00", night_end="05:00", night_pct=0.50,
        wait_free_min=5, wait_per_min=3.5,
    ),
    "delhi": dict(
        label="Delhi",
        base_fare=30, base_km=1.5, per_km=11,
        night_start="23:00", night_end="05:00", night_pct=0.25,
        wait_free_min=0, wait_per_min=0.5,
    ),
}
DEFAULT_CITY = "bengaluru"
LANGS = {"hi-IN", "kn-IN", "ta-IN", "en-IN"}
IST = ZoneInfo("Asia/Kolkata")
for _k in ("GNANI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_MAPS_API_KEY"):  # drop stray spaces / line breaks
    if os.getenv(_k):
        os.environ[_k] = "".join(os.environ[_k].split())
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "22"))  # seconds to wait for Gemini / Claude
STT_TIMEOUT = float(os.getenv("STT_TIMEOUT", "20"))  # seconds to wait for Prisma
TTS_TIMEOUT = float(os.getenv("TTS_TIMEOUT", "20"))  # seconds to wait for Timbre
MAP_TIMEOUT = float(os.getenv("MAP_TIMEOUT", "40"))  # seconds to wait for the whole place + route lookup
VOICE = os.getenv("TTS_VOICE", "Kaveri")  # see GnaniTTSClient.supported_voices(model="timbre-v2.5")
GOOGLE_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
DENOISE_ON = os.getenv("DENOISE", "1") != "0"


def _mins(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def is_night(hhmm: str, t: dict) -> bool:
    x, s, e = _mins(hhmm), _mins(t["night_start"]), _mins(t["night_end"])
    return x >= s or x < e  # window wraps past midnight


def compute_fare(city: str, distance_km: float, hhmm: str, waiting_min: float) -> dict:
    t = TARIFFS[city]
    extra_km = max(0.0, distance_km - t["base_km"])
    distance_part = t["base_fare"] + extra_km * t["per_km"]
    wait_part = max(0.0, waiting_min - t["wait_free_min"]) * t["wait_per_min"]
    subtotal = distance_part + wait_part
    night = is_night(hhmm, t)
    night_part = subtotal * t["night_pct"] if night else 0.0
    expected = math.ceil(subtotal + night_part)  # meters round up to the next rupee
    return dict(
        city=t["label"], distance_km=distance_km, time=hhmm, night=night,
        base_fare=t["base_fare"], base_km=t["base_km"], extra_km=round(extra_km, 2),
        distance_part=round(distance_part, 1), wait_part=round(wait_part, 1),
        night_part=round(night_part, 1), expected=expected,
        fair_max=math.ceil(expected * 1.10),  # allow ~10% for route / meter variance
    )


def judge(expected: int, fair_max: int, quoted):
    if quoted is None:
        return "no_quote"
    if quoted <= fair_max + 5:
        return "fair"
    if quoted <= expected * 1.40:
        return "high"
    return "overcharge"


VERDICT_UI = {
    "fair": "✅ Fair price",
    "high": "🟠 Higher than meter - bargain or ask for the meter",
    "overcharge": "🔴 Overcharge - insist on the meter",
    "no_quote": "ℹ️ No price quoted - this is what the meter should show",
}

# --------------------------------------------------------------------------- #
# 1b. PLACES -> DISTANCE  (NEW: say "Majestic to Mantri Square", we fetch the km)
# --------------------------------------------------------------------------- #
CITY_CENTER = {
    "bengaluru": (12.9716, 77.5946),
    "chennai": (13.0827, 80.2707),
    "delhi": (28.6139, 77.2090),
}
UA = {"User-Agent": "MeterSach/1.0 (auto fare checker demo)"}


class NoLocation(Exception):
    """Start point is 'my location' but the browser gave us no GPS."""


class PlaceNotFound(Exception):
    """A spoken place name could not be found on the map."""


def _haversine_km(a, b) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def nearest_city(gps, max_km: float = 80.0):
    """Which tariff city is the user's GPS in? None if they are far from all three."""
    if not gps:
        return None
    best = min(CITY_CENTER, key=lambda c: _haversine_km(gps, CITY_CENTER[c]))
    return best if _haversine_km(gps, CITY_CENTER[best]) <= max_km else None


def _geocode_google(place: str, city: str, near=None):
    lat0, lon0 = near if near else CITY_CENTER[city]
    d = 0.15 if near else 0.4  # ~16 km box around the user, else the whole city
    r = requests.get(
        "https://maps.googleapis.com/maps/api/geocode/json",
        params={"address": f"{place}, {TARIFFS[city]['label']}", "components": "country:IN",
                "bounds": f"{lat0 - d},{lon0 - d}|{lat0 + d},{lon0 + d}", "key": GOOGLE_KEY},
        timeout=8,
    )
    j = r.json()
    if j.get("status") != "OK" or not j.get("results"):
        return None
    res = j["results"][0]
    loc = res["geometry"]["location"]
    return loc["lat"], loc["lng"], ", ".join(res["formatted_address"].split(",")[:2])


def _geocode_osm(place: str, city: str, near=None):
    lat0, lon0 = CITY_CENTER[city]
    box = f"{lon0 - 0.5},{lat0 + 0.5},{lon0 + 0.5},{lat0 - 0.5}"  # left, top, right, bottom
    for q in (f"{place}, {TARIFFS[city]['label']}", place):
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": q, "format": "jsonv2", "limit": 10, "viewbox": box, "bounded": 1, "countrycodes": "in"},
            headers=UA, timeout=8,
        )
        data = r.json()
        if data:
            # landmarks first: a query like "Majestic" also matches roads / wards named after it - skip those if we can
            good = [d for d in data if (d.get("category") or d.get("class")) not in ("highway", "boundary", "place")
                    or d.get("type") in ("neighbourhood", "suburb", "quarter")]
            data = good or data
            if near:  # many places share a name (Mantri Mall, Forum...): take the one nearest to the user
                data.sort(key=lambda d: _haversine_km(near, (float(d["lat"]), float(d["lon"]))))
            else:
                data.sort(key=lambda d: -float(d.get("importance") or 0))
            d = data[0]
            return float(d["lat"]), float(d["lon"]), ", ".join(d["display_name"].split(",")[:2])
        time.sleep(1.0)  # Nominatim allows ~1 request per second
    return None


def geocode(place: str, city: str, near=None):
    if GOOGLE_KEY:
        try:
            g = _geocode_google(place, city, near)
            if g:
                return g
        except Exception as e:  # noqa: BLE001
            print("GOOGLE geocode failed, using OSM:", e)
    return _geocode_osm(place, city, near)


def _route_km(a, b):
    """Road distance in km between two (lat, lon) points. Returns (km, source)."""
    if GOOGLE_KEY:
        try:
            r = requests.get(
                "https://maps.googleapis.com/maps/api/distancematrix/json",
                params={"origins": f"{a[0]},{a[1]}", "destinations": f"{b[0]},{b[1]}",
                        "mode": "driving", "key": GOOGLE_KEY},
                timeout=8,
            )
            el = r.json()["rows"][0]["elements"][0]
            if el.get("status") == "OK":
                return el["distance"]["value"] / 1000.0, "Google Maps road distance"
        except Exception as e:  # noqa: BLE001
            print("GOOGLE route failed, using OSRM:", e)
    try:
        r = requests.get(
            f"https://router.project-osrm.org/route/v1/driving/{a[1]},{a[0]};{b[1]},{b[0]}",
            params={"overview": "false"}, headers=UA, timeout=10,
        )
        return r.json()["routes"][0]["distance"] / 1000.0, "OpenStreetMap road distance"
    except Exception as e:  # noqa: BLE001
        print("OSRM failed, using straight line x1.3:", e)
    return _haversine_km(a, b) * 1.3, "straight-line distance x1.3 (road routing unavailable)"


def fetch_distance(origin, destination, cities, gps):
    """origin: place name, 'CURRENT' or None (None / CURRENT -> use GPS).
    cities: candidate cities in priority order; the first city where BOTH places are found wins.
    Returns a dict with km + labels + the city that matched."""
    if not destination:
        raise PlaceNotFound("no destination")

    def is_current(p):
        return str(p).strip().upper() in ("CURRENT", "MY LOCATION", "HERE")

    def resolve(p, city):
        if is_current(p):
            if not gps:
                raise NoLocation()
            return (gps[0], gps[1], "Your location")
        g = geocode(p, city, gps)  # prefer matches near the user
        if not g:
            raise PlaceNotFound(p)
        return g

    last = None
    for city in cities:
        try:
            # no start place said (and destination is a real place) -> start from the user's GPS
            a = resolve(origin if origin else "CURRENT", city)
            b = resolve(destination, city)
        except PlaceNotFound as e:  # not in this city: try the next one
            last = e
            continue
        km, how = _route_km(a, b)
        return dict(km=round(km, 1), origin=a[2], destination=b[2], method=how, city=city)
    raise last or PlaceNotFound(destination)


# --------------------------------------------------------------------------- #
# 2. LLM helpers (Gemini or Claude) - used for extraction + wording only
# --------------------------------------------------------------------------- #

def _provider() -> str:
    p = os.getenv("LLM_PROVIDER", "").lower()
    if p in ("gemini", "claude"):
        return p
    return "gemini" if os.getenv("GEMINI_API_KEY") else "claude"


@lru_cache(maxsize=1)
def gemini_models() -> tuple:
    """Fast models to try, in order: GEMINI_MODEL first, then whatever Flash / Flash-Lite models YOUR key can use."""
    from google import genai

    names = []
    if os.getenv("GEMINI_MODEL"):
        names.append(os.environ["GEMINI_MODEL"])
    try:
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        found = []
        for m in client.models.list():
            n = (m.name or "").replace("models/", "")
            acts = m.supported_actions or []
            if ("flash" in n and (not acts or "generateContent" in acts)
                    and not any(b in n for b in ("image", "tts", "live", "audio", "embed", "native", "robotics", "computer"))):
                found.append(n)
        # Flash-Lite first (fastest), then newest version first
        found.sort(key=lambda n: ("lite" not in n, [-int(p) for p in re.findall(r"\d+", n)]))
        names += [n for n in found if "omni" not in n and "gemini-2.5" not in n and n not in ("gemini-3.7-flash", "gemini-3.8-flash")]
    except Exception as e:  # noqa: BLE001
        print("MODEL LIST failed:", e)
    names += ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite"]  # fast on tested keys
    out = []
    for n in names:
        if n not in out:
            out.append(n)
    print("GEMINI models to try:", out[:3])
    return tuple(out[:3])


def _gemini_call(model: str, system: str, user: str, as_json: bool) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                          http_options=types.HttpOptions(timeout=int(LLM_TIMEOUT * 1000)))  # ms
    # Gemini 3 uses thinking_level, Gemini 2.5 uses thinking_budget. Minimal thinking = fast.
    think = (types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL) if "gemini-3" in model
             else types.ThinkingConfig(thinking_budget=0))

    def go(with_think: bool):
        kw = dict(system_instruction=system, temperature=0,
                  response_mime_type="application/json" if as_json else None)
        if with_think:
            kw["thinking_config"] = think
        return client.models.generate_content(
            model=model, contents=user, config=types.GenerateContentConfig(**kw)).text or ""

    def once():
        try:
            return go(True)
        except Exception as e:  # noqa: BLE001
            if "thinking" in str(e).lower():  # model rejects the thinking setting: retry without it
                return go(False)
            raise

    for attempt in range(3):  # Google's "503 high demand" spikes usually clear in a second or two
        try:
            return once()
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if attempt < 2 and ("503" in msg or "UNAVAILABLE" in msg or "overloaded" in msg.lower()):
                time.sleep(0.7 * (attempt + 1))
                continue
            raise


def _claude_call(system: str, user: str) -> str:
    import anthropic

    client = anthropic.Anthropic(timeout=LLM_TIMEOUT)
    r = client.messages.create(
        model=os.getenv("CLAUDE_MODEL", "claude-haiku-5-5"),
        max_tokens=600,
        temperature=0,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return r.content[0].text


def llm(system: str, user: str, as_json: bool) -> str:
    if _provider() == "claude":
        return _claude_call(system, user)

    from concurrent.futures import as_completed

    models = gemini_models()[:4]
    ex = ThreadPoolExecutor(max_workers=len(models))
    t0 = time.perf_counter()

    def one(m):
        out = _gemini_call(m, system, user, as_json)
        if as_json:
            parse_json(out)  # a model that answers with broken JSON counts as failed
        return m, out

    futs = [ex.submit(one, m) for m in models]  # ask several models at once, first good answer wins
    last = None
    try:
        for f in as_completed(futs, timeout=LLM_TIMEOUT + 3):
            try:
                m, out = f.result()
                print(f"GEMINI winner {m}: {time.perf_counter() - t0:.1f}s")
                return out
            except Exception as e:  # noqa: BLE001
                last = e
                print(f"GEMINI a model failed after {time.perf_counter() - t0:.1f}s: {str(e)[:140]}")
    except FutureTimeout:
        last = RuntimeError(f"Gemini did not answer within {LLM_TIMEOUT + 3:.0f} s")
    finally:
        ex.shutdown(wait=False, cancel_futures=True)

    if os.getenv("ANTHROPIC_API_KEY"):  # automatic backup if you have a Claude key
        print("GEMINI failed everywhere -> falling back to Claude")
        return _claude_call(system, user)
    raise last or RuntimeError("no Gemini model available")


def parse_json(raw: str) -> dict:
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", raw, re.S)
    return json.loads(m.group(0) if m else raw)


EXTRACT_SYSTEM = """You turn a messy voice transcript about an auto-rickshaw ride into JSON.
The speaker mixes English, Hindi, Kannada and Tamil, in any script or in Latin letters.
Spoken numbers may be words ("chaar sau pachaas" = 450, "saade teen" = 3.5, "dhai" = 2.5).

Return ONLY a JSON object with exactly these keys:
{"distance_km": number|null, "origin": string|null, "destination": string|null,
 "start_time_24h": "HH:MM"|null, "waiting_minutes": number,
 "quoted_fare": number|null, "city": "bengaluru"|"chennai"|"delhi"|null,
 "vehicle": "auto"|"cab"|"other", "reply_language": "hi-IN"|"kn-IN"|"ta-IN"|"en-IN"}

Rules:
- distance_km: ONLY if the speaker clearly says a distance. Convert miles/metres to km. null if they only name places. Never calculate fares yourself.
- origin / destination: the start and end PLACE names (landmark, area, mall, station, airport ...), written in English / Latin letters
  exactly as a map would know them (e.g. "Majestic", "Mantri Square", "T Nagar"). Drop words like se, to, ge, ku, from.
  Translate or transliterate Kannada / Hindi / Tamil script names to English. Do not add city names.
  For famous short names use the full official map name: "Majestic" -> "Kempegowda Bus Station", "KR Market" -> "Krishna Rajendra Market",
  "Central" -> "Chennai Central railway station", "ISBT" -> "Kashmere Gate ISBT", "Airport" -> the city's airport name.
- "CURRENT" means where the speaker is right now ("yahan se", "yahan tak", "here", "my location", "ikkinda", "ikkige", "inge irundhu", "inge varai").
  Use "CURRENT" as origin if the ride STARTS at the speaker's location ("from my location to Indiranagar", "yahan se Indiranagar").
  Use "CURRENT" as destination if the ride ENDS there ("from Mantri Mall to my location", "Mantri Mall se mere location tak").
  If TWO places are named ("A to B", "A se B", "A ilinda B ge", "A la irundhu B"), origin = A and destination = B. NEVER use "CURRENT" when a start place is named.
  origin = null if no start place is mentioned. destination = null if no end place is mentioned.
- start_time_24h: "raat 11 baje" = "23:00", "subah 5" = "05:00". null if no time is mentioned (never guess).
- waiting_minutes = 0 if not mentioned.
- quoted_fare = rupees the driver asked or said, null if none.
- city only if a city or a clearly city-specific place is mentioned, else null.
- reply_language = the language the speaker mainly uses (Hinglish -> hi-IN, Kanglish -> kn-IN, Tanglish -> ta-IN)."""

def _num(x):
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def _place(x):
    s = x.strip() if isinstance(x, str) else ""
    return s if s and s.lower() not in ("null", "none", "unknown") else None


def extract(transcript: str, detected: str | None) -> dict:
    now = datetime.now(IST).strftime("%A %H:%M")
    user = f"Current time in India: {now}\nSTT detected language: {detected}\nTranscript: {transcript}"
    try:
        d = parse_json(llm(EXTRACT_SYSTEM, user, as_json=True))
    except Exception as e:  # noqa: BLE001
        raise gr.Error(f"Could not understand the request ({e}). Please try again.")
    d["distance_km"] = _num(d.get("distance_km"))
    d["quoted_fare"] = _num(d.get("quoted_fare"))
    d["waiting_minutes"] = _num(d.get("waiting_minutes")) or 0.0
    d["origin"] = _place(d.get("origin"))
    d["destination"] = _place(d.get("destination"))
    # STT's language detection is the most reliable signal for the reply language
    if detected in LANGS:
        d["reply_language"] = detected
    if d.get("reply_language") not in LANGS:
        d["reply_language"] = "en-IN"
    return d


# Reply sentences. Every number comes from the Python fare math; no 2nd LLM call = much faster.
REPLY = {
    "en-IN": {
        "no_origin": "Where does the ride start? Please say the starting place, like Majestic to Indiranagar.",
        "no_quote": "For {km} kilometres in {city}, the meter fare should be about {exp} rupees.",
        "fair": "The meter fare is about {exp} rupees. {q} rupees is fair.",
        "high": "The meter fare is about {exp} rupees. {q} rupees is a bit high. Ask him to use the meter.",
        "overcharge": "The meter fare is only about {exp} rupees. {q} rupees is too much, he is asking {diff} rupees extra. Insist on the meter.",
        "cab": "Cab prices are set by the app, not a meter. Check the fare in the app before booking.",
        "no_distance": "I did not catch the distance. Please say roughly how many kilometres the ride is, or say the start and destination names.",
        "no_location": "I could not get your location. Tap the Use my location button, or say the starting place.",
        "no_place": "I could not find those places on the map. Please say the start and destination names again, or tell me the distance in kilometres.",
    },
    "hi-IN": {
        "no_origin": "सफ़र कहाँ से शुरू होगा? शुरुआती जगह का नाम बोलिए, जैसे Majestic से Indiranagar।",
        "no_quote": "{city} में {km} किलोमीटर के लिए मीटर का किराया लगभग {exp} रुपये बनता है।",
        "fair": "मीटर के हिसाब से किराया लगभग {exp} रुपये है। {q} रुपये ठीक है।",
        "high": "मीटर के हिसाब से लगभग {exp} रुपये बनता है। {q} रुपये थोड़ा ज़्यादा है। मीटर से चलने को कहिए।",
        "overcharge": "मीटर के हिसाब से सिर्फ़ लगभग {exp} रुपये बनता है। {q} रुपये बहुत ज़्यादा है, वह {diff} रुपये extra माँग रहा है। मीटर चालू करवाइए।",
        "cab": "कैब का किराया ऐप तय करता है, मीटर नहीं। बुक करने से पहले ऐप में किराया देख लीजिए।",
        "no_distance": "दूरी समझ नहीं आई। कृपया सफ़र के किलोमीटर बताइए, या शुरुआत और मंज़िल का नाम बोलिए।",
        "no_location": "आपकी लोकेशन नहीं मिली। 'Use my location' बटन दबाइए, या शुरुआती जगह का नाम बोलिए।",
        "no_place": "नक्शे पर ये जगहें नहीं मिलीं। कृपया शुरुआत और मंज़िल का नाम फिर से बोलिए, या किलोमीटर बताइए।",
    },
    "kn-IN": {
        "no_origin": "ಪ್ರಯಾಣ ಎಲ್ಲಿಂದ ಶುರು? ಪ್ರಾರಂಭದ ಸ್ಥಳದ ಹೆಸರು ಹೇಳಿ, ಉದಾಹರಣೆಗೆ Majestic ಇಂದ Indiranagar.",
        "no_quote": "{city}ನಲ್ಲಿ {km} ಕಿಲೋಮೀಟರ್‌ಗೆ ಮೀಟರ್ ಪ್ರಕಾರ ಸುಮಾರು {exp} ರೂಪಾಯಿ ಆಗುತ್ತದೆ.",
        "fair": "ಮೀಟರ್ ಪ್ರಕಾರ ಸುಮಾರು {exp} ರೂಪಾಯಿ. {q} ರೂಪಾಯಿ ಸರಿಯಾಗಿದೆ.",
        "high": "ಮೀಟರ್ ಪ್ರಕಾರ ಸುಮಾರು {exp} ರೂಪಾಯಿ. {q} ರೂಪಾಯಿ ಸ್ವಲ್ಪ ಜಾಸ್ತಿ. ಮೀಟರ್ ಹಾಕಲು ಹೇಳಿ.",
        "overcharge": "ಮೀಟರ್ ಪ್ರಕಾರ ಸುಮಾರು {exp} ರೂಪಾಯಿ ಮಾತ್ರ. {q} ರೂಪಾಯಿ ತುಂಬಾ ಜಾಸ್ತಿ, {diff} ರೂಪಾಯಿ ಹೆಚ್ಚು ಕೇಳುತ್ತಿದ್ದಾರೆ. ಮೀಟರ್ ಹಾಕಲು ಹೇಳಿ.",
        "cab": "ಕ್ಯಾಬ್ ದರವನ್ನು ಆ್ಯಪ್ ನಿರ್ಧರಿಸುತ್ತದೆ, ಮೀಟರ್ ಅಲ್ಲ. ಬುಕ್ ಮಾಡುವ ಮೊದಲು ಆ್ಯಪ್‌ನಲ್ಲಿ ದರ ನೋಡಿ.",
        "no_distance": "ದೂರ ಅರ್ಥವಾಗಲಿಲ್ಲ. ಪ್ರಯಾಣ ಸುಮಾರು ಎಷ್ಟು ಕಿಲೋಮೀಟರ್ ಎಂದು ಹೇಳಿ, ಅಥವಾ ಪ್ರಾರಂಭ ಮತ್ತು ಗಮ್ಯಸ್ಥಾನದ ಹೆಸರು ಹೇಳಿ.",
        "no_location": "ನಿಮ್ಮ ಸ್ಥಳ ಸಿಗಲಿಲ್ಲ. 'Use my location' ಬಟನ್ ಒತ್ತಿ, ಅಥವಾ ಪ್ರಾರಂಭದ ಸ್ಥಳದ ಹೆಸರು ಹೇಳಿ.",
        "no_place": "ನಕ್ಷೆಯಲ್ಲಿ ಈ ಸ್ಥಳಗಳು ಸಿಗಲಿಲ್ಲ. ಪ್ರಾರಂಭ ಮತ್ತು ಗಮ್ಯಸ್ಥಾನದ ಹೆಸರನ್ನು ಮತ್ತೆ ಹೇಳಿ, ಅಥವಾ ಕಿಲೋಮೀಟರ್ ಹೇಳಿ.",
    },
    "ta-IN": {
        "no_origin": "பயணம் எங்கிருந்து தொடங்குகிறது? புறப்படும் இடத்தின் பெயரைச் சொல்லுங்கள், உதாரணம் Majestic இலிருந்து Indiranagar.",
        "no_quote": "{city}யில் {km} கிலோமீட்டருக்கு மீட்டர் படி சுமார் {exp} ரூபாய் ஆகும்.",
        "fair": "மீட்டர் படி சுமார் {exp} ரூபாய். {q} ரூபாய் சரியானது.",
        "high": "மீட்டர் படி சுமார் {exp} ரூபாய். {q} ரூபாய் கொஞ்சம் அதிகம். மீட்டர் போடச் சொல்லுங்கள்.",
        "overcharge": "மீட்டர் படி சுமார் {exp} ரூபாய் மட்டுமே. {q} ரூபாய் மிக அதிகம், {diff} ரூபாய் கூடுதலாகக் கேட்கிறார். மீட்டர் போடச் சொல்லுங்கள்.",
        "cab": "கேப் கட்டணத்தை ஆப் தீர்மானிக்கிறது, மீட்டர் இல்லை. புக் செய்வதற்கு முன் ஆப்பில் கட்டணத்தைப் பாருங்கள்.",
        "no_distance": "தூரம் புரியவில்லை. பயணம் சுமார் எத்தனை கிலோமீட்டர் என்று சொல்லுங்கள், அல்லது புறப்படும் இடம், சேரும் இடம் பெயர்களைச் சொல்லுங்கள்.",
        "no_location": "உங்கள் இருப்பிடம் கிடைக்கவில்லை. 'Use my location' பொத்தானை அழுத்துங்கள், அல்லது புறப்படும் இடத்தின் பெயரைச் சொல்லுங்கள்.",
        "no_place": "வரைபடத்தில் அந்த இடங்கள் கிடைக்கவில்லை. புறப்படும் இடம், சேரும் இடம் பெயர்களை மீண்டும் சொல்லுங்கள், அல்லது கிலோமீட்டர் சொல்லுங்கள்.",
    },
}


def reply_text(kind: str, lang: str, **v) -> str:
    return REPLY.get(lang, REPLY["en-IN"])[kind].format(**v)


def _km(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:.1f}"



# --------------------------------------------------------------------------- #
# 3. Gnani clients
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def stt_client() -> GnaniSTTClient:
    return GnaniSTTClient()  # reads GNANI_API_KEY


@lru_cache(maxsize=1)
def tts_client() -> GnaniTTSClient:
    return GnaniTTSClient()  # reads GNANI_API_KEY


_pool = ThreadPoolExecutor(max_workers=4)


def with_timeout(fn, seconds: int, what: str, *args, **kwargs):
    """Run a blocking SDK call; give up with a clear error instead of hanging the UI."""
    fut = _pool.submit(fn, *args, **kwargs)
    try:
        return fut.result(timeout=seconds)
    except FutureTimeout:
        raise gr.Error(f"{what} timed out after {seconds:.0f}s. Check your internet connection and API key, then try again.")


MIC_STATS: dict = {}


def bandpass(x: np.ndarray, lo: float = 200.0, hi: float = 4500.0, sr: int = 16000) -> np.ndarray:
    """Keep the band where the human voice lives; engine rumble below and hiss above are cut."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    m = np.clip((f - lo * 0.7) / (lo * 0.3), 0, 1) * np.clip((hi * 1.2 - f) / (hi * 0.2), 0, 1)
    return np.fft.irfft(X * m, n=len(x)).astype(np.float32)


def denoise(x: np.ndarray, sr: int = 16000, strength: float = 1.0) -> np.ndarray:
    """Spectral noise gating for traffic / engine / wind / crowd hiss.
    Learns the noise spectrum from the quietest ~15% of the clip (the gaps between words),
    then subtracts it from every frame with a smoothed mask. Pure numpy, no extra install.
    (It cannot separate a second person talking nearby - hold the mic close to your mouth for that.)"""
    n_fft, hop = 512, 128
    if len(x) < n_fft * 8:
        return x
    win = np.hanning(n_fft).astype(np.float32)
    pad = np.pad(x, (n_fft // 2, n_fft // 2))
    nfr = 1 + (len(pad) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(nfr)[:, None]
    S = np.fft.rfft(pad[idx] * win, axis=1)
    mag2 = np.abs(S) ** 2

    energy = mag2.sum(axis=1)
    quiet = np.argsort(energy)[: max(6, int(0.15 * nfr))]
    noise_q = mag2[quiet].mean(axis=0)  # noise seen in the quietest frames (gaps between words)
    noise_p = np.percentile(mag2, 20, axis=0) * 4.0  # low percentile per frequency: still works when there are NO clean gaps
    noise2 = 0.5 * (noise_q + noise_p) + 1e-12  # noise power per frequency bin

    # power spectral subtraction with over-subtraction, never fully silent (floor avoids "underwater" artefacts)
    alpha, floor = 1.8 * strength, max(0.12 / strength, 0.05)
    gain = np.sqrt(np.clip(1.0 - alpha * noise2[None, :] / (mag2 + 1e-12), floor ** 2, 1.0))
    # smooth the mask in time and frequency so it does not "twinkle"
    gain = (gain + np.roll(gain, 1, axis=0) + np.roll(gain, -1, axis=0)) / 3.0
    gain = (gain + np.roll(gain, 1, axis=1) + np.roll(gain, -1, axis=1)) / 3.0

    frames = np.fft.irfft(S * gain, n=n_fft, axis=1) * win
    out = np.zeros(len(pad), dtype=np.float32)
    wsum = np.zeros(len(pad), dtype=np.float32)
    w2 = win ** 2
    for i in range(nfr):
        a = i * hop
        out[a:a + n_fft] += frames[i]
        wsum[a:a + n_fft] += w2
    out = out / np.maximum(wsum, 1e-6)
    out = out[n_fft // 2: n_fft // 2 + len(x)].astype(np.float32)

    snr_db = 10 * np.log10((np.percentile(energy, 95) + 1e-12) / (np.median(energy[quiet]) + 1e-12))
    print(f"AUDIO    : denoise applied, rough speech-to-noise ratio {snr_db:.1f} dB")
    MIC_STATS["snr_db"] = round(float(snr_db), 1)
    return out


def adaptive_gain(x: np.ndarray) -> np.ndarray:
    """Level every part of the clip separately, so speech that gets quieter halfway
    (browser auto-gain dropping, speaker turning away, ...) is still loud enough for Prisma."""
    fl = 320  # 20 ms frames at 16 kHz
    n = len(x) // fl
    if n < 8:
        return x
    rms = np.sqrt((x[: n * fl].reshape(n, fl) ** 2).mean(axis=1)) + 1e-9
    if float(np.percentile(rms, 90)) / float(np.percentile(rms, 10)) < 2.0:
        MIC_STATS["flat"] = True
        return x  # flat loudness = no clear speech (just noise): don't blow it up
    # Level by the SPEECH frames only. Pauses just inherit the gain of the nearest speech, so the noise in
    # the pauses is never lifted toward speech loudness (the old version undid the noise removal).
    thr = max(float(np.percentile(rms, 10)) * 2.5, float(np.percentile(rms, 90)) * 0.1, 1e-4)
    if int((rms >= thr).sum()) < 3:
        return x
    k = 20  # follow the loudest speech frame within ~0.4 s, so words are not chopped
    pad = np.pad(np.where(rms >= thr, rms, 0.0), (k // 2, k // 2), mode="constant")
    loc = np.array([pad[i : i + k].max() for i in range(n)])
    have = np.nonzero(loc > 0)[0]
    env = np.maximum(np.interp(np.arange(n), have, loc[have]), thr)
    gain = np.clip(0.1 / env, 1.0, 60.0)
    gain = np.convolve(np.pad(gain, (5, 5), mode="edge"), np.ones(11) / 11, mode="valid")
    per_sample = np.interp(np.arange(len(x)), (np.arange(n) + 0.5) * fl, gain)
    return x * per_sample.astype(np.float32)


def speech_gate(x: np.ndarray):
    """Close-talk gate: the person holding the mic is the loudest voice. Frames far quieter than the
    loudest speech (people talking in the distance, traffic between words) are muted, and leading /
    trailing noise is cut off. Returns (per-sample gain, start_sample, end_sample)."""
    fl = 320  # 20 ms
    n = len(x) // fl
    ones = np.ones(len(x), dtype=np.float32)
    if n < 10:
        return ones, 0, len(x)
    rms = np.sqrt((x[: n * fl].reshape(n, fl) ** 2).mean(axis=1)) + 1e-9
    loud = float(np.percentile(rms, 95))
    noise = float(np.percentile(rms, 10))
    thr = max(loud * 0.12, noise * 2.5)  # about 18 dB below the loud speech
    active = (rms >= thr).astype(np.float32)
    active = (np.convolve(active, np.ones(13), mode="same") > 0).astype(np.float32)  # keep ~250 ms around speech so word edges survive
    idx = np.nonzero(active)[0]
    if len(idx) == 0:
        return ones, 0, len(x)
    g = np.where(active > 0, 1.0, 0.03)
    g = np.convolve(np.pad(g, (2, 2), mode="edge"), np.ones(5) / 5, mode="valid")
    per = np.interp(np.arange(len(x)), (np.arange(n) + 0.5) * fl, g).astype(np.float32)
    pad = 12  # frames, ~0.25 s
    s = max(0, (int(idx[0]) - pad) * fl)
    e = min(len(x), (int(idx[-1]) + 1 + pad) * fl)
    print(f"AUDIO    : speech gate kept {(e - s) / 16000:.1f}s of {len(x) / 16000:.1f}s")
    return per, s, e


def save_clean_wav(audio, mode: str = "clean") -> str:
    """mode: 'clean' = normal denoise, 'strong' = heavy denoise + close-talk gate, 'raw' = no denoise."""
    """Gradio gives (sample_rate, samples). Make it what Prisma likes best: 16 kHz, mono, 16-bit,
    rumble removed, background noise reduced, volume normalised. (no ffmpeg needed)."""
    sr, data = audio
    x = np.asarray(data)
    print(f"AUDIO in : sr={sr} dtype={x.dtype} shape={x.shape}")
    if x.dtype.kind in "iu":
        x = x.astype(np.float32) / float(np.iinfo(x.dtype).max)
    else:
        x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x - float(x.mean() if x.size else 0.0)
    if x.size < int(sr * 0.3):
        raise gr.Error("Recording is too short. Hold the mic for the whole sentence.")
    # resample to 16 kHz (box low-pass first so there is no aliasing)
    if int(sr) != 16000:
        k = max(1, int(round(sr / 16000)))
        if k > 1:
            x = np.convolve(x, np.ones(k, dtype=np.float32) / k, mode="same")
        n_out = int(round(len(x) * 16000 / sr))
        x = np.interp(np.linspace(0, len(x) - 1, n_out), np.arange(len(x)), x).astype(np.float32)
    # high-pass ~100 Hz: removes fan / table / breath rumble that confuses speech models
    x = x - np.convolve(x, np.ones(160, dtype=np.float32) / 160, mode="same")
    level = float(np.percentile(np.abs(x), 99.5))
    print(f"AUDIO    : level before boost = {level:.4f}")
    if level < 1e-4:
        raise gr.Error("The recording is silent. Pick the right microphone in Chrome (lock icon in the address bar) and in Windows Sound settings.")
    secs = [round(float(np.sqrt((x[i * 16000:(i + 1) * 16000] ** 2).mean())), 4) for i in range(int(len(x) // 16000) + 1) if len(x[i * 16000:(i + 1) * 16000])]
    print("AUDIO    : loudness per second (RMS) before leveling:", secs)
    MIC_STATS.clear()
    MIC_STATS.update(mic_peak=round(level, 4), loudness_per_second=secs, flat=False)
    MIC_STATS["mode"] = mode
    mask = None
    if mode == "nr" and _nr is not None:  # extra denoiser (noisereduce package), handles changing noise
        try:
            x = _nr.reduce_noise(y=x, sr=16000, stationary=False, prop_decrease=0.9).astype(np.float32)
        except Exception as e:  # noqa: BLE001
            print("noisereduce failed:", e)
    elif mode in ("clean", "strong") and DENOISE_ON:  # remove background noise before leveling
        try:
            x = denoise(x, strength=1.6 if mode == "strong" else 1.0)
            if mode == "strong":
                x = bandpass(x)  # voice lives at ~200-4500 Hz
                mask = speech_gate(x)  # computed BEFORE leveling, so far-away voices stay quieter than yours
        except Exception as e:  # noqa: BLE001  (never let noise removal break the pipeline)
            print("DENOISE failed, using original audio:", e)
    if mode != "raw":  # 'raw' = untouched apart from rumble filter + loudness, in case cleanup hurts
        x = adaptive_gain(x)
    if mask is not None:
        per, s, e = mask
        x = (x * per)[s:e]
    peak = float(np.percentile(np.abs(x), 99.5)) or 1.0
    x = np.clip(x * min(0.6 / peak, 100.0), -1.0, 1.0)
    pcm = (x * 32767).astype("<i2")
    path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm.tobytes())
    print(f"AUDIO out: 16000 Hz mono, {len(pcm) / 16000:.1f}s -> {path}")
    return path


def transcribe(path: str, quick: bool = False):
    """indic-auto (+hints) first; if Prisma hears nothing, retry quickly with other settings."""
    topic = "Auto rickshaw ride: start and destination place names, distance in km, time of day, waiting time, fare asked by driver in rupees"
    attempts = [
        dict(language_code="indic-auto", model="prisma-v3.0"),  # plain: proven to work in test_gnani.py
        dict(language_code="indic-auto", model="prisma-v3.0",
             language_candidates=["kn-IN", "hi-IN", "ta-IN", "en-IN"], topic=topic),
        dict(language_code="hi-IN", model="prisma-v3.0"),
        dict(language_code="en-IN", model="prisma-v3.0"),
    ]
    raw = {}
    for kw in (attempts[:1] if quick else attempts):
        try:
            raw = with_timeout(stt_client().transcribe, STT_TIMEOUT, "Prisma speech recognition", path, **kw)
        except gr.Error:
            raise
        except (TypeError, ValueError):  # option not accepted by this SDK version
            continue
        except Exception as e:  # noqa: BLE001  (auth, bad audio, network ...)
            raise gr.Error(f"Speech recognition failed: {e}")
        print("PRISMA RAW:", kw["language_code"], raw)
        text = (raw.get("transcript") or "").strip()
        if text:
            return text, raw.get("detected_language_code"), raw
    return "", None, raw


def speak(text: str, lang: str):
    audio = with_timeout(
        tts_client().synthesize, TTS_TIMEOUT, "Timbre voice reply",
        text,
        voice=VOICE,
        model="timbre-v2.5",
        language=lang,
        speed=0.95,
        audio_config=AudioConfig(container="wav", encoding="linear_pcm", sample_rate=24000),
    )
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    f.write(audio)
    f.close()
    return f.name


# --------------------------------------------------------------------------- #
# 4. Pipeline: Voice Input -> LLM Action -> Voice Output
# --------------------------------------------------------------------------- #

def breakdown_md(r: dict, quoted, verdict: str, route=None) -> str:
    rows = [
        f"| Base fare (first {r['base_km']} km) | ₹{r['base_fare']} |",
        f"| Extra {r['extra_km']} km | ₹{round(r['distance_part'] - r['base_fare'], 1)} |",
        f"| Waiting charge | ₹{r['wait_part']} |",
        f"| Night surcharge ({'applies' if r['night'] else 'no'}, {r['time']}) | ₹{r['night_part']} |",
        f"| **Expected meter fare** | **₹{r['expected']}** |",
    ]
    if quoted is not None:
        rows.append(f"| Driver's quote | ₹{int(quoted)} (+₹{int(quoted - r['expected'])}) |")
    route_line = ""
    if route:
        route_line = (f"📍 **{route['origin']} → {route['destination']}**: {route['km']} km "
                      f"<sub>({route['method']})</sub>\n\n")
    return (
        f"### {VERDICT_UI[verdict]}\n\n{route_line}| {r['city']} auto tariff | Amount |\n|---|---|\n"
        + "\n".join(rows)
        + "\n\n<sub>Tariff table = demo values. Confirm against the current official notification.</sub>"
    )


def _score(d: dict) -> int:
    """How many ride details did this reading of the audio give us?"""
    n = sum(1 for k in ("distance_km", "origin", "destination", "quoted_fare", "start_time_24h") if d.get(k))
    return n + (1 if d.get("vehicle") == "cab" else 0)


def hear_best(audio):
    """Noise-proof listening. Make several versions of the recording (normal cleanup, untouched, heavy cleanup,
    optional noisereduce), send ALL of them to Prisma at the same time, understand each transcript, and keep the
    one that gives the most ride details. Returns (wav_path, transcript, detected_lang, prisma_raw, parsed_json_or_None)."""
    modes = ["clean", "raw"]  # 2 requests in parallel; 'strong' is made only if these hear nothing
    paths, base = {}, None
    for m in modes:
        paths[m] = save_clean_wav(audio, m)
        if base is None:
            base = dict(MIC_STATS)  # mic level info of the first version
    MIC_STATS.clear()
    MIC_STATS.update(base)

    def stt(m):
        try:
            t, det, rw = transcribe(paths[m], quick=True)
            return m, t, det, rw
        except Exception as e:  # noqa: BLE001
            print(f"STT variant {m} failed:", e)
            return m, "", None, {"error": str(e)}

    with ThreadPoolExecutor(max_workers=len(modes)) as ex:
        heard = list(ex.map(stt, modes))
    for m, t, _, _ in heard:
        print(f"VARIANT {m:6s} heard: {t!r}")

    uniq = {}
    for item in heard:  # identical transcripts need only one LLM call
        if item[1]:
            uniq.setdefault(item[1].strip().lower(), item)

    def ext(item):
        try:
            return item, extract(item[1], item[2])
        except Exception as e:  # noqa: BLE001
            print(f"extract failed for variant {item[0]}:", e)
            return item, None

    if uniq:
        with ThreadPoolExecutor(max_workers=len(uniq)) as ex:
            done = list(ex.map(ext, list(uniq.values())))
    else:
        done = []
    ok = [(it, dd) for it, dd in done if dd is not None]
    order = {m: i for i, m in enumerate(modes)}
    if ok:
        (m, t, det, rw), dd = max(ok, key=lambda p: (_score(p[1]), -order[p[0][0]]))
        print(f"VARIANT chosen: {m} (score {_score(dd)})")
        dd["stt_variants"] = {mm: tt for mm, tt, _, _ in heard}
        return paths[m], t, det, rw, dd
    first = next(((m, t, det, rw) for m, t, det, rw in heard if t), None)
    if first:  # something was heard but the LLM step failed: let the normal path raise a clear error
        m, t, det, rw = first
        return paths[m], t, det, rw, None
    # Nothing heard at all. In noise Prisma's language auto-detection often gives up, so force each language
    # (on the heavy-cleanup and the normal version), all in parallel, and keep the longest transcript.
    def forced(job):
        m, lang = job
        try:
            rw = with_timeout(stt_client().transcribe, STT_TIMEOUT, "Prisma speech recognition",
                              paths[m], language_code=lang, model="prisma-v3.0")
            return m, lang, (rw.get("transcript") or "").strip(), rw
        except Exception as e:  # noqa: BLE001
            print(f"forced {lang} on {m} failed:", e)
            return m, lang, "", {"error": str(e)}

    keep = dict(MIC_STATS)
    paths["strong"] = save_clean_wav(audio, "strong")
    MIC_STATS.clear()
    MIC_STATS.update(keep)
    jobs = [("strong", "indic-auto")] + [(m, lang) for m in ("clean",) for lang in ("hi-IN", "en-IN", "kn-IN")]
    with ThreadPoolExecutor(max_workers=2) as ex:
        res = list(ex.map(forced, jobs))
    MIC_STATS["stt_raw"] = {f"{m}/{lang}": rw for m, lang, _, rw in res}
    MIC_STATS["stt_raw"].update({f"{m}/first": rw for m, _, _, rw in heard})
    for m, lang, t, _ in res:
        print(f"FORCED {lang} on {m}: {t!r}")
    m, lang, t, rw = max(res, key=lambda r: len(r[2]))
    if t:
        return paths[m], t, lang, rw, None
    return paths["clean"], "", None, rw, None


def _usable(d: dict) -> bool:
    """Did the transcript give us anything to work with (a distance, a destination, a price, or a cab)?"""
    return bool(d.get("distance_km") or d.get("destination") or d.get("quoted_fare") or d.get("vehicle") == "cab")


def _gps(lat, lon):
    try:
        if lat is None or lon is None or lat == "" or lon == "":
            return None
        la, lo = float(lat), float(lon)
        return (la, lo) if -90 <= la <= 90 and -180 <= lo <= 180 else None
    except (TypeError, ValueError):
        return None


def _run(audio, typed_text, default_city, lat=None, lon=None, from_place=None, to_place=None, use_gps=True):
    t0 = time.perf_counter()
    gps = _gps(lat, lon) if use_gps else None
    # ---- 1. Voice input ----
    detected, transcript, raw, heard = None, "", None, None
    from_audio = False
    pre_d = None
    typed = (typed_text or "").strip()
    if audio is not None:
        heard, transcript, detected, raw, pre_d = hear_best(audio)  # best of several cleaned versions (shown in "Under the hood")
        from_audio = bool(transcript)
        if not transcript and typed:
            gr.Warning("Prisma returned no text for the recording, so the typed text was used.")
    if not transcript and typed:
        transcript, detected = typed, None
    from_place, to_place = (from_place or "").strip(), (to_place or "").strip()
    if not transcript and (from_place or to_place):  # only the From / To boxes were filled
        transcript, detected = f"{from_place or 'my location'} to {to_place or 'my location'}", None
    if not transcript:
        if audio is not None and isinstance(raw, dict) and raw.get("error"):  # a real failure, not silence
            raise gr.Error(f"Prisma error: {raw['error']}  (Set GNANI_API_KEY in THIS terminal, then restart app.py)")
        if audio is not None:
            gr.Warning("Prisma heard no words. Open 'Under the hood' and play 'What Prisma received'.")
            return (None, "I could not hear any words. Please speak a little louder and closer to the mic.",
                    f"### ❓ Prisma heard no words\n\n**Mic level captured: {MIC_STATS.get('mic_peak')}** "
                    "(normal speech is about 0.05 to 0.5; below 0.01 means the mic is not picking you up). "
                    "Play **What Prisma received** under *Under the hood* to hear what was sent. "
                    "In a very noisy place: hold the mic 5-10 cm from your mouth and speak a little louder, or restart the app "
                    "with `MIC_RAW=1` (switches off the browser's own noise filter, which can swallow quiet speech).",
                    "", {"mic_level": dict(MIC_STATS), "prisma_raw_response": raw}, heard)
        raise gr.Error("Record or type something first, e.g. 'Majestic se Indiranagar 11 km, raat 11 baje, 450 maang raha hai'.")

    # ---- 2. LLM action: messy speech -> JSON, then exact Python math ----
    print(f"STAGE 1 OK ({time.perf_counter() - t0:.1f}s) transcript:", transcript)
    d = pre_d or extract(transcript, detected)
    if from_place or to_place:  # places typed in the boxes win over what was heard; look the distance up from them
        if from_place:
            d["origin"] = from_place
        if to_place:
            d["destination"] = to_place
        d["distance_km"] = None
    print(f"STAGE 2 OK ({time.perf_counter() - t0:.1f}s) LLM JSON:", d)
    lang = d["reply_language"]

    if d.get("vehicle") == "cab":
        msg = reply_text("cab", lang)
        return _out(msg, lang, "ℹ️ This tool checks auto-rickshaw meter fares only.", transcript, d, heard)

    # city: what the user said > where their GPS is > dropdown
    # candidate cities, best guess first: said by user > where GPS is > dropdown > the rest
    cands = []
    for c in (d.get("city"), nearest_city(gps), default_city, *TARIFFS):
        if c in TARIFFS and c not in cands:
            cands.append(c)
    city = cands[0]

    # ---- 2b. NEW: no km spoken -> look it up from the place names ----
    route = None
    if not d["distance_km"] or d["distance_km"] <= 0:
        if d.get("destination") and not use_gps and not d.get("origin"):
            msg = reply_text("no_origin", lang)
            return _out(msg, lang, "❓ No start place said - say it (e.g. *Majestic to Indiranagar*) or tick **Start from my location**.", transcript, d, heard)
        if d.get("destination"):
            try:
                route = with_timeout(fetch_distance, MAP_TIMEOUT, "Map lookup",
                                     d.get("origin"), d["destination"], cands, gps)
                d["distance_km"] = route["km"]
                city = route["city"]  # the city where both places were found decides the tariff
                d["route"] = route
                print(f"STAGE 2b OK ({time.perf_counter() - t0:.1f}s) route:", route)
            except NoLocation:
                msg = reply_text("no_location", lang)
                return _out(msg, lang, "📍 Location missing - press **Use my location** or say the start place.", transcript, d, heard)
            except PlaceNotFound as e:
                print("PLACE NOT FOUND:", e)
                msg = reply_text("no_place", lang)
                return _out(msg, lang, f"❓ Could not find **{e}** on the map - say the place name again or give the km.", transcript, d, heard)
            except Exception as e:  # noqa: BLE001  (network down, rate limit, ...)
                print("MAP LOOKUP FAILED:", e)
                gr.Warning(f"Map lookup failed ({e}).")
                msg = reply_text("no_place", lang)
                return _out(msg, lang, "⚠️ Map lookup failed - check your internet or give the km directly.", transcript, d, heard)

    if not d["distance_km"] or d["distance_km"] <= 0:
        msg = reply_text("no_distance", lang)
        return _out(msg, lang, "❓ Distance missing - ask the user again.", transcript, d, heard)

    t_raw = d.get("start_time_24h")
    hhmm = t_raw if isinstance(t_raw, str) and re.fullmatch(r"\d{1,2}:\d{2}", t_raw) else datetime.now(IST).strftime("%H:%M")

    r = compute_fare(city, d["distance_km"], hhmm, d["waiting_minutes"])
    verdict = judge(r["expected"], r["fair_max"], d["quoted_fare"])

    msg = reply_text(
        verdict if verdict in ("no_quote", "fair", "high", "overcharge") else "no_quote", lang,
        city=r["city"], km=_km(r["distance_km"]), exp=r["expected"],
        q=0 if d["quoted_fare"] is None else int(d["quoted_fare"]),
        diff=0 if d["quoted_fare"] is None else int(d["quoted_fare"] - r["expected"]),
    )
    return _out(msg, lang, breakdown_md(r, d["quoted_fare"], verdict, route), transcript, d, heard)


def run(audio, typed_text, default_city, lat=None, lon=None, from_place=None, to_place=None, use_gps=True):
    """Wrapper: any failure is shown permanently on the page with where it happened and how long it took."""
    t0 = time.perf_counter()
    try:
        return _run(audio, typed_text, default_city, lat, lon, from_place, to_place, use_gps)
    except Exception as e:  # noqa: BLE001  (gr.Error and anything unexpected)
        import traceback
        traceback.print_exc()
        secs = time.perf_counter() - t0
        why = getattr(e, "message", None) or str(e) or type(e).__name__
        print(f"FAILED after {secs:.1f}s: {why}")
        return (None, "Something went wrong. See the message on the right.",
                f"### ⚠️ Failed after {secs:.0f}s\n\n**{why}**\n\n"
                "Tell me this message (and the terminal lines starting with `STAGE` / `PRISMA RAW`).",
                "", {"error": why, "seconds": round(secs, 1), "mic_level": dict(MIC_STATS)}, None)


def _out(msg, lang, md, transcript, parsed, heard=None):
    # ---- 3. Voice output ----
    print("STAGE 3 - reply text ready, calling Timbre:", msg)
    try:
        audio = speak(msg, lang)
        print("STAGE 3 OK - voice reply ready")
    except Exception as e:  # noqa: BLE001
        gr.Warning(f"Voice output failed ({e}). Showing text only.")
        audio = None
    return audio, msg, md, transcript, parsed, heard


# --------------------------------------------------------------------------- #
# 5. UI
# --------------------------------------------------------------------------- #


# Chrome's auto-gain / noise-suppression often "ducks" the mic a couple of seconds into a recording.
# This switches them off for the recorder; adaptive_gain() above handles the level instead.
MIC_JS = """
<script>
(function () {
  const md = navigator.mediaDevices;
  if (!md || !md.getUserMedia) return;
  const orig = md.getUserMedia.bind(md);
  md.getUserMedia = function (c) {
    if (c && c.audio) {
      c = Object.assign({}, c);
      const a = c.audio === true ? {} : Object.assign({}, c.audio);
      a.autoGainControl = false; a.noiseSuppression = false; a.echoCancellation = false;
      c.audio = a;
    }
    return orig(c);
  };
})();
</script>
"""

# NEW: asks the browser for the user's GPS and writes it into the latitude / longitude boxes.
# Works on localhost or https (use SHARE=1 for a public https link on your phone).
LOCATION_JS = """
() => new Promise((resolve) => {
  if (!navigator.geolocation) { resolve([null, null]); return; }
  navigator.geolocation.getCurrentPosition(
    (p) => resolve([p.coords.latitude, p.coords.longitude]),
    (e) => { console.log('Location error', e); resolve([null, null]); },
    { enableHighAccuracy: true, timeout: 10000, maximumAge: 60000 }
  );
})
"""

EXAMPLES = [
    "Majestic se Indiranagar 11 km, raat 11 baje, driver chaar sau pachaas maang raha hai",
    "Majestic se Mantri Square kitna padega, driver 150 maang raha hai",
    "yahan se Indiranagar jaana hai, raat 11 baje, driver 300 bol raha hai",
    "Anna Nagar to T Nagar sumaar 8 kilometer, night 12 mani, driver 300 rupaye solraar",
    "Koramangala ge hogbeku, 5 km, bere auto illa, avanu 200 rupayi keltaane",
]

_use_js = os.getenv("MIC_RAW") == "1"  # off by default: it made some laptop mics quieter
_blocks_kw = {"head": MIC_JS} if _use_js and "head" in inspect.signature(gr.Blocks.__init__).parameters else {}
with gr.Blocks(title="Meter Sach", **_blocks_kw) as demo:
    gr.Markdown(
        "# 🛺 Meter Sach\n"
        "Say the ride in **any mix of English / हिन्दी / ಕನ್ನಡ / தமிழ்** - say the **km**, or just the **start and destination names** "
        "(e.g. *Majestic to Mantri Square*) and I'll find the distance. Get the real meter fare and whether the driver's price is fair."
    )
    with gr.Row():
        with gr.Column():
            mic = gr.Audio(sources=["microphone", "upload"], type="numpy",
                           label="🎙️ Speak (max ~30 sec)")
            typed = gr.Textbox(label="...or type it (backup if the mic fails)", lines=2)
            city = gr.Dropdown(choices=[(v["label"], k) for k, v in TARIFFS.items()],
                               value=DEFAULT_CITY, label="City (used if you don't say it / no GPS)")
            with gr.Row():
                loc_btn = gr.Button("📍 Use my location", size="sm")
                lat = gr.Number(label="Latitude", precision=6, interactive=True)
                lon = gr.Number(label="Longitude", precision=6, interactive=True)
            with gr.Row():
                from_box = gr.Textbox(label="From (optional)", placeholder="blank = my location, or what you said", scale=1)
                to_box = gr.Textbox(label="To (optional)", placeholder="e.g. Mantri Square", scale=1)
            use_gps = gr.Checkbox(value=True, label="📍 Start from my location when I don't say a start place (untick to ride between two other places)")
            go = gr.Button("Check my fare", variant="primary")
            gr.Examples(EXAMPLES, inputs=typed, label="Try these")
        with gr.Column():
            out_audio = gr.Audio(label="🔊 Reply", autoplay=True, type="filepath")
            out_text = gr.Textbox(label="What was spoken", lines=3)
            out_md = gr.Markdown()
            with gr.Accordion("Under the hood", open=True):
                out_tr = gr.Textbox(label="Prisma transcript (indic-auto)")
                out_js = gr.JSON(label="LLM-extracted JSON")
                out_heard = gr.Audio(label="What Prisma received (after boost + noise cleanup)", type="filepath")
    loc_btn.click(fn=None, inputs=None, outputs=[lat, lon], js=LOCATION_JS)
    demo.load(fn=None, inputs=None, outputs=[lat, lon], js=LOCATION_JS)  # ask for location when the page opens
    go.click(run, [mic, typed, city, lat, lon, from_box, to_box, use_gps], [out_audio, out_text, out_md, out_tr, out_js, out_heard])

if __name__ == "__main__":
    _launch_kw = {"head": MIC_JS} if _use_js and "head" in inspect.signature(demo.launch).parameters else {}
    if os.getenv("SHARE") == "1":  # public https link: lets you use your PHONE's mic/browser
        _launch_kw["share"] = True
    demo.launch(**_launch_kw)