"""
Find out WHY Prisma returns an empty transcript.

  python test_gnani.py              self-test: Timbre speaks a sentence, Prisma listens to it
  python test_gnani.py my.wav       also tests your own recording (raw, then converted)

Needs GNANI_API_KEY in the environment (same terminal as for app.py).
"""
import os
import shutil
import subprocess
import sys
import tempfile

from gnani.stt import GnaniSTTClient
from gnani.tts import AudioConfig, GnaniTTSClient

if os.getenv("GNANI_API_KEY"):  # drop stray spaces / line breaks from pasting
    os.environ["GNANI_API_KEY"] = "".join(os.environ["GNANI_API_KEY"].split())
stt, tts = GnaniSTTClient(), GnaniTTSClient()
print("ffmpeg:", shutil.which("ffmpeg") or "NOT FOUND")


def run(label, path, **kw):
    try:
        r = stt.transcribe(path, **kw)
        print(f"  {label:34} -> {r.get('transcript')!r}  lang={r.get('detected_language_code')}")
    except Exception as e:  # noqa: BLE001
        print(f"  {label:34} -> ERROR {type(e).__name__}: {e}")


def suite(name, path):
    print(f"\n=== {name}: {path} ({os.path.getsize(path)} bytes) ===")
    run("v3 indic-auto", path, language_code="indic-auto", model="prisma-v3.0")
    run("v3 indic-auto + hints", path, language_code="indic-auto", model="prisma-v3.0",
        language_candidates=["hi-IN", "en-IN"])
    run("v3 hi-IN", path, language_code="hi-IN", model="prisma-v3.0")
    run("v2.5 hi-IN (default model)", path, language_code="hi-IN")
    run("v2.5 en-IN (default model)", path, language_code="en-IN")


# 1) Self-test: audio made by Gnani's own Timbre, so the audio itself is known-good
sentence = "Majestic se Indiranagar gyarah kilometer, raat gyarah baje, driver chaar sau pachaas maang raha hai"
audio = tts.synthesize(sentence, voice="Kaveri", model="timbre-v2.5", language="hi-IN",
                       audio_config=AudioConfig(container="wav", encoding="linear_pcm", sample_rate=16000))
selftest = os.path.join(tempfile.mkdtemp(), "selftest.wav")
with open(selftest, "wb") as f:
    f.write(audio)
print("Sentence spoken by Timbre:", sentence)
suite("SELF-TEST (Timbre -> Prisma)", selftest)

# 2) Your own recording, if given
if len(sys.argv) > 1:
    mine = sys.argv[1]
    suite("YOUR FILE, raw", mine)
    if shutil.which("ffmpeg"):
        conv = os.path.join(tempfile.mkdtemp(), "converted.wav")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mine, "-ac", "1", "-ar", "16000",
                        "-af", "highpass=f=100,loudnorm=I=-16:TP=-1.5:LRA=11", "-c:a", "pcm_s16le", conv], check=True)
        suite("YOUR FILE, converted", conv)
    if shutil.which("ffprobe"):
        print("\nffprobe of your file:")
        subprocess.run(["ffprobe", "-hide_banner", "-loglevel", "error", "-show_entries",
                        "stream=codec_name,sample_rate,channels,duration", "-of", "default=nw=1", mine])