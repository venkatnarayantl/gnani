# 🛺 Meter Sach

**Auto drivers quote random prices. Meter Sach tells you the real meter fare, by voice, in your own language.**

Speak your ride in any mix of English / हिन्दी / ಕನ್ನಡ / தமிழ். Meter Sach finds the distance, computes the official meter fare, and tells you whether the driver's price is fair, overpriced or an overcharge.

Built for the Gnani AI *Great Indian AI Internship Challenge*.

## How it works

```
Voice → Gnani Prisma (indic-auto) → LLM extracts JSON → Maps gives km → Python computes fare → Gnani Timbre speaks the verdict
```

1. **Listen:** Gnani **Prisma** STT with `indic-auto` handles Kanglish, Hinglish and Tanglish.
2. **Understand:** an LLM (Gemini Flash, with Claude Haiku as backup) turns the messy transcript into strict JSON: start, destination, km, time, waiting time, quoted price, language. No keyword matching.
3. **Find the distance:** if no km is said, the start and destination names are looked up on the map (OpenStreetMap, or Google Maps if a key is set). "From my location" uses the browser GPS.
4. **Compute the fare:** plain Python with a tariff table (base fare, per-km rate, night surcharge, waiting charge). The math is exact and not LLM-guessed.
5. **Answer by voice:** a fixed multilingual reply is spoken by Gnani **Timbre** TTS in the user's own language, and a fare breakdown is shown on screen.

## Example inputs

- `Majestic se Indiranagar 11 km, raat 11 baje, driver chaar sau pachaas maang raha hai`
- `Majestic to Mantri Square, driver 150 maang raha hai`
- `Anna Nagar to T Nagar sumaar 8 kilometer, night 12 mani, driver 300 rupaye solraar`
- `Koramangala ge hogbeku, 5 km, avanu 200 rupayi keltaane`

## Tech stack

| Part | What is used |
|---|---|
| Language and UI | Python, Gradio |
| Speech to text | Gnani Prisma (`prisma-v3.0`, `indic-auto`) |
| Understanding | Google Gemini Flash (Claude Haiku as automatic backup) |
| Maps | OpenStreetMap (Nominatim + OSRM), or Google Maps Geocoding + Distance Matrix |
| Fare logic | Python tariff table (Bengaluru, Chennai, Delhi) |
| Text to speech | Gnani Timbre (`timbre-v2.5`, voice Kaveri) |
| Audio cleanup | NumPy: resampling, noise reduction, auto gain, voice-band filter (no ffmpeg needed) |
| Libraries | `gradio`, `numpy`, `requests`, `gnani-vachana`, `google-genai`, `anthropic` |

## Run it

```
pip install -r requirements.txt
copy .env.example .env      # then put your keys in .env
python app.py
```

Open http://127.0.0.1:7860 and allow the microphone (and location, if you want "from my location").

### Keys (`.env`)

```
GNANI_API_KEY=your_gnani_key
GEMINI_API_KEY=your_gemini_key        # or ANTHROPIC_API_KEY
GOOGLE_MAPS_API_KEY=optional          # better place lookup than OpenStreetMap
```

### Optional settings

| Variable | What it does |
|---|---|
| `SHARE=1` | Public https link, so you can use your phone's mic and GPS |
| `GEMINI_MODEL` | Pick the Gemini model, e.g. `gemini-3.6-flash` |
| `LLM_PROVIDER` | `gemini` or `claude` |
| `TTS_VOICE` | Timbre voice name (default Kaveri) |
| `DENOISE=0` | Turn off noise reduction |
| `MIC_RAW=1` | Switch off the browser's own noise filter |

## Tips for good results

- Use the laptop or phone mic, not Bluetooth earpods (their mic is very quiet in call mode).
- Hold the mic 10–15 cm from your mouth.
- To check a ride between two other places, say both names, or untick "Start from my location".
- If a place is found wrongly, type the full name in the **To** box.

## Roadmap: same engine, more everyday bills

Meter Sach is a "voice → exact tariff check" engine. The LLM only extracts the facts, Python does the math from an official tariff table, and Timbre speaks the verdict. The same pipeline extends to:

| Next | What the user says | What it checks |
|---|---|---|
| 🚇 **Metro** | "Majestic se Whitefield, how much, and the smart card discount?" | Official station-to-station fare, smart card vs token, wrong-fare or overcharge at the counter |
| 🚌 **City / state buses** | "Majestic to Electronic City, AC bus, kitna ticket?" | Fare by stage and bus type (ordinary, deluxe, AC), conductor charged too much |
| 🚖 **Cabs** | "Indiranagar to airport, app shows 900" | Meter-equivalent fare vs app price, surge detection |
| ⚡ **Electricity (EB / BESCOM / TNEB)** | "This month 320 units, bill came 2400" | Slab-wise bill from the units, free-units subsidy, meter or billing mistakes |
| 💧 **Water bill** | "Water bill 1200 for 18 kilolitres" | Slab calculation and sewerage charge, whether the bill is correct |
| 🛢️ **Others** | LPG refill, parking, toll, taxi-airport | Any government-notified rate card |

Why it scales: every item is just a new tariff table plus a few fields for the LLM to extract. Adding a new service doesn't need a new model or new speech code.

## Notes and limits

- Tariff numbers are **demo values**. Verify them against the current official fare notification before relying on them.
- Fares are estimates and not an official ruling. Real routes and meters vary a little (the app allows about 10%).
- Kannada and Tamil reply wording should be reviewed by a native speaker.
- The free OpenStreetMap search can mis-match some landmarks. A Google Maps key fixes most of that.
- Never commit `.env`. It is listed in `.gitignore`.
