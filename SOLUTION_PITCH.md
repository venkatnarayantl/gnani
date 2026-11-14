# Meter Sach: Solution Pitch (1 page)

**Team:** Venkat Narayan T L, Sri Venkateswara College of Engineering (SVCE)
**Challenge:** Gnani AI, Great Indian AI Internship Challenge

## Problem
Every day, millions of Indians take auto-rickshaws where the driver quotes a price and the passenger has no quick way to check it. Newcomers, students and travelers who don't speak the local language are overcharged the most. The official meter fare exists (base fare, per-km rate, night surcharge, waiting charge), but nobody can work it out while standing on the road.

## Solution
**Meter Sach** is a voice fare checker. The user speaks the ride in any mix of English, Hindi, Kannada and Tamil, for example *"Majestic se Indiranagar, raat 11 baje, driver 450 maang raha hai"*. It replies by voice, in the same language, with the real meter fare and whether the driver's price is fair, high or an overcharge. If no distance is said, it finds the road distance from the place names (or from the user's GPS).

## How it works
1. **Gnani Prisma v3.0 (`indic-auto`)** transcribes code-mixed Kanglish, Hinglish and Tanglish speech.
2. **An LLM** turns the messy transcript into strict JSON (start, destination, km, time, waiting time, quoted price, language). There is no keyword matching.
3. **Maps** (OpenStreetMap, or Google Maps if a key is set) gives the road distance.
4. **Python** computes the exact fare from the official tariff table. The LLM never guesses a price.
5. **Gnani Timbre v2.5** speaks the verdict in the user's own language, and a fare breakdown is shown on screen.

## Why it is different
- **Exact, not guessed:** the LLM only extracts facts. The maths is deterministic Python.
- **Voice-first and multilingual:** it works the way people actually talk, mixing languages mid-sentence.
- **Robust in real conditions:** audio cleanup (noise reduction, auto gain, 16 kHz resampling), retries, and several LLM models raced for a fast reply.

## Impact
Fair-price information at the moment of the ride, in the user's own language, helps commuters avoid overcharging and builds trust between drivers and passengers.

## Gnani APIs used
Prisma v3.0 speech-to-text (`indic-auto`) and Timbre v2.5 text-to-speech, through the official REST APIs.

## Roadmap
The same engine (voice, then LLM extraction, then exact tariff math, then voice reply) extends to metro and bus fares, cabs versus meter, and electricity and water bills.

## Honest notes
Tariff values in the demo are sample values and must be checked against the current official fare notification. Fares are estimates, not an official ruling.
