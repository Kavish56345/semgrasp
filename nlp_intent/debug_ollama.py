"""Quick debug script to see raw phi3 output."""
import requests, json

SYSTEM_PROMPT = (
    "You are a robotic intent parser. "
    "Given a user command, respond with ONLY a JSON object. "
    "No text before or after the JSON. No markdown. No explanation.\n\n"
    "Schema:\n"
    '{"action": "<ACTION>", "target": "<TARGET_OR_NULL>", "location": "<LOCATION_OR_NULL>"}\n\n'
    "Rules:\n"
    "- action MUST be one of: pick, place, move, find\n"
    "- target must be a SINGLE WORD object noun (e.g. cup, bottle, box). "
    "Strip adjectives like color or size.\n"
    "- location must be a SINGLE WORD or short noun (e.g. table, shelf, kitchen). "
    "Use null if not mentioned.\n"
    "- target and location must be null (not a string) if not specified.\n"
    "- Output ONLY the JSON object. Nothing else."
)

tests = [
    "pick up the red cup from the table",
    "move to the kitchen",
    "find the bottle",
]

for cmd in tests:
    print(f"\n{'='*60}")
    print(f"Command: {cmd}")
    try:
        r = requests.post("http://localhost:11434/api/generate", json={
            "model": "phi3",
            "system": SYSTEM_PROMPT,
            "prompt": cmd,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 80},
        }, timeout=120)
        raw = r.json().get("response", "")
        print(f"Raw response: {raw!r}")
    except Exception as e:
        print(f"Error: {e}")
