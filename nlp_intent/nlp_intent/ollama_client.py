"""
Ollama LLM client for intent extraction.

Sends natural language commands to a local Ollama instance and returns
structured JSON with action, target, and location fields.
"""

import json
import re
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3"
REQUEST_TIMEOUT = 60  # seconds (phi3 cold start on CPU can be slow)

ALLOWED_ACTIONS = {"pick", "place", "move", "find"}

SYSTEM_PROMPT = (
    "You are a robotic intent parser. "
    "Given a user command, extract the intent and respond with ONLY a raw JSON object.\n\n"
    "STRICT RULES:\n"
    "1. Output ONLY the JSON object — no markdown, no code fences, no explanation.\n"
    "2. action MUST be exactly one of: pick, place, move, find\n"
    "3. target is the OBJECT being acted on — a single noun (e.g. cup, bottle, box). "
    "Remove adjectives. Set to null ONLY if no object is mentioned.\n"
    "4. location is WHERE the action happens — a single word or coordinate "
    "(e.g. table, shelf, kitchen, or x:5,y:3 for coordinates). "
    "Set to null if not mentioned.\n\n"
    "Schema: {\"action\": string, \"target\": string|null, \"location\": string|null}\n\n"
    "Examples:\n"
    "User: pick up the red cup from the table\n"
    "{\"action\":\"pick\",\"target\":\"cup\",\"location\":\"table\"}\n"
    "User: find the bottle\n"
    "{\"action\":\"find\",\"target\":\"bottle\",\"location\":null}\n"
    "User: move to x 5 y 3\n"
    "{\"action\":\"move\",\"target\":null,\"location\":\"x:5,y:3\"}\n"
    "User: go to the kitchen\n"
    "{\"action\":\"move\",\"target\":null,\"location\":\"kitchen\"}\n"
    "User: place the box on the shelf\n"
    "{\"action\":\"place\",\"target\":\"box\",\"location\":\"shelf\"}"
)

FALLBACK_RESPONSE = {"action": "unknown", "target": None, "location": None}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _preprocess_command(command: str) -> str:
    """Lowercase, strip whitespace."""
    return command.lower().strip()


def _strip_markdown(text: str) -> str:
    """Remove markdown code fences if present."""
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    return text.strip()


def _extract_json_from_text(text: str) -> dict | None:
    """Try to extract a JSON object from raw LLM output."""
    text = _strip_markdown(text)

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON block in noisy output
    match = re.search(r'\{[^{}]+\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def _validate_intent(data: dict) -> dict:
    """Validate and normalize the parsed intent dictionary."""
    required_keys = {"action", "target", "location"}

    # Check keys
    if not required_keys.issubset(data.keys()):
        return FALLBACK_RESPONSE.copy()

    action = str(data["action"]).lower().strip()
    target = data.get("target")
    location = data.get("location")

    # Normalize target
    if isinstance(target, str):
        target = target.lower().strip()
        if not target or target == "null":
            target = None
        else:
            # Take last word (strip adjectives like "red cup" → "cup")
            target = target.split()[-1]

    # Normalize location
    if isinstance(location, str):
        location = location.lower().strip()
        if not location or location == "null":
            location = None

    # Validate action
    if action not in ALLOWED_ACTIONS:
        action = "unknown"

    return {"action": action, "target": target, "location": location}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_intent(command: str,
                   model: str = MODEL_NAME,
                   api_url: str = OLLAMA_API_URL,
                   timeout: int = REQUEST_TIMEOUT) -> dict:
    """
    Send a natural language command to Ollama and return a validated intent dict.

    Returns:
        {"action": str, "target": str|None, "location": str|None}
    """
    command = _preprocess_command(command)
    if not command:
        return FALLBACK_RESPONSE.copy()

    payload = {
        "model": model,
        "system": SYSTEM_PROMPT,
        "prompt": command,
        "stream": False,
        "keep_alive": "5m",
        "options": {
            "temperature": 0.0,
            "num_predict": 80,
        },
    }

    try:
        resp = requests.post(api_url, json=payload, timeout=timeout)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        print("[ollama_client] ERROR: Request timed out")
        return FALLBACK_RESPONSE.copy()
    except requests.exceptions.ConnectionError:
        print("[ollama_client] ERROR: Cannot connect to Ollama at", api_url)
        return FALLBACK_RESPONSE.copy()
    except requests.exceptions.RequestException as e:
        print(f"[ollama_client] ERROR: {e}")
        return FALLBACK_RESPONSE.copy()

    raw = resp.json().get("response", "")
    parsed = _extract_json_from_text(raw)

    if parsed is None:
        print(f"[ollama_client] WARN: Could not parse JSON from: {raw!r}")
        return FALLBACK_RESPONSE.copy()

    return _validate_intent(parsed)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_commands = [
        "pick up the red cup from the table",
        "move to the kitchen",
        "find the bottle on the shelf",
        "place the box near the door",
        "go to x 5 y 3",
        "do a backflip",
        "",
    ]

    for cmd in test_commands:
        print(f"\nCommand: {cmd!r}")
        result = extract_intent(cmd)
        print(f"  Intent: {json.dumps(result)}")
