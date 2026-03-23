"""
MCP Server using stdio (JSON-RPC over stdin/stdout).

This server reads JSON-RPC requests from stdin and writes responses to stdout.
Supports the MCP protocol methods: tools/list and tools/call.
"""

import sys
import json
import math
import hashlib
from datetime import datetime, timezone


# ── Tool implementations ─────────────────────────────────────────────────────

def calculate(expression: str) -> dict:
    """Evaluate a math expression safely."""
    allowed = {
        "abs": abs, "round": round, "min": min, "max": max,
        "pow": pow, "sum": sum, "len": len,
        "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "pi": math.pi, "e": math.e, "inf": math.inf,
        "floor": math.floor, "ceil": math.ceil,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"expression": expression, "error": str(e)}


def datetime_now(tz: str = "UTC") -> dict:
    """Get current date/time."""
    now = datetime.now(timezone.utc)
    return {
        "datetime": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "day_of_week": now.strftime("%A"),
        "timezone": tz,
    }


def text_stats(text: str) -> dict:
    """Compute text statistics."""
    words = text.split()
    sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    return {
        "characters": len(text),
        "words": len(words),
        "sentences": len(sentences),
        "avg_word_length": round(sum(len(w) for w in words) / max(len(words), 1), 2),
    }


def hash_text(text: str, algorithm: str = "sha256") -> dict:
    """Compute a hash of the text."""
    algos = {"sha256": hashlib.sha256, "md5": hashlib.md5, "sha1": hashlib.sha1}
    fn = algos.get(algorithm, hashlib.sha256)
    return {"algorithm": algorithm, "hash": fn(text.encode()).hexdigest()}


def unit_convert(value: float, from_unit: str, to_unit: str) -> dict:
    """Convert between units."""
    conversions = {
        ("celsius", "fahrenheit"): lambda v: v * 9 / 5 + 32,
        ("fahrenheit", "celsius"): lambda v: (v - 32) * 5 / 9,
        ("km", "miles"): lambda v: v * 0.621371,
        ("miles", "km"): lambda v: v / 0.621371,
        ("kg", "lbs"): lambda v: v * 2.20462,
        ("lbs", "kg"): lambda v: v / 2.20462,
        ("meters", "feet"): lambda v: v * 3.28084,
        ("feet", "meters"): lambda v: v / 3.28084,
    }
    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        result = conversions[key](value)
        return {"value": value, "from": from_unit, "to": to_unit, "result": round(result, 4)}
    return {"error": f"Unknown conversion: {from_unit} -> {to_unit}"}


def top_words(text: str, n: int = 5) -> dict:
    """Find the most frequent words in text."""
    words = text.lower().split()
    # Strip punctuation
    words = [w.strip(".,!?;:\"'()[]{}") for w in words]
    words = [w for w in words if w]
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: -x[1])[:n]
    return {"top_words": [{"word": w, "count": c} for w, c in sorted_words]}


# ── Tool registry ────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression. Supports sqrt, log, trig, etc.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression to evaluate"}
            },
            "required": ["expression"],
        },
    },
    {
        "name": "datetime_now",
        "description": "Get the current date and time in UTC.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tz": {"type": "string", "description": "Timezone (default UTC)", "default": "UTC"}
            },
        },
    },
    {
        "name": "text_stats",
        "description": "Get statistics about a text: character count, word count, sentence count, avg word length.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to analyze"}
            },
            "required": ["text"],
        },
    },
    {
        "name": "hash_text",
        "description": "Compute a cryptographic hash (sha256, md5, sha1) of the given text.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to hash"},
                "algorithm": {"type": "string", "description": "Hash algorithm", "default": "sha256"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "unit_convert",
        "description": "Convert between units (celsius/fahrenheit, km/miles, kg/lbs, meters/feet).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "value": {"type": "number", "description": "Value to convert"},
                "from_unit": {"type": "string", "description": "Source unit"},
                "to_unit": {"type": "string", "description": "Target unit"},
            },
            "required": ["value", "from_unit", "to_unit"],
        },
    },
    {
        "name": "top_words",
        "description": "Find the most frequent words in the given text.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to analyze"},
                "n": {"type": "integer", "description": "Number of top words", "default": 5},
            },
            "required": ["text"],
        },
    },
]

TOOL_HANDLERS = {
    "calculate": lambda args: calculate(args["expression"]),
    "datetime_now": lambda args: datetime_now(args.get("tz", "UTC")),
    "text_stats": lambda args: text_stats(args["text"]),
    "hash_text": lambda args: hash_text(args["text"], args.get("algorithm", "sha256")),
    "unit_convert": lambda args: unit_convert(args["value"], args["from_unit"], args["to_unit"]),
    "top_words": lambda args: top_words(args["text"], args.get("n", 5)),
}


# ── JSON-RPC handler ─────────────────────────────────────────────────────────

def handle_request(request: dict) -> dict:
    """Handle a JSON-RPC 2.0 request and return a response."""
    req_id = request.get("id", "0")
    method = request.get("method", "")
    params = request.get("params", {})

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "result": {"tools": TOOLS},
            "id": req_id,
        }

    elif method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        handler = TOOL_HANDLERS.get(tool_name)
        if not handler:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
                "id": req_id,
            }

        try:
            result = handler(arguments)
            return {
                "jsonrpc": "2.0",
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result)}]
                },
                "id": req_id,
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32000, "message": str(e)},
                "id": req_id,
            }

    elif method == "initialize":
        return {
            "jsonrpc": "2.0",
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "flowgentra-mcp-tools", "version": "1.0.0"},
                "capabilities": {"tools": {}},
            },
            "id": req_id,
        }

    else:
        return {
            "jsonrpc": "2.0",
            "error": {"code": -32601, "message": f"Method not found: {method}"},
            "id": req_id,
        }


# ── Main loop ────────────────────────────────────────────────────────────────

def main():
    """Read JSON-RPC requests from stdin, write responses to stdout."""
    # Ensure stdout is unbuffered for real-time communication
    sys.stdout.reconfigure(line_buffering=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            response = {
                "jsonrpc": "2.0",
                "error": {"code": -32700, "message": f"Parse error: {e}"},
                "id": None,
            }
            print(json.dumps(response), flush=True)
            continue

        response = handle_request(request)
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
