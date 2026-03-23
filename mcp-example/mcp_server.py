"""
Simple MCP HTTP server exposing tools via REST endpoints.

Endpoints:
  GET  /tools  → list available tools
  POST /call   → call a tool with arguments

Usage:
  python mcp_server.py            # default port 9500
  python mcp_server.py --port 9501
"""

import json
import math
import datetime
import hashlib
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler


# ─── Tool definitions ────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "calculate",
        "description": "Evaluate a math expression. Supports +, -, *, /, **, sqrt, sin, cos, pi, e.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression to evaluate"}
            },
            "required": ["expression"],
        },
    },
    {
        "name": "datetime_now",
        "description": "Get the current date and time in ISO 8601 format, with optional timezone offset.",
        "input_schema": {
            "type": "object",
            "properties": {
                "utc": {"type": "boolean", "description": "If true, return UTC time. Default false (local)."}
            },
        },
    },
    {
        "name": "text_stats",
        "description": "Compute statistics for a block of text: word count, char count, sentence count, most frequent words.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to analyze"}
            },
            "required": ["text"],
        },
    },
    {
        "name": "hash_text",
        "description": "Compute a hash digest of the given text. Supported algorithms: md5, sha1, sha256.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to hash"},
                "algorithm": {"type": "string", "description": "Hash algorithm (md5, sha1, sha256). Default: sha256"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "unit_convert",
        "description": "Convert between common units. Supported: temperature (c/f/k), length (m/km/mi/ft), weight (kg/lb/oz).",
        "input_schema": {
            "type": "object",
            "properties": {
                "value": {"type": "number", "description": "Numeric value to convert"},
                "from_unit": {"type": "string", "description": "Source unit (e.g. 'c', 'km', 'kg')"},
                "to_unit": {"type": "string", "description": "Target unit (e.g. 'f', 'mi', 'lb')"},
            },
            "required": ["value", "from_unit", "to_unit"],
        },
    },
]


# ─── Tool implementations ────────────────────────────────────────────────────

def tool_calculate(args):
    expr = args.get("expression", "")
    safe_ns = {
        "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
        "tan": math.tan, "log": math.log, "log10": math.log10,
        "abs": abs, "round": round, "pi": math.pi, "e": math.e,
        "pow": pow, "ceil": math.ceil, "floor": math.floor,
    }
    try:
        result = eval(expr, {"__builtins__": {}}, safe_ns)
        return {"expression": expr, "result": result}
    except Exception as e:
        return {"expression": expr, "error": str(e)}


def tool_datetime_now(args):
    utc = args.get("utc", False)
    now = datetime.datetime.utcnow() if utc else datetime.datetime.now()
    return {
        "iso": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "weekday": now.strftime("%A"),
        "utc": utc,
    }


def tool_text_stats(args):
    text = args.get("text", "")
    words = text.split()
    sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]

    # Word frequency
    freq = {}
    for w in words:
        w_clean = w.strip(".,!?;:\"'()[]{}").lower()
        if len(w_clean) > 2:
            freq[w_clean] = freq.get(w_clean, 0) + 1

    top_words = sorted(freq.items(), key=lambda x: -x[1])[:5]

    return {
        "word_count": len(words),
        "char_count": len(text),
        "sentence_count": len(sentences),
        "top_words": [{"word": w, "count": c} for w, c in top_words],
    }


def tool_hash_text(args):
    text = args.get("text", "")
    algo = args.get("algorithm", "sha256").lower()
    data = text.encode("utf-8")

    if algo == "md5":
        digest = hashlib.md5(data).hexdigest()
    elif algo == "sha1":
        digest = hashlib.sha1(data).hexdigest()
    elif algo == "sha256":
        digest = hashlib.sha256(data).hexdigest()
    else:
        return {"error": f"Unsupported algorithm: {algo}"}

    return {"text_length": len(text), "algorithm": algo, "digest": digest}


def tool_unit_convert(args):
    value = args.get("value", 0)
    from_u = args.get("from_unit", "").lower()
    to_u = args.get("to_unit", "").lower()

    conversions = {
        ("c", "f"): lambda v: v * 9 / 5 + 32,
        ("f", "c"): lambda v: (v - 32) * 5 / 9,
        ("c", "k"): lambda v: v + 273.15,
        ("k", "c"): lambda v: v - 273.15,
        ("m", "km"): lambda v: v / 1000,
        ("km", "m"): lambda v: v * 1000,
        ("km", "mi"): lambda v: v * 0.621371,
        ("mi", "km"): lambda v: v * 1.60934,
        ("m", "ft"): lambda v: v * 3.28084,
        ("ft", "m"): lambda v: v / 3.28084,
        ("kg", "lb"): lambda v: v * 2.20462,
        ("lb", "kg"): lambda v: v / 2.20462,
        ("kg", "oz"): lambda v: v * 35.274,
        ("oz", "kg"): lambda v: v / 35.274,
    }

    key = (from_u, to_u)
    if key not in conversions:
        return {"error": f"Unsupported conversion: {from_u} -> {to_u}"}

    result = conversions[key](value)
    return {"value": value, "from": from_u, "to": to_u, "result": round(result, 4)}


TOOL_FNS = {
    "calculate": tool_calculate,
    "datetime_now": tool_datetime_now,
    "text_stats": tool_text_stats,
    "hash_text": tool_hash_text,
    "unit_convert": tool_unit_convert,
}


# ─── HTTP handler ─────────────────────────────────────────────────────────────

class MCPHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # Suppress default request logging
        pass

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/tools":
            self._send_json(TOOLS)
        elif self.path == "/health":
            self._send_json({"status": "ok"})
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        if self.path == "/call":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}

            tool_name = body.get("tool", "")
            arguments = body.get("arguments", {})

            fn = TOOL_FNS.get(tool_name)
            if fn is None:
                self._send_json({"error": f"Unknown tool: {tool_name}"}, 404)
                return

            try:
                result = fn(arguments)
                self._send_json(result)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
        else:
            self._send_json({"error": "Not found"}, 404)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    port = 9500
    host = "0.0.0.0"
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--port" and i + 2 < len(sys.argv):
            port = int(sys.argv[i + 2])
        if arg == "--host" and i + 2 < len(sys.argv):
            host = sys.argv[i + 2]

    server = HTTPServer((host, port), MCPHandler)
    print(f"MCP server running on http://{host}:{port}")
    print(f"  GET  /tools  -> list tools")
    print(f"  POST /call   -> call a tool")
    print(f"  GET  /health -> health check")
    sys.stdout.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
