"""
RecallForge — local dev server
Serves the app AND proxies Ollama requests to avoid browser CORS issues.

Run:  python server.py
Then open:  http://localhost:8080
"""

import http.server
import urllib.request
import urllib.error
import json
import os

PORT = 8080
OLLAMA = "http://localhost:11434"

class Handler(http.server.SimpleHTTPRequestHandler):

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_POST(self):
        if self.path.startswith("/api/"):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                req = urllib.request.Request(
                    OLLAMA + self.path,
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = resp.read()
                self.send_response(200)
                self._cors()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(data)
            except urllib.error.URLError as e:
                self.send_response(502)
                self._cors()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, fmt, *args):
        print(f"  {self.address_string()} -> {fmt % args}")

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("""
RecallForge local server
  App:    http://localhost:{}
  Ollama: proxied from port 11434

Press Ctrl+C to stop.
""".format(PORT))

with http.server.HTTPServer(("", PORT), Handler) as httpd:
    httpd.serve_forever()
