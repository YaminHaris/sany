import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
OUT = sys.argv[1]
class H(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers['Content-Length']))
        with open(OUT, 'ab') as f:
            f.write(body + b'\n')
        resp = {"message": {"role": "assistant", "content": "ok"}, "done": True, "done_reason": "stop", "prompt_eval_count": 1, "eval_count": 1}
        data = json.dumps(resp).encode()
        self.send_response(200); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)
    def log_message(self, *a): pass
HTTPServer(('127.0.0.1', 11999), H).serve_forever()
