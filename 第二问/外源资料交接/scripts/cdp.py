"""Minimal Chrome DevTools Protocol client using only the Python standard library.

Implements just enough of RFC 6455 (client side, masked frames, text + ping/pong)
to talk to a locally running Chrome launched with --remote-debugging-port.
"""

import base64
import json
import os
import socket
import struct
import time
import urllib.request

DEBUG_HOST = "127.0.0.1"
DEBUG_PORT = int(os.environ.get("CDP_PORT", "9222"))


def http_json(path):
    url = "http://%s:%d%s" % (DEBUG_HOST, DEBUG_PORT, path)
    with urllib.request.urlopen(url, timeout=15) as r:
        return json.loads(r.read().decode("utf-8"))


def list_targets():
    return http_json("/json/list")


def new_tab(url="about:blank"):
    url_enc = urllib.request.quote(url, safe="")
    return http_json("/json/new?%s" % url_enc)


class WebSocket:
    def __init__(self, ws_url, timeout=30.0):
        # ws://host:port/path
        assert ws_url.startswith("ws://"), ws_url
        rest = ws_url[5:]
        hostport, _, path = rest.partition("/")
        host, _, port = hostport.partition(":")
        self.sock = socket.create_connection((host, int(port or 80)), timeout=timeout)
        self.sock.settimeout(timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        req = (
            "GET /%s HTTP/1.1\r\n"
            "Host: %s:%s\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            "Sec-WebSocket-Key: %s\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n" % (path, host, port, key)
        )
        self.sock.sendall(req.encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise RuntimeError("websocket handshake failed: connection closed")
            buf += chunk
        head, _, self.buf = buf.partition(b"\r\n\r\n")
        if b" 101 " not in head.split(b"\r\n")[0]:
            raise RuntimeError("websocket handshake rejected: %r" % head[:200])
        self._id = 0
        self.events = []

    # ---- framing -------------------------------------------------------
    def _send_frame(self, payload, opcode=0x1):
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        header = bytearray()
        header.append(0x80 | opcode)
        n = len(payload)
        if n < 126:
            header.append(0x80 | n)
        elif n < 65536:
            header.append(0x80 | 126)
            header += struct.pack(">H", n)
        else:
            header.append(0x80 | 127)
            header += struct.pack(">Q", n)
        mask = os.urandom(4)
        header += mask
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self.sock.sendall(bytes(header) + masked)

    def _recv_exact(self, n):
        while len(self.buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise RuntimeError("websocket closed")
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def _recv_frame(self):
        b0, b1 = self._recv_exact(2)
        opcode = b0 & 0x0F
        masked = b1 & 0x80
        n = b1 & 0x7F
        if n == 126:
            n = struct.unpack(">H", self._recv_exact(2))[0]
        elif n == 127:
            n = struct.unpack(">Q", self._recv_exact(8))[0]
        mask = self._recv_exact(4) if masked else None
        data = self._recv_exact(n) if n else b""
        if mask:
            data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
        return opcode, data

    # ---- rpc -----------------------------------------------------------
    def send(self, method, params=None):
        self._id += 1
        msg = {"id": self._id, "method": method, "params": params or {}}
        self._send_frame(json.dumps(msg))
        return self._id

    def _handle_frame(self):
        opcode, data = self._recv_frame()
        if opcode == 0x9:  # ping
            self._send_frame(data, opcode=0xA)
            return None
        if opcode == 0x8:  # close
            raise RuntimeError("websocket closed by peer")
        if opcode not in (0x1, 0x0):
            return None
        msg = json.loads(data.decode("utf-8"))
        if "id" not in msg:
            self.events.append(msg)
            if len(self.events) > 5000:
                del self.events[:2000]
        return msg

    def drain(self, seconds):
        """Read frames for `seconds`, buffering events. Useful to collect network events."""
        deadline = time.time() + seconds
        while True:
            remain = deadline - time.time()
            if remain <= 0:
                return
            self.sock.settimeout(min(1.0, remain))
            try:
                self._handle_frame()
            except socket.timeout:
                continue

    def wait_for(self, msg_id, timeout=120.0):
        deadline = time.time() + timeout
        while True:
            if time.time() > deadline:
                raise TimeoutError("CDP timeout waiting for id=%s" % msg_id)
            self.sock.settimeout(max(1.0, min(5.0, deadline - time.time())))
            try:
                msg = self._handle_frame()
            except socket.timeout:
                continue
            if msg is None:
                continue
            if msg.get("id") == msg_id:
                if "error" in msg:
                    raise RuntimeError("CDP error: %s" % json.dumps(msg["error"])[:800])
                return msg.get("result", {})

    def call(self, method, params=None, timeout=120.0):
        return self.wait_for(self.send(method, params), timeout=timeout)

    def close(self):
        try:
            self._send_frame(b"", opcode=0x8)
        except Exception:
            pass
        try:
            self.sock.close()
        except Exception:
            pass


class Page:
    """Convenience wrapper around a single page target."""

    def __init__(self, target_id=None, url=None):
        if target_id is None:
            targets = [t for t in list_targets() if t.get("type") == "page"]
            if url:
                targets = [t for t in targets if url in t.get("url", "")] or targets
            if not targets:
                raise RuntimeError("no page target available")
            target_id = targets[0]["id"]
            ws_url = targets[0]["webSocketDebuggerUrl"]
        else:
            info = [t for t in list_targets() if t["id"] == target_id][0]
            ws_url = info["webSocketDebuggerUrl"]
        self.ws = WebSocket(ws_url)
        self.ws.call("Page.enable")
        self.ws.call("Runtime.enable")

    def goto(self, url, timeout=90000):
        r = self.ws.call("Page.navigate", {"url": url})
        if r.get("errorText"):
            raise RuntimeError("navigate failed: %s" % r["errorText"])
        return r

    def eval(self, expression, timeout=120000, await_promise=False):
        r = self.ws.call(
            "Runtime.evaluate",
            {
                "expression": expression,
                "returnByValue": True,
                "awaitPromise": await_promise,
                "userGesture": True,
            },
            timeout=timeout / 1000.0,
        )
        if r.get("exceptionDetails"):
            raise RuntimeError("JS exception: %s" % json.dumps(r["exceptionDetails"])[:1500])
        return r.get("result", {}).get("value")

    def close(self):
        self.ws.close()
