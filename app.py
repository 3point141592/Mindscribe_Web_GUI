import json
import threading
import time
from flask import Flask, render_template
from flask_sock import Sock

import bridge

try:
    from simple_websocket import ConnectionClosed
except Exception:
    ConnectionClosed = Exception

app = Flask(__name__, template_folder="templates")

# Keep websockets alive with ping/pong
app.config["SOCK_SERVER_OPTIONS"] = {"ping_interval": 25}

sock = Sock(app)

_core_started = False
_core_lock = threading.Lock()


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    return {"ok": True}


def start_core_once():
    """
    Start MindScribe core loop once (in a background daemon thread).

    IMPORTANT:
      Use app.run(..., use_reloader=False) so Flask doesn't start twice.
    """
    global _core_started
    with _core_lock:
        if _core_started:
            return
        _core_started = True

        def runner():
            import mindscribe_core
            mindscribe_core.run()

        t = threading.Thread(target=runner, daemon=True)
        t.start()


@sock.route("/ws")
def ws_route(ws):
    """
    WebSocket pump (robust across flask-sock/simple-websocket versions).

    Why this design:
      - Some installs don't support ws.receive(timeout=...) and will crash (client sees code=1006).
      - Even when timeout exists, non-blocking receive may return None spuriously.

    Strategy:
      - Receiver thread blocks on ws.receive() and pushes inbound messages to bridge.
      - This handler thread sends outbound messages (ctrl + rate-limited latest signal).
      - Only one thread sends on the websocket (avoids write contention).
    """
    start_core_once()
    cid, out = bridge.register_client()
    print(f"[WS] Client connected: {cid[:8]}")
    bridge.emit({"type": "log", "text": f"Client connected: {cid[:8]}"})

    closed = threading.Event()

    def rx_loop():
        try:
            while True:
                data = ws.receive()  # blocking receive
                if data is None:
                    break

                if isinstance(data, (bytes, bytearray)):
                    try:
                        data = data.decode("utf-8", errors="ignore")
                    except Exception:
                        continue

                try:
                    msg = json.loads(data)
                except Exception:
                    continue

                bridge.push_cmd(msg)

        except Exception as e:
            print(f"[WS] RX error for {cid[:8]}: {e}")
        finally:
            closed.set()

    threading.Thread(target=rx_loop, daemon=True).start()

    last_signal_sent = 0.0
    SIGNAL_MAX_HZ = 30.0  # network-friendly cap

    try:
        while not closed.is_set():
            # Outbound: control messages
            for _ in range(50):
                payload = out.pop_ctrl_nowait()
                if payload is None:
                    break
                try:
                    ws.send(payload)
                except Exception as e:
                    print(f"[WS] send error (ctrl) for {cid[:8]}: {e}")
                    raise

            # Outbound: signal (latest only), rate-limited
            now = time.time()
            if now - last_signal_sent >= (1.0 / SIGNAL_MAX_HZ):
                sig = out.pop_signal_if_dirty()
                if sig is not None:
                    try:
                        ws.send(sig)
                    except Exception as e:
                        print(f"[WS] send error (signal) for {cid[:8]}: {e}")
                        raise
                    last_signal_sent = now

            time.sleep(0.01)

    except Exception as e:
        print(f"[WS] TX loop error for {cid[:8]}: {e}")
    finally:
        bridge.unregister_client(cid)
        print(f"[WS] Client disconnected: {cid[:8]}")
        bridge.emit({"type": "log", "text": f"Client disconnected: {cid[:8]}"})


if __name__ == "__main__":

    # Disable reloader so core doesn't start twice.
    app.run(host="0.0.0.0", port=8000, debug=False, use_reloader=False, threaded=True)
