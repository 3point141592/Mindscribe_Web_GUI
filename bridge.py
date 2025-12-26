import json
import queue
import threading
import uuid
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

# Core waits for this (frontend sends {type:"gui_ready"} once WebSocket is open)
gui_ready = threading.Event()

# Frontend -> core commands
cmd_q: "queue.Queue[dict]" = queue.Queue(maxsize=2000)

# Cache of last "state" messages so new clients can render immediately
_STATE_TYPES = ("init", "gate_info", "convo_spec", "update", "last_window")
_state_lock = threading.Lock()
_last_state: Dict[str, str] = {}  # msg_type -> JSON string


@dataclass
class ClientOutbox:
    """
    Per-client outgoing buffers.
      - ctrl_q holds low-rate messages (bounded, drop-oldest on overflow)
      - signal_latest holds only the newest high-rate signal frame
    """
    ctrl_q: "queue.Queue[str]"
    _sig_lock: threading.Lock
    _sig_latest: Optional[str]
    _sig_dirty: bool

    @staticmethod
    def create(ctrl_max: int = 800) -> "ClientOutbox":
        return ClientOutbox(
            ctrl_q=queue.Queue(maxsize=ctrl_max),
            _sig_lock=threading.Lock(),
            _sig_latest=None,
            _sig_dirty=False,
        )

    def offer(self, msg_type: str, payload_json: str) -> None:
        """Non-blocking offer; drops data if client is slow."""
        if msg_type == "signal":
            with self._sig_lock:
                self._sig_latest = payload_json
                self._sig_dirty = True
            return

        try:
            self.ctrl_q.put_nowait(payload_json)
        except queue.Full:
            # Drop one oldest and retry
            try:
                _ = self.ctrl_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.ctrl_q.put_nowait(payload_json)
            except queue.Full:
                pass

    def pop_ctrl_nowait(self) -> Optional[str]:
        try:
            return self.ctrl_q.get_nowait()
        except queue.Empty:
            return None

    def pop_signal_if_dirty(self) -> Optional[str]:
        with self._sig_lock:
            if (not self._sig_dirty) or (self._sig_latest is None):
                return None
            payload = self._sig_latest
            self._sig_dirty = False
            return payload


_clients_lock = threading.Lock()
_clients: Dict[str, ClientOutbox] = {}


def register_client() -> Tuple[str, ClientOutbox]:
    """Register a new client and preload cached state messages."""
    cid = uuid.uuid4().hex
    out = ClientOutbox.create()

    with _state_lock:
        for t in _STATE_TYPES:
            payload = _last_state.get(t)
            if payload:
                out.offer(t, payload)

    with _clients_lock:
        _clients[cid] = out

    return cid, out


def unregister_client(cid: str) -> None:
    with _clients_lock:
        _clients.pop(cid, None)


def emit(msg: dict) -> None:
    """
    Core -> all clients (broadcast).
    Non-blocking by design.
    """
    msg_type = str(msg.get("type", ""))
    payload = json.dumps(msg, ensure_ascii=False)

    if msg_type in _STATE_TYPES:
        with _state_lock:
            _last_state[msg_type] = payload

    with _clients_lock:
        outs: List[ClientOutbox] = list(_clients.values())

    for out in outs:
        out.offer(msg_type, payload)


def push_cmd(msg: dict) -> None:
    """
    Web -> core (enqueue a command).
    Non-blocking; drops oldest on overflow.
    """
    t = str(msg.get("type", "")).lower()
    if t == "gui_ready":
        gui_ready.set()
        return

    try:
        cmd_q.put_nowait(msg)
    except queue.Full:
        try:
            _ = cmd_q.get_nowait()
        except queue.Empty:
            pass
        try:
            cmd_q.put_nowait(msg)
        except queue.Full:
            pass


def poll_cmd_nowait() -> Optional[dict]:
    try:
        return cmd_q.get_nowait()
    except queue.Empty:
        return None
