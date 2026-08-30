import json
import logging
from typing import Dict, List

from fastapi import WebSocket

logger = logging.getLogger("websocket_manager")


class WSManager:
    def __init__(self):
        self._connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, engine_id: str, websocket: WebSocket):
        await websocket.accept()
        self._connections.setdefault(engine_id, []).append(websocket)

    def disconnect(self, engine_id: str, websocket: WebSocket):
        conns = self._connections.get(engine_id, [])
        if websocket in conns:
            conns.remove(websocket)
        if not conns and engine_id in self._connections:
            del self._connections[engine_id]

    async def broadcast(self, engine_id: str, message: dict):
        conns = self._connections.get(engine_id, [])
        dead = []
        for ws in conns:
            try:
                await ws.send_text(json.dumps(message))
            except Exception as e:
                logger.warning("Failed to send to a websocket client: %s", e)
                dead.append(ws)
        for ws in dead:
            self.disconnect(engine_id, ws)


ws_manager = WSManager()
