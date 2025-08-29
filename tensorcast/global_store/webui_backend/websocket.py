#  Copyright (c) 2025, TensorCast Team.

"""WebSocket manager for real-time updates."""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import WebSocket

from tensorcast.global_store.webui_backend.models import WebSocketMessage

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections and broadcasts."""

    def __init__(self, max_queue_size: int = 10000):
        """Initialize WebSocket manager.

        Args:
            max_queue_size: Maximum number of messages in queue before dropping oldest
        """
        self.active_connections: list[WebSocket] = []
        self.subscriptions: dict[WebSocket, set[str]] = {}
        self.update_queue: asyncio.Queue[WebSocketMessage] = asyncio.Queue(
            maxsize=max_queue_size
        )
        self.max_queue_size = max_queue_size
        self.dropped_messages = 0
        self.grpc_client = None
        self._last_state = {}  # Track last known state for polling

    async def connect(self, websocket: WebSocket):
        """Accept and track new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.subscriptions[websocket] = {
            "heartbeat",
            "replica_update",
            "transport",
        }  # Default topics
        logger.info(
            f"WebSocket client connected. Total connections: {len(self.active_connections)}"
        )

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.subscriptions:
            del self.subscriptions[websocket]
        logger.info(
            f"WebSocket client disconnected. Total connections: {len(self.active_connections)}"
        )

    async def subscribe(self, websocket: WebSocket, topics: list[str]):
        """Update subscription topics for a connection."""
        if websocket in self.subscriptions:
            self.subscriptions[websocket] = set(topics)
            await websocket.send_json({"status": "subscribed", "topics": topics})

    async def send_update(self, topic: str, payload: dict[str, Any]):
        """Queue an update for broadcast."""
        message = WebSocketMessage(topic=topic, payload=payload)

        # Try to add to queue, drop oldest if full
        if self.update_queue.full():
            try:
                # Drop the oldest message
                self.update_queue.get_nowait()
                self.dropped_messages += 1
                if self.dropped_messages % 100 == 0:
                    logger.warning(
                        f"WebSocket queue full, dropped {self.dropped_messages} messages total"
                    )
            except asyncio.QueueEmpty:
                pass

        try:
            self.update_queue.put_nowait(message)
        except asyncio.QueueFull:
            logger.error("Failed to add message to WebSocket queue even after dropping")

    async def broadcast_updates(self):
        """Background task to broadcast queued updates."""
        while True:
            try:
                # Batch updates - wait up to 1 second for multiple updates
                updates: list[WebSocketMessage] = []
                deadline = asyncio.get_event_loop().time() + 1.0

                while True:
                    timeout = deadline - asyncio.get_event_loop().time()
                    if timeout <= 0:
                        break

                    try:
                        message = await asyncio.wait_for(
                            self.update_queue.get(), timeout=timeout
                        )
                        updates.append(message)
                    except asyncio.TimeoutError:
                        break

                if updates:
                    await self._broadcast_batch(updates)

            except Exception as e:
                logger.error(f"Error in broadcast loop: {e}")
                await asyncio.sleep(1)

    async def _broadcast_batch(self, updates: list[WebSocketMessage]):
        """Broadcast a batch of updates to subscribed clients."""
        # Group updates by topic
        by_topic: dict[str, list[dict[str, Any]]] = {}
        for update in updates:
            if update.topic not in by_topic:
                by_topic[update.topic] = []
            by_topic[update.topic].append(update.model_dump(mode="json"))

        # Send to each connection based on subscriptions
        disconnected = []
        for websocket in self.active_connections:
            try:
                subscribed_topics = self.subscriptions.get(websocket, set())
                relevant_updates = []

                for topic, topic_updates in by_topic.items():
                    if topic in subscribed_topics:
                        relevant_updates.extend(topic_updates)

                if relevant_updates:
                    await websocket.send_json(
                        {
                            "type": "updates",
                            "timestamp": str(datetime.now(tz=timezone.utc).isoformat()),
                            "updates": relevant_updates,
                        }
                    )

            except Exception as e:
                logger.error(f"Error sending to websocket: {e}")
                disconnected.append(websocket)

        # Clean up disconnected clients
        for websocket in disconnected:
            self.disconnect(websocket)

    def set_grpc_client(self, client):
        """Set the gRPC client for polling updates."""
        self.grpc_client = client
        # Start polling task
        if self.grpc_client:
            asyncio.create_task(self._poll_for_updates())

    async def _poll_for_updates(self):
        """Poll gRPC for state changes and generate events."""
        poll_interval = 10.0  # Poll every 5 seconds

        while True:
            try:
                if not self.grpc_client:
                    await asyncio.sleep(poll_interval)
                    continue

                if len(self.active_connections) == 0:
                    await asyncio.sleep(poll_interval)
                    continue

                # Poll workers
                workers = await self.grpc_client.list_active_workers(
                    include_unavailable=True
                )
                current_workers = {w.worker_id: w for w in workers}

                # Detect worker changes
                if "workers" in self._last_state:
                    last_workers = self._last_state["workers"]

                    # Check for heartbeat updates
                    for worker_id, worker in current_workers.items():
                        if worker_id in last_workers:
                            last_worker = last_workers[worker_id]
                            if (
                                worker.last_heartbeat_timestamp
                                != last_worker.last_heartbeat_timestamp
                            ):
                                last_dt = worker.last_heartbeat_datetime
                                await self.send_update(
                                    "heartbeat",
                                    {
                                        "worker_id": worker_id,
                                        "status": "active"
                                        if worker.accepting_new_requests
                                        else "unavailable",
                                        "last_heartbeat": last_dt.isoformat()
                                        if last_dt
                                        else None,
                                        "mem_pool_available": worker.mem_pool_available_size,
                                    },
                                )

                self._last_state["workers"] = current_workers

                # Poll replicas (simplified - in production you'd track more changes)
                all_replicas = await self.grpc_client.list_replicas()

                # For now, just count total replicas
                total_replicas = sum(
                    len(replicas) for replicas in all_replicas.values()
                )
                if self._last_state.get("total_replicas") != total_replicas:
                    await self.send_update(
                        "replica_update",
                        {
                            "total_replicas": total_replicas,
                            "artifacts": len(all_replicas),
                        },
                    )
                    self._last_state["total_replicas"] = total_replicas

                await asyncio.sleep(poll_interval)

            except Exception as e:
                logger.error(f"Error polling for updates: {e}")
                await asyncio.sleep(poll_interval)
