import asyncio
from typing import Any, Callable, Coroutine
#一个一个推理，防止同时推理多个
class InferenceQueue:
    def __init__(self) -> None:
        self._sem = asyncio.Semaphore(1)

    async def submit(self, job: Callable[[], Coroutine[Any, Any, Any]]) -> Any:
        async with self._sem:
            return await job()

queue = InferenceQueue()
