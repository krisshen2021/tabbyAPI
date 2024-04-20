"""Common utility functions"""

import asyncio
import socket
import traceback
from fastapi import Request
from loguru import logger
from pydantic import BaseModel
from typing import Optional

from common.concurrency import release_semaphore


class TabbyRequestErrorMessage(BaseModel):
    """Common request error type."""

    message: str
    trace: Optional[str] = None


class TabbyRequestError(BaseModel):
    """Common request error type."""

    error: TabbyRequestErrorMessage


def get_generator_error(message: str, exc_info: bool = True):
    """Get a generator error."""

    generator_error = handle_request_error(message, exc_info)

    return generator_error.model_dump_json()


def handle_request_error(message: str, exc_info: bool = True):
    """Log a request error to the console."""

    error_message = TabbyRequestErrorMessage(
        message=message, trace=traceback.format_exc()
    )

    request_error = TabbyRequestError(error=error_message)

    # Log the error and provided message to the console
    if error_message.trace and exc_info:
        logger.error(error_message.trace)

    logger.error(f"Sent to request: {message}")

    return request_error


def handle_request_disconnect(message: str):
    """Wrapper for handling for request disconnection."""

    release_semaphore()
    logger.error(message)


async def request_disconnect_loop(request: Request):
    """Polls for a starlette request disconnect."""

    while not await request.is_disconnected():
        await asyncio.sleep(0.5)


async def run_with_request_disconnect(
    request: Request, call_task: asyncio.Task, disconnect_message: str
):
    """Utility function to cancel if a request is disconnected."""

    _, unfinished = await asyncio.wait(
        [
            call_task,
            asyncio.create_task(request_disconnect_loop(request)),
        ],
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in unfinished:
        task.cancel()

    try:
        return call_task.result()
    except (asyncio.CancelledError, asyncio.InvalidStateError):
        handle_request_disconnect(disconnect_message)


def is_port_in_use(port: int) -> bool:
    """
    Checks if a port is in use

    From https://stackoverflow.com/questions/2470971/fast-way-to-test-if-a-port-is-in-use-using-python
    """

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0
