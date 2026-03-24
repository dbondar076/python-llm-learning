import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.request_context import request_id_var
from app.services.llm_errors import (
    LLMExtractionError,
    LLMRetryError,
    LLMTimeoutError,
)


logger = logging.getLogger(__name__)


def build_error_response(error: str, error_type: str) -> dict[str, str]:
    return {
        "error": error,
        "type": error_type,
        "request_id": request_id_var.get(),
    }


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(LLMExtractionError)
    async def handle_llm_extraction_error(
        request: Request, exc: LLMExtractionError
    ) -> JSONResponse:
        logger.warning("LLMExtractionError on %s: %s", request.url.path, exc)

        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=build_error_response(str(exc), "LLMExtractionError"),
        )

    @app.exception_handler(LLMTimeoutError)
    async def handle_llm_timeout_error(
        request: Request, exc: LLMTimeoutError
    ) -> JSONResponse:
        logger.warning("LLMTimeoutError on %s: %s", request.url.path, exc)

        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content=build_error_response(str(exc), "LLMTimeoutError"),
        )

    @app.exception_handler(LLMRetryError)
    async def handle_llm_retry_error(
        request: Request, exc: LLMRetryError
    ) -> JSONResponse:
        logger.warning("LLMRetryError on %s: %s", request.url.path, exc)

        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content=build_error_response(str(exc), "LLMRetryError"),
        )

    @app.exception_handler(RuntimeError)
    async def handle_runtime_error(
            request: Request, exc: RuntimeError
    ) -> JSONResponse:
        logger.error("RuntimeError on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=build_error_response(str(exc), "RuntimeError"),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(
            request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception("Unhandled exception on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=build_error_response("Internal server error", "UnhandledException"),
        )