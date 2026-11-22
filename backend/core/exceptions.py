from typing import Optional, Dict, Any
from fastapi import Request
from fastapi.responses import JSONResponse

class AppException(Exception):
    def __init__(
        self,
        status_code: int,
        title: str,
        detail: str,
        type_uri: str = "about:blank",
        instance: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None
    ):
        super().__init__(detail)
        self.status_code = status_code
        self.title = title
        self.detail = detail
        self.type_uri = type_uri
        self.instance = instance
        self.extra = extra or {}

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "type": self.type_uri,
            "title": self.title,
            "status": self.status_code,
            "detail": self.detail,
        }
        if self.instance:
            data["instance"] = self.instance
        if self.extra:
            data.update(self.extra)
        return data

class BadRequestException(AppException):
    def __init__(self, detail: str, extra: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=400,
            title="Bad Request",
            detail=detail,
            type_uri="https://autodata-hub.local/errors/bad-request",
            extra=extra
        )

class EntityTooLargeException(AppException):
    def __init__(self, detail: str, extra: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=413,
            title="Payload Too Large",
            detail=detail,
            type_uri="https://autodata-hub.local/errors/entity-too-large",
            extra=extra
        )

class UnsupportedMediaTypeException(AppException):
    def __init__(self, detail: str, extra: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=415,
            title="Unsupported Media Type",
            detail=detail,
            type_uri="https://autodata-hub.local/errors/unsupported-media-type",
            extra=extra
        )

class UnprocessableEntityException(AppException):
    def __init__(self, detail: str, extra: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=422,
            title="Unprocessable Entity",
            detail=detail,
            type_uri="https://autodata-hub.local/errors/unprocessable-entity",
            extra=extra
        )

class InternalEngineException(AppException):
    def __init__(self, detail: str, extra: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=500,
            title="Internal Processing Error",
            detail=detail,
            type_uri="https://autodata-hub.local/errors/internal-engine-error",
            extra=extra
        )

async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    if not exc.instance:
        exc.instance = request.url.path
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict(),
        headers={"Content-Type": "application/problem+json"}
    )
