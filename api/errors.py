class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, details=None):
        super().__init__(message)
        self.status, self.code, self.message, self.details = status, code, message, details


def error_body(request_id, code: str, message: str, details=None) -> dict:
    return {
        "error": {"code": code, "message": message, "request_id": request_id, "details": details}
    }
