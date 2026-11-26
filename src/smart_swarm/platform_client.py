"""Client for A's local v1 service and its effective compute reservation."""

from contextlib import contextmanager, suppress
import json
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
import uuid

from .contract import VERSION, ProtocolError


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Robot Platform redirected a scoped request")


class PlatformClient:
    def __init__(self, origin, recorder_token=None):
        parsed = urlsplit(origin)
        if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or not parsed.port
                or parsed.path not in ("", "/") or parsed.username or parsed.query or parsed.fragment):
            raise ValueError("Robot Platform origin must be an explicit loopback port")
        self.origin = origin.rstrip("/")
        self.token = recorder_token
        self.opener = build_opener(ProxyHandler({}), NoRedirect())

    def request(self, method, path, body=None):
        request_id = uuid.uuid4().hex
        headers = {"X-Request-ID": request_id}
        data = None
        if body is not None:
            data = json.dumps({**body, "protocol_version": VERSION, "request_id": request_id}).encode()
            headers["Content-Type"] = "application/json"
        if self.token:
            headers["X-Robot-Recorder"] = self.token
        req = Request(self.origin + path, data=data, method=method, headers=headers)
        try:
            with self.opener.open(req, timeout=5) as response:
                envelope = json.load(response)
        except HTTPError as error:
            envelope = json.load(error)
        if envelope.get("protocol_version") != VERSION or envelope.get("request_id") != request_id:
            raise ValueError("Robot Platform response version or request ID mismatch")
        if envelope.get("error"):
            err = envelope["error"]
            raise ProtocolError(err["code"], err["message"], err.get("retryable", False), err.get("details", {}))
        return envelope["data"]

    def get(self, path):
        return self.request("GET", path)

    def post(self, path, body):
        return self.request("POST", path, body)

    def delete(self, path):
        return self.request("DELETE", path)


def production_client(config):
    from .episode import load

    record = load(config["platform_handover"] / "compute_reservation.json")
    endpoint = record.get("production_endpoint")
    if record.get("status") != "available" or not endpoint:
        raise ProtocolError("OFFLINE", record["reason"])
    if endpoint != "/v1/compute/reservations" or not config.get("platform_production_origin"):
        raise ProtocolError("OFFLINE", "A production compute origin is not configured for the verified endpoint")
    return PlatformClient(config["platform_production_origin"])


@contextmanager
def reservation(client, job, *, host="homebrain", ttl=3600, memory_mb=2048):
    root = "/v1/compute/reservations"
    grant = client.post(root, {"host_id": host, "job_id": job, "device": "cpu",
                               "memory_limit_mb": memory_mb, "requested_ttl_s": ttl,
                               "protected_activities": ["hardware_recording", "hardware_inference"]})
    path = f"{root}/{grant['reservation_id']}"
    try:
        if host == "homebrain" and grant.get("effective_lease_backend") in (None, "fixture-lease"):
            raise ProtocolError("OFFLINE", "A fixture grant cannot admit a Homebrain research job")
        yield grant, lambda: client.post(f"{path}/heartbeat", {})
    finally:
        with suppress(ProtocolError, OSError):
            client.delete(path)
