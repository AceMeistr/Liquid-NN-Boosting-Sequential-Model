from __future__ import annotations

import getpass
import os
from typing import Any, Callable

import upstox_client
from loguru import logger
from upstox_client.rest import ApiException


_SESSION_TOKEN: str | None = None


def _is_unauthorized(exc: Exception) -> bool:
    """Return True when an API exception corresponds to HTTP 401."""
    status = getattr(exc, "status", None)
    if status == 401:
        return True
    return "401" in str(exc)


def _validate_token(token: str) -> None:
    """Validate token by calling Upstox UserApi.get_profile.

    Args:
        token: Runtime access token entered by the user.

    Raises:
        ValueError: If token is invalid or expired.
        ApiException: For non-401 Upstox failures.
    """
    config = upstox_client.Configuration()
    config.access_token = token
    client = upstox_client.ApiClient(config)

    try:
        upstox_client.UserApi(client).get_profile("2.0")
        logger.info("Upstox token validated successfully.")
    except ApiException as exc:
        if _is_unauthorized(exc):
            raise ValueError(
                "Invalid Upstox access token. Obtain a fresh token and retry."
            ) from exc
        raise


def get_access_token() -> str:
    """Retrieve access token from session cache or secure runtime prompt.

    Returns:
        Active Upstox access token for current process session.

    Raises:
        ValueError: If user enters empty token or validation fails.
    """
    global _SESSION_TOKEN

    if _SESSION_TOKEN:
        return _SESSION_TOKEN

    cached_env = os.environ.get("UPSTOX_TOKEN")
    if cached_env:
        try:
            _validate_token(cached_env)
            _SESSION_TOKEN = cached_env
            return _SESSION_TOKEN
        except Exception:
            # Cached token may be stale from a previous run/day; prompt fresh token.
            os.environ.pop("UPSTOX_TOKEN", None)

    token = getpass.getpass("Enter Upstox Access Token (not stored, session-only): ").strip()
    if not token:
        raise ValueError("Access token cannot be empty.")

    _validate_token(token)
    _SESSION_TOKEN = token
    os.environ["UPSTOX_TOKEN"] = token
    return _SESSION_TOKEN


def get_upstox_client() -> upstox_client.ApiClient:
    """Return configured Upstox API client with validated runtime token."""
    config = upstox_client.Configuration()
    config.access_token = get_access_token()
    return upstox_client.ApiClient(config)


def _refresh_bound_client_token(fn: Callable[..., Any], token: str) -> None:
    """Update token on a bound Upstox API object when possible."""
    owner = getattr(fn, "__self__", None)
    api_client = getattr(owner, "api_client", None)
    cfg = getattr(api_client, "configuration", None)
    if cfg is not None:
        setattr(cfg, "access_token", token)


def handle_401_retry(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Retry an Upstox call once on 401 by re-prompting for token.

    Args:
        fn: API callable to invoke.
        *args: Positional arguments for callable.
        **kwargs: Keyword arguments for callable.

    Returns:
        Result of API callable.

    Raises:
        RuntimeError: If second attempt still fails with 401.
        ApiException: For other non-auth API errors.
    """
    global _SESSION_TOKEN

    try:
        return fn(*args, **kwargs)
    except ApiException as exc:
        if not _is_unauthorized(exc):
            raise

    logger.warning("Upstox token expired mid-run. Re-prompting for a fresh token.")
    _SESSION_TOKEN = None
    os.environ.pop("UPSTOX_TOKEN", None)

    fresh_token = get_access_token()
    _refresh_bound_client_token(fn, fresh_token)

    try:
        return fn(*args, **kwargs)
    except ApiException as exc:
        if _is_unauthorized(exc):
            raise RuntimeError(
                "Upstox token refresh failed on retry (second 401). Aborting run."
            ) from exc
        raise
