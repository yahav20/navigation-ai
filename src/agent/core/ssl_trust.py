"""Make Python's TLS verification use the operating-system certificate store.

On corporate Windows machines, HTTPS is often intercepted by a proxy/antivirus
that presents a certificate signed by a private root CA. That CA lives in the
Windows certificate store, but Python's ``requests``/``urllib3`` verify against
``certifi``'s static bundle instead — so calls to external APIs (e.g. Tavily)
fail with ``CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate``
and the search silently returns no results.

``truststore.inject_into_ssl()`` patches the ``ssl`` module to verify against the
OS trust store (which DOES contain the corporate root CA), fixing every HTTPS
client process-wide. This is idempotent and safe to call more than once.
"""
from __future__ import annotations

_INJECTED = False


def ensure_os_trust() -> None:
    """Route Python TLS verification through the OS certificate store (idempotent)."""
    global _INJECTED
    if _INJECTED:
        return
    try:
        import truststore  # noqa: PLC0415

        truststore.inject_into_ssl()
        _INJECTED = True
    except Exception:  # noqa: BLE001
        # truststore unavailable or injection failed — fall back to certifi defaults.
        pass
