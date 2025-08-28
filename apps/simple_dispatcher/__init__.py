"""Simple dispatcher service.

This module provides a very small stand‑in implementation used by the
integration tests.  The real project dispatches tasks to a queue and later
invokes agent services.  For the purposes of the exercises we merely record
the enqueue request and return a confirmation dictionary.
"""

from typing import Any, Dict


class SimpleDispatcher:
    """Extremely small dispatcher used in tests.

    The :meth:`enqueue` method mimics the behaviour of the production service
    by accepting a handful of parameters and returning a result mapping.
    """

    def enqueue(
        self,
        agent: str,
        aor: Dict[str, Any],
        idempotency_key: str,
        compiled_pipeline: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Pretend to enqueue a task and return a confirmation mapping."""

        return {
            "agent": agent,
            "idempotency_key": idempotency_key,
            "status": "queued",
        }


__all__ = ["SimpleDispatcher"]
