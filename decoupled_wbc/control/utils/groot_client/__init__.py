"""Vendored client-side glue for the Isaac-GR00T policy server.

Only re-exports ``PolicyClient`` so the WBC container can talk to a
``gr00t/eval/run_gr00t_server.py`` instance over ZMQ without importing the
upstream ``gr00t`` package (which pulls in torch + transformers).
"""

from .server_client import PolicyClient

__all__ = ["PolicyClient"]
