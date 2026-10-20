"""Every Goal015C fixture forbids external sockets; no real API exercise."""
import socket

import pytest


@pytest.fixture(autouse=True)
def no_external_transport(monkeypatch):
    def refused(*args, **kwargs):
        raise AssertionError("Goal015C synthetic fixtures prohibit real network transport")
    monkeypatch.setattr(socket.socket, "connect", refused)
    monkeypatch.setattr(socket.socket, "connect_ex", refused)
    monkeypatch.setattr(socket, "create_connection", refused)
    monkeypatch.setattr(socket, "getaddrinfo", refused)
