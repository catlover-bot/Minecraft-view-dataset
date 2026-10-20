"""No preparation fixture may use external sockets or DNS."""
import socket
import pytest


@pytest.fixture(autouse=True)
def deny_external_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("external network forbidden in Goal015D fixtures")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)
