"""离线与安全约束测试。

铁律：本项目是纸面交易/模拟撮合，**不得**包含任何真实交易所连接、网络请求或密钥。
这里用静态扫描 + 运行时 socket 拦截双重保证。
"""
from __future__ import annotations

import ast
import os
import socket
import sys

import kairos_crypto
from kairos_crypto import (
    DcaStrategy,
    Exchange,
    GridStrategy,
    MomentumStrategy,
    PaperExchange,
    PaperTradingEngine,
    SyntheticCandles,
)

PKG_DIR = os.path.dirname(os.path.abspath(kairos_crypto.__file__))
ROOT = os.path.dirname(PKG_DIR)
SCAN_DIRS = [PKG_DIR, os.path.join(ROOT, "examples"), os.path.join(ROOT, "tests")]

#: 一旦出现即视为「可能联网」的模块
FORBIDDEN_MODULES = {
    "socket", "ssl", "http", "urllib", "urllib3", "requests", "aiohttp", "httpx",
    "websocket", "websockets", "ccxt", "ftplib", "telnetlib", "xmlrpc", "socketserver",
    "subprocess", "asyncio", "binance", "okx", "coinbase", "kraken",
}

#: 源码里不应出现的字符串（密钥/端点等）
FORBIDDEN_SUBSTRINGS = (
    "api_key", "apikey", "api_secret", "secret_key", "password", "passphrase",
    "https://", "http://", "wss://", "ws://", "localhost", "127.0.0.1",
)

#: 包内允许的第三方依赖（SPEC：只用 numpy / pandas）
ALLOWED_THIRD_PARTY = {"numpy", "pandas"}

#: 包内允许的标准库模块
ALLOWED_STDLIB = {
    "__future__", "abc", "ast", "collections", "dataclasses", "datetime", "enum",
    "itertools", "math", "os", "sys", "typing",
}


#: 本文件自身需要 import socket 才能拦截它，因此从扫描名单中排除（并在下方显式说明）
SELF = os.path.abspath(__file__)


def python_files(dirs=None, skip_self=True):
    for directory in (dirs or SCAN_DIRS):
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            if not name.endswith(".py"):
                continue
            path = os.path.join(directory, name)
            if skip_self and os.path.abspath(path) == SELF:
                continue
            yield path


def imported_roots(path):
    """静态解析出文件导入的顶层模块名。"""
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    return roots


def test_no_network_modules_imported_anywhere():
    """除本文件（为拦截而 import socket）外，任何源码都不得导入网络相关模块。"""
    offenders = {}
    for path in python_files():
        bad = imported_roots(path) & FORBIDDEN_MODULES
        if bad:
            offenders[os.path.relpath(path, ROOT)] = sorted(bad)
    assert offenders == {}


def test_no_credentials_or_endpoints_in_sources():
    offenders = {}
    for path in python_files():
        with open(path, encoding="utf-8") as handle:
            text = handle.read().lower()
        hits = [token for token in FORBIDDEN_SUBSTRINGS if token in text]
        if hits:
            offenders[os.path.relpath(path, ROOT)] = hits
    assert offenders == {}


def test_package_only_depends_on_numpy_and_pandas():
    roots = set()
    for path in python_files([PKG_DIR]):
        roots |= imported_roots(path)
    roots -= {"kairos_crypto"}
    third_party = roots - ALLOWED_STDLIB
    assert third_party <= ALLOWED_THIRD_PARTY, f"意外的第三方依赖: {third_party}"


def test_no_network_libraries_loaded_after_import():
    assert kairos_crypto.__version__                 # 确保包已完整导入
    for name in ("requests", "ccxt", "aiohttp", "httpx", "urllib3", "websocket",
                 "websockets", "urllib.request", "http.client"):
        assert name not in sys.modules, f"{name} 不应被加载"


def test_exchange_has_no_connection_surface():
    """抽象接口只有行情/交易/账户方法，没有 url/session/密钥之类的连接面。"""
    leaked = [name for name in dir(Exchange)
              if any(token in name.lower()
                     for token in ("url", "host", "session", "socket", "http",
                                   "api_key", "secret", "token", "auth"))]
    assert leaked == []
    paper = PaperExchange(quote="USDT", initial_quote=1.0)
    assert paper.is_paper is True
    assert not any(isinstance(value, str) and value.startswith(("http://", "https://", "ws"))
                   for value in vars(paper).values())


def test_full_session_without_any_socket(monkeypatch):
    """运行时拦截：跑完三种策略的完整纸面交易，全程不得触碰 socket。"""

    def blocked(*args, **kwargs):
        raise AssertionError("纸面交易不得创建任何网络连接")

    for name in ("socket", "create_connection", "getaddrinfo", "gethostbyname",
                 "socketpair"):
        monkeypatch.setattr(socket, name, blocked, raising=False)

    candles = SyntheticCandles(n=300, seed=17, interval="1h").candles
    strategies = [
        MomentumStrategy(fast=10, slow=30),
        GridStrategy(levels=3, spacing=0.02, qty_per_grid=0.01),
        DcaStrategy(quote_per_buy=50.0, every_n=12),
    ]
    for strategy in strategies:
        exchange = PaperExchange(quote="USDT", initial_quote=10_000.0, fee_rate=0.001)
        result = PaperTradingEngine(candles, strategy, exchange).run()
        assert len(result.equity) == len(candles)
        assert result.check_accounting()
        assert result.equity.notna().all()


def test_version_and_exports():
    assert kairos_crypto.__version__ == "0.1.0"
    for name in kairos_crypto.__all__:
        assert hasattr(kairos_crypto, name), name
