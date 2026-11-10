import pytest

import helpers  # noqa: F401  (puts the project root on sys.path)


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", help="include the 27-minute golden recording")
    parser.addoption("--llm", default="ollama", help="summary provider for golden summary tests ollama or claude")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long golden recordings, need --run-slow")
    config.addinivalue_line("markers", "llm: calls a language model (Ollama or Claude)")


def pytest_generate_tests(metafunc):
    if "golden" in metafunc.fixturenames:
        run_slow = metafunc.config.getoption("--run-slow")
        items = [i for i in helpers.golden_items() if run_slow or i["duration_sec"] < 600]
        metafunc.parametrize("golden", items, ids=[i["name"] for i in items])


@pytest.fixture(scope="session")
def provider(request):
    return request.config.getoption("--llm")
