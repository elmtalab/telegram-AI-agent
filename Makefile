.PHONY: install lint test run-chat run-router run-orchestrator run-dispatch run-simple-dispatcher

install:
pip install -e .[dev]

lint:
flake8

test:
pytest

run-chat:
uvicorn apps.chat_adapter.main:app --reload --port 8001

run-router:
uvicorn apps.router.main:app --reload --port 8002

run-orchestrator:
uvicorn apps.orchestrator.main:app --reload --port 8003

run-dispatch:
uvicorn apps.dispatch_controller.main:app --reload --port 8004

run-simple-dispatcher:
uvicorn apps.simple_dispatcher.main:app --reload --port 8005
