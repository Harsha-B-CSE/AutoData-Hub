"""Shared FastAPI and database fixtures for API integration tests."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import backend.main as backend_main
from backend.core import orchestrator
from backend.db import models as _models  # noqa: F401 - registers ORM tables
from backend.db.session import Base, get_db
from backend.engines.performance.cache import PipelineCache
from backend.main import app
from backend.services import export_service


SQLALCHEMY_TEST_DATABASE_URL = "sqlite:///:memory:"

test_engine = create_engine(
    SQLALCHEMY_TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=test_engine,
)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="function", autouse=True)
def setup_test_database(monkeypatch):
    """Create an isolated schema and bind every API/worker DB entrypoint to it."""
    Base.metadata.create_all(bind=test_engine)
    app.dependency_overrides[get_db] = override_get_db

    # BackgroundTasks and export helpers bypass Depends(get_db), so they must
    # use the same StaticPool-backed session factory as the request routes.
    monkeypatch.setattr(orchestrator, "SessionLocal", TestingSessionLocal)
    monkeypatch.setattr(export_service, "SessionLocal", TestingSessionLocal)
    monkeypatch.setattr(
        backend_main,
        "init_db",
        lambda: Base.metadata.create_all(bind=test_engine),
    )
    PipelineCache._cache_index.clear()

    yield

    app.dependency_overrides.clear()
    PipelineCache._cache_index.clear()
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def client(setup_test_database):
    """Provide a TestClient bound to the active in-memory database."""
    with TestClient(app) as test_client:
        yield test_client
