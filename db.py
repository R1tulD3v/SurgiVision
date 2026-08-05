"""Database engine, session factory, and declarative base (SQLAlchemy 2.0).

Uses ``config.DATABASE_URL`` — a local SQLite file by default, or a Postgres URL
in production (set ``SURGIVISION_DATABASE_URL``). The API depends on
``get_session`` so tests can override it with a throwaway database.
"""
from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

import config


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


def make_engine(url: str | None = None):
    url = url or config.DATABASE_URL
    # SQLite needs check_same_thread=False to be usable across FastAPI threads.
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args, future=True)


engine = make_engine()
SessionLocal = sessionmaker(
    bind=engine, autoflush=False, expire_on_commit=False, class_=Session
)


def init_db() -> None:
    """Create tables for all registered models (dev/first-run convenience).

    Production schema is owned by Alembic migrations; this is a safe no-op when
    the tables already exist.
    """
    import db_models  # noqa: F401  (import registers models on Base.metadata)

    Base.metadata.create_all(bind=engine)


def get_session() -> Iterator[Session]:
    """FastAPI dependency that yields a scoped session and always closes it."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
