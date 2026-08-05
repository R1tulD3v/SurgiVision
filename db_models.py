"""SQLAlchemy ORM models."""
from __future__ import annotations

import datetime as dt

from sqlalchemy import Boolean, DateTime, Float, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from db import Base


class AnalysisRecord(Base):
    """One persisted anomaly-detection run (audit trail + history)."""

    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    pipeline: Mapped[str] = mapped_column(String(64), nullable=False)
    reconstruction_error: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    is_anomaly: Mapped[bool] = mapped_column(Boolean, nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
