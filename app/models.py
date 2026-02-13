from __future__ import annotations
from sqlalchemy import String, Integer, Float, DateTime, Boolean, ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from datetime import datetime
from .db import Base

class Basket(Base):
    __tablename__ = "baskets"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, nullable=False, default="Mayfair Capital Basket")
    poll_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=60)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    holdings: Mapped[list["Holding"]] = relationship("Holding", back_populates="basket", cascade="all, delete-orphan")
    triggers: Mapped[list["TriggerLevel"]] = relationship("TriggerLevel", back_populates="basket", cascade="all, delete-orphan")

class Holding(Base):
    __tablename__ = "holdings"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    basket_id: Mapped[int] = mapped_column(ForeignKey("baskets.id"), nullable=False)
    ticker: Mapped[str] = mapped_column(String(20), nullable=False)
    weight: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    basket: Mapped["Basket"] = relationship("Basket", back_populates="holdings")

    __table_args__ = (UniqueConstraint("basket_id", "ticker", name="uq_basket_ticker"),)

class MonthReference(Base):
    __tablename__ = "month_reference"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    basket_id: Mapped[int] = mapped_column(ForeignKey("baskets.id"), nullable=False)
    month_key: Mapped[str] = mapped_column(String(7), nullable=False)  # "YYYY-MM"
    ticker: Mapped[str] = mapped_column(String(20), nullable=False)
    ref_price: Mapped[float] = mapped_column(Float, nullable=False)
    ref_time: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (UniqueConstraint("basket_id", "month_key", "ticker", name="uq_ref_month"),)

class TriggerLevel(Base):
    __tablename__ = "trigger_levels"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    basket_id: Mapped[int] = mapped_column(ForeignKey("baskets.id"), nullable=False)
    kind: Mapped[str] = mapped_column(String(20), nullable=False)  # "TP" or "SL"
    level_pct: Mapped[float] = mapped_column(Float, nullable=False) # e.g. 0.03, -0.05
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    basket: Mapped["Basket"] = relationship("Basket", back_populates="triggers")

class TriggerEvent(Base):
    __tablename__ = "trigger_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    basket_id: Mapped[int] = mapped_column(ForeignKey("baskets.id"), nullable=False)
    month_key: Mapped[str] = mapped_column(String(7), nullable=False)
    triggered_level_pct: Mapped[float] = mapped_column(Float, nullable=False)
    kind: Mapped[str] = mapped_column(String(20), nullable=False)  # "TP"/"SL"
    basket_change_pct: Mapped[float] = mapped_column(Float, nullable=False)
    event_time: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    note: Mapped[str] = mapped_column(String(300), nullable=False, default="")
