"""Pydantic request/response schemas for the /predict endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class WeldFeatures(BaseModel):
    """Process parameters for a single laser weld.

    All units match a pulsed Nd:YAG laser weld shop floor.
    """
    peak_power_W: float = Field(..., ge=100, le=5000, description="Peak laser power in watts")
    pulse_duration_ms: float = Field(..., ge=0.1, le=50, description="Pulse duration in milliseconds")
    spot_diameter_mm: float = Field(..., ge=0.1, le=2.0, description="Focused spot diameter in mm")
    travel_speed_mm_s: float = Field(..., ge=0.5, le=100, description="Head travel speed in mm/s")
    assist_gas_flow_lpm: float = Field(..., ge=0.0, le=50, description="Shielding/assist gas flow in L/min")

    def to_vector(self) -> list[float]:
        return [
            self.peak_power_W,
            self.pulse_duration_ms,
            self.spot_diameter_mm,
            self.travel_speed_mm_s,
            self.assist_gas_flow_lpm,
        ]


class WeldFeaturesBatch(BaseModel):
    items: list[WeldFeatures] = Field(..., min_length=1, max_length=1024)


class ClassProbability(BaseModel):
    label: str
    probability: float


class PredictionResponse(BaseModel):
    label: str
    probabilities: list[ClassProbability]
    model_version: str


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_version: str
    model_cv_accuracy: float
    uptime_seconds: float
