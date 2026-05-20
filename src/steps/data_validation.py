"""Data validation schemas using Pandera for pipeline data quality."""

from typing import Optional

import numpy as np
import pandera as pa
from pandera import Column, DataFrameModel, Check


class TrainingDataSchema(DataFrameModel):
    """Schema for training data validation."""

    # Feature columns - should be float values
    feature_0: Optional[Column[float, pa.Ge(-10), pa.Le(10)]] = None
    feature_1: Optional[Column[float, pa.Ge(-10), pa.Le(10)]] = None
    feature_2: Optional[Column[float, pa.Ge(-10), pa.Le(10)]] = None
    feature_3: Optional[Column[float, pa.Ge(-10), pa.Le(10)]] = None
    feature_4: Optional[Column[float, pa.Ge(-10), pa.Le(10)]] = None

    # Label column
    label: Column[int, pa.Check.isin([0, 1])]


class InferenceInputSchema(DataFrameModel):
    """Schema for inference input data validation."""

    feature_0: Column[float, pa.Ge(-10), pa.Le(10)]
    feature_1: Column[float, pa.Ge(-10), pa.Le(10)]
    feature_2: Column[float, pa.Ge(-10), pa.Le(10)]
    feature_3: Column[float, pa.Ge(-10), pa.Le(10)]
    feature_4: Column[float, pa.Ge(-10), pa.Le(10)]


class PredictionResultSchema(DataFrameModel):
    """Schema for prediction results."""

    score: Column[float, pa.Ge(0), pa.Le(1)]
    is_anomaly: Column[bool]
    threshold: Column[float, pa.Ge(0), pa.Le(1)]


def validate_training_data(df):
    """Validate training data against schema."""
    return TrainingDataSchema.validate(df)


def validate_inference_input(df):
    """Validate inference input against schema."""
    return InferenceInputSchema.validate(df)


def validate_prediction_result(df):
    """Validate prediction results."""
    return PredictionResultSchema.validate(df)