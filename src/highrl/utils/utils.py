"""Utilties implementation for HighRL"""
import os


def get_device() -> str:
    """Get device from environment variables"""
    return os.environ["TRAIN_DEVICE"]
