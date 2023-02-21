"""Setting up pre-call methods and library version"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Shutting up tensorflow warnings
__version__ = "0.1.6"
