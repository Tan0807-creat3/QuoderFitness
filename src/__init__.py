"""QoderFitness - Multi-task AI + Nano LLM built from scratch."""

from .model import QoderFitness, Layer, TaskHead
from .data import generate_multitask_dataset, train_test_split
from .utils import load_scaler, save_scaler
from .train import train_model
from .infer import load_model_for_inference, predict_single, predict_batch
from .tokenizer import CharTokenizer
from .llm import NanoLLM

__all__ = [
    "QoderFitness", "Layer", "TaskHead",
    "generate_multitask_dataset", "train_test_split",
    "load_scaler", "save_scaler",
    "train_model",
    "load_model_for_inference", "predict_single", "predict_batch",
    "CharTokenizer", "NanoLLM",
]

__version__ = "2.1.0"
