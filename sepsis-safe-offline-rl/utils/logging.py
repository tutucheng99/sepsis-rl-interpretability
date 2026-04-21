"""
Standardized logging utilities for the Sepsis Safe Offline RL project.

This module provides:
- Experiment tracking integration (MLflow/Weights & Biases)
- Structured logging for safety interventions
- Metric logging with proper formatting
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.

    Args:
        name: Logger name (typically __name__ of calling module)
        log_file: Optional path to log file
        level: Logging level (default: INFO)
        format_string: Optional custom format string

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class ExperimentTracker:
    """
    Wrapper for experiment tracking (MLflow, Weights & Biases, etc.)
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize experiment tracker.

        Args:
            config_path: Path to environment config YAML
        """
        self.logger = setup_logger(__name__)

        if config_path is not None and config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
            self.tracker_type = config.get("tools", {}).get(
                "experiment_tracker", {}
            ).get("type", "mlflow")
        else:
            self.tracker_type = "mlflow"
            self.logger.warning(
                f"Config not found at {config_path}, defaulting to MLflow"
            )

        self._init_tracker()

    def _init_tracker(self):
        """Initialize the experiment tracking backend."""
        if self.tracker_type == "mlflow":
            try:
                import mlflow
                self.backend = mlflow
                self.logger.info("MLflow experiment tracking initialized")
            except ImportError:
                self.logger.error("MLflow not installed")
                self.backend = None

        elif self.tracker_type == "wandb":
            try:
                import wandb
                self.backend = wandb
                self.logger.info("Weights & Biases experiment tracking initialized")
            except ImportError:
                self.logger.error("wandb not installed")
                self.backend = None

        else:
            self.logger.warning(f"Unknown tracker type: {self.tracker_type}")
            self.backend = None

    def log_params(self, params: Dict[str, Any]):
        """Log experiment parameters."""
        if self.backend is None:
            return

        if self.tracker_type == "mlflow":
            self.backend.log_params(params)
        elif self.tracker_type == "wandb":
            self.backend.config.update(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        if self.backend is None:
            return

        if self.tracker_type == "mlflow":
            for key, value in metrics.items():
                self.backend.log_metric(key, value, step=step)
        elif self.tracker_type == "wandb":
            self.backend.log(metrics, step=step)

    def log_artifact(self, local_path: Path):
        """Log an artifact (file)."""
        if self.backend is None:
            return

        if self.tracker_type == "mlflow":
            self.backend.log_artifact(str(local_path))
        elif self.tracker_type == "wandb":
            self.backend.save(str(local_path))


class SafetyLogger:
    """
    Specialized logger for safety-critical events and interventions.
    """

    def __init__(self, log_file: Path):
        """
        Initialize safety logger.

        Args:
            log_file: Path to safety intervention log file
        """
        self.logger = setup_logger(
            "safety",
            log_file=log_file,
            format_string="%(asctime)s - %(levelname)s - %(message)s"
        )

    def log_l1_intervention(
        self,
        state: Dict[str, Any],
        blocked_actions: list,
        reason: str
    ):
        """
        Log L1 (semantic safety) intervention.

        Args:
            state: Current state dictionary
            blocked_actions: List of actions blocked by L1
            reason: Human-readable reason for blocking
        """
        self.logger.warning(
            f"L1_INTERVENTION | State: {state} | "
            f"Blocked actions: {blocked_actions} | Reason: {reason}"
        )

    def log_l2_intervention(
        self,
        state: Dict[str, Any],
        ood_score: float,
        uncertainty: Dict[str, float],
        reason: str
    ):
        """
        Log L2 (cognitive safety) intervention.

        Args:
            state: Current state dictionary
            ood_score: OOD detection score
            uncertainty: Confounding uncertainty per action
            reason: Human-readable reason
        """
        self.logger.warning(
            f"L2_INTERVENTION | State: {state} | "
            f"OOD score: {ood_score:.4f} | Uncertainty: {uncertainty} | "
            f"Reason: {reason}"
        )

    def log_fallback(self, state: Dict[str, Any], fallback_action: int):
        """
        Log fallback policy invocation.

        Args:
            state: Current state
            fallback_action: Action taken by fallback policy
        """
        self.logger.warning(
            f"FALLBACK_INVOKED | State: {state} | "
            f"Fallback action: {fallback_action}"
        )

    def log_hard_violation(self, state: Dict[str, Any], action: int, rule: str):
        """
        Log hard constraint violation (CRITICAL).

        Args:
            state: Current state
            action: Violating action
            rule: Violated rule/constraint
        """
        self.logger.critical(
            f"HARD_VIOLATION | State: {state} | "
            f"Action: {action} | Rule: {rule}"
        )


# Convenience function for quick logger setup
def get_logger(name: str, log_dir: Path = Path("./logs")) -> logging.Logger:
    """
    Get a logger with standard configuration.

    Args:
        name: Logger name
        log_dir: Directory for log files

    Returns:
        Configured logger
    """
    log_file = log_dir / f"{name}.log"
    return setup_logger(name, log_file=log_file)
