"""
ops/ — MLOps, Monitoring & Deployment Operations

Handles model serving, drift detection, performance monitoring,
Azure deployment workflows, and operational health checks for
all Aladdin-Risk-Mesh production pipelines.
"""

from .monitor import DriftMonitor
from .deployer import AzureDeployer

__all__ = ["DriftMonitor", "AzureDeployer"]
