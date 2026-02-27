"""
ops/deployer.py — Azure Deployment Automation

Handles containerized deployment of Aladdin-Risk-Mesh services
to Azure Container Apps, Azure Kubernetes Service (AKS),
and Azure Machine Learning endpoints.

Copyright (c) Venkatesh Vadaga. All Rights Reserved.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AzureConfig:
    """Azure deployment configuration."""

    subscription_id: str = field(default_factory=lambda: os.getenv("AZURE_SUBSCRIPTION_ID", ""))
    resource_group: str = field(default_factory=lambda: os.getenv("AZURE_RESOURCE_GROUP", "aladdin-rg"))
    registry: str = field(default_factory=lambda: os.getenv("AZURE_REGISTRY", "aladdinregistry.azurecr.io"))
    aks_cluster: str = field(default_factory=lambda: os.getenv("AKS_CLUSTER", "aladdin-aks"))
    location: str = "eastus"
    image_name: str = "aladdin-risk-mesh"


class AzureDeployer:
    """
    Automates deployment of Aladdin-Risk-Mesh to Azure.

    Supports:
    - Azure Container Registry (ACR) image push
    - Azure Container Apps deployment
    - Azure Kubernetes Service (AKS) rollout
    - Azure ML endpoint registration
    """

    def __init__(self, config: Optional[AzureConfig] = None) -> None:
        self.config = config or AzureConfig()
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate that required environment variables are set."""
        if not self.config.subscription_id:
            logger.warning("AZURE_SUBSCRIPTION_ID not set; deployment will fail")

    def build_and_push(self, tag: Optional[str] = None) -> str:
        """
        Build Docker image and push to Azure Container Registry.

        Args:
            tag: Image tag (defaults to timestamp)

        Returns:
            Full image URI pushed to ACR
        """
        tag = tag or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        full_image = f"{self.config.registry}/{self.config.image_name}:{tag}"

        logger.info(f"Building image: {full_image}")
        self._run(["docker", "build", "-t", full_image, "."])

        logger.info(f"Pushing to ACR: {full_image}")
        self._run(["docker", "push", full_image])

        return full_image

    def deploy_container_app(
        self,
        image_uri: str,
        app_name: str = "aladdin-api",
        cpu: float = 1.0,
        memory: str = "2Gi",
        min_replicas: int = 1,
        max_replicas: int = 10,
    ) -> None:
        """
        Deploy to Azure Container Apps with autoscaling.

        Args:
            image_uri: Full ACR image URI
            app_name: Container app name
            cpu: CPU cores per instance
            memory: Memory per instance (e.g. '2Gi')
            min_replicas: Minimum replicas for scale-to-zero
            max_replicas: Maximum replicas under load
        """
        logger.info(f"Deploying {app_name} to Azure Container Apps")
        cmd = [
            "az", "containerapp", "update",
            "--name", app_name,
            "--resource-group", self.config.resource_group,
            "--image", image_uri,
            "--cpu", str(cpu),
            "--memory", memory,
            "--min-replicas", str(min_replicas),
            "--max-replicas", str(max_replicas),
        ]
        self._run(cmd)
        logger.info(f"Deployment complete: {app_name}")

    def deploy_aks(
        self,
        image_uri: str,
        namespace: str = "aladdin",
        deployment_name: str = "aladdin-api",
    ) -> None:
        """
        Rolling update deployment on AKS.

        Args:
            image_uri: Full ACR image URI
            namespace: Kubernetes namespace
            deployment_name: K8s deployment name
        """
        logger.info(f"Rolling update on AKS: {deployment_name}")
        # Get AKS credentials
        self._run([
            "az", "aks", "get-credentials",
            "--resource-group", self.config.resource_group,
            "--name", self.config.aks_cluster,
            "--overwrite-existing",
        ])
        # Update image
        self._run([
            "kubectl", "set", "image",
            f"deployment/{deployment_name}",
            f"api={image_uri}",
            "-n", namespace,
        ])
        # Wait for rollout
        self._run([
            "kubectl", "rollout", "status",
            f"deployment/{deployment_name}",
            "-n", namespace,
            "--timeout=300s",
        ])
        logger.info("AKS rollout complete")

    def _run(self, cmd: list[str]) -> subprocess.CompletedProcess:
        """Execute shell command with logging."""
        logger.debug(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if result.stdout:
            logger.info(result.stdout.strip())
        return result
