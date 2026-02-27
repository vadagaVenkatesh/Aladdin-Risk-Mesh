"""ML Model Registry Module

Central registry for ML models with metadata tracking, governance, and model risk management.
Aligned with regulatory expectations for AI/ML model inventories and lifecycle management.
"""

import json
import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import pandas as pd
import yaml


@dataclass
class ModelCard:
    """Model metadata card for governance and documentation."""
    
    model_id: str
    model_name: str
    model_type: str  # "panel_regression", "gradient_boosting", "ranking", etc.
    version: str
    description: str
    
    # Training metadata
    features: List[str]
    target: str
    training_start_date: str
    training_end_date: str
    training_samples: int
    
    # Performance metrics
    in_sample_r2: Optional[float] = None
    out_of_sample_r2: Optional[float] = None
    information_coefficient: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    
    # Regime coverage
    regime_coverage: Optional[Dict[str, float]] = None  # {regime: proportion}
    
    # Governance metadata
    validation_date: Optional[str] = None
    validation_status: str = "pending"  # "pending", "approved", "rejected", "monitoring"
    validator: Optional[str] = None
    next_review_date: Optional[str] = None
    
    # Risk classification
    risk_tier: str = "medium"  # "low", "medium", "high", "critical"
    explainability_score: Optional[float] = None  # 0-1 scale
    
    # Operational metadata
    created_at: str = datetime.datetime.now().isoformat()
    updated_at: str = datetime.datetime.now().isoformat()
    deployed: bool = False
    deployment_date: Optional[str] = None
    
    # Documentation
    documentation_url: Optional[str] = None
    code_repository: Optional[str] = None
    contact_owner: Optional[str] = None


class ModelRegistry:
    """Central registry for ML models with governance capabilities."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize model registry.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.governance_config = self.config.get('model_governance', {})
        self.models: Dict[str, ModelCard] = {}
        self.performance_history: Dict[str, List[Dict]] = {}  # model_id -> performance snapshots
        
    def register_model(self, model_card: ModelCard) -> str:
        """Register a new model or update existing model.
        
        Args:
            model_card: ModelCard with model metadata
            
        Returns:
            model_id of registered model
        """
        model_card.updated_at = datetime.datetime.now().isoformat()
        
        # Auto-generate ID if not provided
        if not model_card.model_id:
            model_card.model_id = f"{model_card.model_name}_{model_card.version}"
        
        # Check if update
        if model_card.model_id in self.models:
            print(f"Updating existing model: {model_card.model_id}")
        else:
            print(f"Registering new model: {model_card.model_id}")
        
        self.models[model_card.model_id] = model_card
        
        # Initialize performance history if new
        if model_card.model_id not in self.performance_history:
            self.performance_history[model_card.model_id] = []
        
        return model_card.model_id
    
    def get_model(self, model_id: str) -> Optional[ModelCard]:
        """Retrieve model card by ID.
        
        Args:
            model_id: Model identifier
            
        Returns:
            ModelCard or None if not found
        """
        return self.models.get(model_id)
    
    def list_models(self, 
                   status: Optional[str] = None,
                   risk_tier: Optional[str] = None,
                   deployed: Optional[bool] = None) -> List[ModelCard]:
        """List models with optional filters.
        
        Args:
            status: Filter by validation status
            risk_tier: Filter by risk tier
            deployed: Filter by deployment status
            
        Returns:
            List of ModelCards matching filters
        """
        results = []
        
        for model in self.models.values():
            if status and model.validation_status != status:
                continue
            if risk_tier and model.risk_tier != risk_tier:
                continue
            if deployed is not None and model.deployed != deployed:
                continue
            
            results.append(model)
        
        return results
    
    def update_performance(self,
                          model_id: str,
                          metrics: Dict[str, float],
                          observation_date: Optional[str] = None) -> None:
        """Record performance metrics snapshot.
        
        Args:
            model_id: Model identifier
            metrics: Performance metrics dictionary
            observation_date: Date of observation (defaults to now)
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found in registry")
        
        if observation_date is None:
            observation_date = datetime.datetime.now().isoformat()
        
        snapshot = {
            'date': observation_date,
            'metrics': metrics
        }
        
        self.performance_history[model_id].append(snapshot)
        
        # Update model card with latest metrics
        model = self.models[model_id]
        if 'r2' in metrics:
            model.out_of_sample_r2 = metrics['r2']
        if 'ic' in metrics:
            model.information_coefficient = metrics['ic']
        if 'sharpe' in metrics:
            model.sharpe_ratio = metrics['sharpe']
        
        model.updated_at = observation_date
    
    def get_performance_history(self, model_id: str) -> pd.DataFrame:
        """Get performance time series for a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            DataFrame with performance metrics over time
        """
        if model_id not in self.performance_history:
            return pd.DataFrame()
        
        history = self.performance_history[model_id]
        
        records = []
        for snapshot in history:
            record = {'date': snapshot['date']}
            record.update(snapshot['metrics'])
            records.append(record)
        
        df = pd.DataFrame(records)
        if not df.empty and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
        
        return df
    
    def check_regime_coverage(self,
                             model_id: str,
                             current_regime: str,
                             ood_threshold: Optional[float] = None) -> Dict[str, Any]:
        """Check if model is out-of-distribution for current regime.
        
        Args:
            model_id: Model identifier
            current_regime: Current detected regime
            ood_threshold: Out-of-distribution threshold (default from config)
            
        Returns:
            Dictionary with coverage check results
        """
        model = self.get_model(model_id)
        if not model or not model.regime_coverage:
            return {
                'model_id': model_id,
                'current_regime': current_regime,
                'is_ood': True,
                'coverage': 0.0,
                'warning': 'No regime coverage data available'
            }
        
        if ood_threshold is None:
            ood_threshold = self.governance_config.get('regime_ood_threshold', 0.30)
        
        coverage = model.regime_coverage.get(current_regime, 0.0)
        is_ood = coverage < ood_threshold
        
        return {
            'model_id': model_id,
            'current_regime': current_regime,
            'is_ood': is_ood,
            'coverage': coverage,
            'threshold': ood_threshold,
            'action': 'downweight' if is_ood else 'normal'
        }
    
    def flag_models_for_review(self) -> List[Dict[str, Any]]:
        """Identify models requiring review or attention.
        
        Returns:
            List of models with flags and reasons
        """
        flagged = []
        current_date = datetime.datetime.now()
        
        for model_id, model in self.models.items():
            flags = []
            
            # Check validation status
            if model.validation_status == "pending":
                flags.append("pending_validation")
            
            # Check review date
            if model.next_review_date:
                review_date = datetime.datetime.fromisoformat(model.next_review_date)
                if current_date >= review_date:
                    flags.append("review_overdue")
            
            # Check deployment without validation
            if model.deployed and model.validation_status != "approved":
                flags.append("deployed_without_approval")
            
            # Check performance degradation (if history available)
            if model_id in self.performance_history and len(self.performance_history[model_id]) >= 2:
                recent = self.performance_history[model_id][-1]['metrics']
                baseline = self.performance_history[model_id][0]['metrics']
                
                # Check for significant performance drop
                if 'sharpe' in recent and 'sharpe' in baseline:
                    if baseline['sharpe'] > 0 and recent['sharpe'] < baseline['sharpe'] * 0.7:
                        flags.append("performance_degradation")
            
            if flags:
                flagged.append({
                    'model_id': model_id,
                    'model_name': model.model_name,
                    'flags': flags,
                    'risk_tier': model.risk_tier
                })
        
        return flagged
    
    def export_inventory(self, filepath: str = "model_inventory.json") -> None:
        """Export complete model inventory to JSON.
        
        Args:
            filepath: Output file path
        """
        inventory = {
            'export_date': datetime.datetime.now().isoformat(),
            'total_models': len(self.models),
            'models': [asdict(model) for model in self.models.values()]
        }
        
        with open(filepath, 'w') as f:
            json.dump(inventory, f, indent=2)
        
        print(f"Model inventory exported to {filepath}")
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate summary statistics for governance reporting.
        
        Returns:
            Dictionary with summary metrics
        """
        total = len(self.models)
        deployed = sum(1 for m in self.models.values() if m.deployed)
        
        by_status = {}
        by_tier = {}
        
        for model in self.models.values():
            by_status[model.validation_status] = by_status.get(model.validation_status, 0) + 1
            by_tier[model.risk_tier] = by_tier.get(model.risk_tier, 0) + 1
        
        flagged = self.flag_models_for_review()
        
        return {
            'total_models': total,
            'deployed_models': deployed,
            'models_by_status': by_status,
            'models_by_risk_tier': by_tier,
            'flagged_for_review': len(flagged),
            'flagged_details': flagged
        }


if __name__ == "__main__":
    # Example usage
    registry = ModelRegistry()
    
    # Register a model
    model_card = ModelCard(
        model_id="",
        model_name="macro_panel_regression",
        model_type="panel_regression",
        version="v1.0",
        description="Elastic Net panel regression for macro signal pooling",
        features=["momentum_12m", "carry", "value_zscore", "growth_surprise"],
        target="forward_1m_return",
        training_start_date="2015-01-01",
        training_end_date="2023-12-31",
        training_samples=2520,
        in_sample_r2=0.18,
        out_of_sample_r2=0.12,
        information_coefficient=0.08,
        regime_coverage={
            "goldilocks": 0.35,
            "inflationary_growth": 0.40,
            "stagflation": 0.15,
            "recession": 0.10
        },
        risk_tier="medium",
        contact_owner="VDG Venkatesh"
    )
    
    model_id = registry.register_model(model_card)
    print(f"\nRegistered model: {model_id}")
    
    # Update performance
    registry.update_performance(
        model_id,
        metrics={'r2': 0.10, 'ic': 0.07, 'sharpe': 1.2},
        observation_date="2026-02-01"
    )
    
    # Check regime coverage
    coverage_check = registry.check_regime_coverage(model_id, "inflationary_growth")
    print(f"\nRegime Coverage Check: {coverage_check}")
    
    # Generate summary
    summary = registry.generate_summary_report()
    print(f"\nRegistry Summary:")
    print(json.dumps(summary, indent=2))
