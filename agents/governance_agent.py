"""Governance Agent Module

Agentic layer for AI/ML model risk management and governance aligned with
regulatory expectations for model risk management frameworks.

Capabilities:
- Model inventory monitoring
- Performance drift detection
- Out-of-regime detection
- Data freshness checks
- Capital utilization monitoring
- Automated governance reporting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import json
import yaml
from datetime import datetime, timedelta


class GovernanceAgent:
    """AI/ML Governance agent for model risk management and oversight."""
    
    def __init__(self, 
                 model_registry,
                 config_path: str = "config/settings.yaml"):
        """Initialize governance agent.
        
        Args:
            model_registry: ModelRegistry instance
            config_path: Path to configuration file
        """
        self.model_registry = model_registry
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.governance_config = self.config.get('model_governance', {})
        self.capital_config = self.config.get('capital_optimization', {})
        
    def scan_model_inventory(self) -> Dict[str, Any]:
        """Scan model registry and identify issues requiring attention.
        
        Returns:
            Comprehensive inventory scan results
        """
        print("\n=== Model Inventory Scan ===")
        
        # Get all models
        all_models = self.model_registry.list_models()
        
        # Get flagged models
        flagged_models = self.model_registry.flag_models_for_review()
        
        # Categorize by risk tier
        high_risk_models = self.model_registry.list_models(risk_tier="high")
        critical_models = self.model_registry.list_models(risk_tier="critical")
        
        # Deployment status
        deployed_models = self.model_registry.list_models(deployed=True)
        pending_validation = self.model_registry.list_models(status="pending")
        
        results = {
            'scan_date': datetime.now().isoformat(),
            'total_models': len(all_models),
            'deployed_models': len(deployed_models),
            'high_risk_models': len(high_risk_models),
            'critical_models': len(critical_models),
            'pending_validation': len(pending_validation),
            'flagged_for_review': len(flagged_models),
            'flagged_details': flagged_models,
            'status': 'attention_required' if flagged_models else 'healthy'
        }
        
        return results
    
    def check_performance_drift(self,
                               model_id: str,
                               lookback_days: int = 90,
                               drift_threshold: float = 0.20) -> Dict[str, Any]:
        """Check for performance degradation over time.
        
        Args:
            model_id: Model identifier
            lookback_days: Lookback period for comparison
            drift_threshold: Threshold for significant drift (20% default)
            
        Returns:
            Drift detection results
        """
        performance_history = self.model_registry.get_performance_history(model_id)
        
        if performance_history.empty:
            return {
                'model_id': model_id,
                'drift_detected': False,
                'warning': 'No performance history available'
            }
        
        # Get baseline (first 30 days) and recent performance
        baseline_period = performance_history.head(30)
        recent_period = performance_history.tail(30)
        
        if baseline_period.empty or recent_period.empty:
            return {
                'model_id': model_id,
                'drift_detected': False,
                'warning': 'Insufficient data for drift detection'
            }
        
        # Compare key metrics
        drift_results = {}
        metrics_to_check = ['sharpe', 'ic', 'r2']
        
        for metric in metrics_to_check:
            if metric in baseline_period.columns and metric in recent_period.columns:
                baseline_value = baseline_period[metric].mean()
                recent_value = recent_period[metric].mean()
                
                if baseline_value != 0:
                    pct_change = (recent_value - baseline_value) / abs(baseline_value)
                    drift_results[metric] = {
                        'baseline': baseline_value,
                        'recent': recent_value,
                        'pct_change': pct_change,
                        'significant_drift': abs(pct_change) > drift_threshold
                    }
        
        # Overall drift status
        any_drift = any(m['significant_drift'] for m in drift_results.values() if isinstance(m, dict))
        
        return {
            'model_id': model_id,
            'drift_detected': any_drift,
            'drift_details': drift_results,
            'lookback_days': lookback_days,
            'threshold': drift_threshold,
            'recommendation': 'review_and_retrain' if any_drift else 'continue_monitoring'
        }
    
    def check_data_freshness(self,
                           data_sources: Dict[str, datetime],
                           max_staleness_hours: int = 24) -> Dict[str, Any]:
        """Check if data sources are fresh and up-to-date.
        
        Args:
            data_sources: Dictionary of source_name -> last_update_time
            max_staleness_hours: Maximum acceptable staleness
            
        Returns:
            Data freshness check results
        """
        current_time = datetime.now()
        max_staleness = timedelta(hours=max_staleness_hours)
        
        stale_sources = []
        fresh_sources = []
        
        for source, last_update in data_sources.items():
            staleness = current_time - last_update
            
            if staleness > max_staleness:
                stale_sources.append({
                    'source': source,
                    'last_update': last_update.isoformat(),
                    'staleness_hours': staleness.total_seconds() / 3600
                })
            else:
                fresh_sources.append(source)
        
        return {
            'check_time': current_time.isoformat(),
            'fresh_sources': fresh_sources,
            'stale_sources': stale_sources,
            'total_stale': len(stale_sources),
            'data_quality_status': 'degraded' if stale_sources else 'healthy',
            'max_staleness_hours': max_staleness_hours
        }
    
    def monitor_capital_utilization(self,
                                   current_capital_usage: float,
                                   capital_budget: Optional[float] = None) -> Dict[str, Any]:
        """Monitor capital utilization against budgets.
        
        Args:
            current_capital_usage: Current capital consumption
            capital_budget: Total capital budget (from config if None)
            
        Returns:
            Capital monitoring results
        """
        if capital_budget is None:
            capital_budget = self.capital_config.get('capital_budget_total', 1e6)
        
        utilization_pct = (current_capital_usage / capital_budget) * 100
        
        # Define thresholds
        warning_threshold = 80.0
        critical_threshold = 95.0
        
        if utilization_pct >= critical_threshold:
            status = 'critical'
            action = 'immediate_deleveraging_required'
        elif utilization_pct >= warning_threshold:
            status = 'warning'
            action = 'monitor_closely_consider_deleveraging'
        else:
            status = 'healthy'
            action = 'continue_normal_operations'
        
        return {
            'timestamp': datetime.now().isoformat(),
            'capital_usage': current_capital_usage,
            'capital_budget': capital_budget,
            'utilization_pct': utilization_pct,
            'status': status,
            'action': action,
            'headroom': capital_budget - current_capital_usage
        }
    
    def check_regime_alignment(self,
                              current_regime: str,
                              deployed_models: List[str]) -> Dict[str, Any]:
        """Check if deployed models are aligned with current regime.
        
        Args:
            current_regime: Current detected market regime
            deployed_models: List of deployed model IDs
            
        Returns:
            Regime alignment check results
        """
        ood_models = []
        aligned_models = []
        
        for model_id in deployed_models:
            coverage_check = self.model_registry.check_regime_coverage(model_id, current_regime)
            
            if coverage_check['is_ood']:
                ood_models.append({
                    'model_id': model_id,
                    'coverage': coverage_check['coverage'],
                    'action': coverage_check['action']
                })
            else:
                aligned_models.append(model_id)
        
        return {
            'current_regime': current_regime,
            'total_deployed_models': len(deployed_models),
            'aligned_models': len(aligned_models),
            'ood_models': len(ood_models),
            'ood_details': ood_models,
            'regime_alignment_status': 'healthy' if len(ood_models) == 0 else 'degraded'
        }
    
    def generate_governance_report(self,
                                  current_regime: str,
                                  capital_usage: float,
                                  data_sources: Optional[Dict[str, datetime]] = None) -> str:
        """Generate comprehensive governance and model risk report.
        
        Args:
            current_regime: Current market regime
            capital_usage: Current capital consumption
            data_sources: Data source freshness info (optional)
            
        Returns:
            Formatted governance report as markdown string
        """
        report_lines = []
        report_lines.append("# AI/ML Model Governance Report")
        report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"**Current Regime:** {current_regime}")
        report_lines.append("\n---\n")
        
        # Section 1: Model Inventory
        report_lines.append("## 1. Model Inventory Status")
        inventory_scan = self.scan_model_inventory()
        report_lines.append(f"- **Total Models:** {inventory_scan['total_models']}")
        report_lines.append(f"- **Deployed Models:** {inventory_scan['deployed_models']}")
        report_lines.append(f"- **High Risk Models:** {inventory_scan['high_risk_models']}")
        report_lines.append(f"- **Models Pending Validation:** {inventory_scan['pending_validation']}")
        report_lines.append(f"- **Models Flagged for Review:** {inventory_scan['flagged_for_review']}")
        
        if inventory_scan['flagged_details']:
            report_lines.append("\n**Flagged Models:**")
            for flagged in inventory_scan['flagged_details']:
                report_lines.append(f"  - {flagged['model_name']} ({flagged['model_id']}): {', '.join(flagged['flags'])}")
        
        report_lines.append("\n")
        
        # Section 2: Regime Alignment
        report_lines.append("## 2. Regime Alignment")
        deployed_model_ids = [m.model_id for m in self.model_registry.list_models(deployed=True)]
        regime_check = self.check_regime_alignment(current_regime, deployed_model_ids)
        report_lines.append(f"- **Regime-Aligned Models:** {regime_check['aligned_models']} / {regime_check['total_deployed_models']}")
        report_lines.append(f"- **Out-of-Distribution Models:** {regime_check['ood_models']}")
        
        if regime_check['ood_details']:
            report_lines.append("\n**OOD Model Actions:**")
            for ood in regime_check['ood_details']:
                report_lines.append(f"  - {ood['model_id']}: Coverage {ood['coverage']:.1%} → {ood['action']}")
        
        report_lines.append("\n")
        
        # Section 3: Capital Utilization
        report_lines.append("## 3. Capital Utilization")
        capital_monitor = self.monitor_capital_utilization(capital_usage)
        report_lines.append(f"- **Capital Usage:** ${capital_monitor['capital_usage']:,.2f}")
        report_lines.append(f"- **Capital Budget:** ${capital_monitor['capital_budget']:,.2f}")
        report_lines.append(f"- **Utilization:** {capital_monitor['utilization_pct']:.1f}%")
        report_lines.append(f"- **Status:** {capital_monitor['status'].upper()}")
        report_lines.append(f"- **Action:** {capital_monitor['action'].replace('_', ' ').title()}")
        report_lines.append("\n")
        
        # Section 4: Data Quality
        if data_sources:
            report_lines.append("## 4. Data Quality & Freshness")
            freshness_check = self.check_data_freshness(data_sources)
            report_lines.append(f"- **Fresh Sources:** {len(freshness_check['fresh_sources'])}")
            report_lines.append(f"- **Stale Sources:** {freshness_check['total_stale']}")
            report_lines.append(f"- **Data Quality Status:** {freshness_check['data_quality_status'].upper()}")
            
            if freshness_check['stale_sources']:
                report_lines.append("\n**Stale Data Sources:**")
                for stale in freshness_check['stale_sources']:
                    report_lines.append(f"  - {stale['source']}: {stale['staleness_hours']:.1f} hours stale")
            
            report_lines.append("\n")
        
        # Section 5: Recommendations
        report_lines.append("## 5. Governance Recommendations")
        
        recommendations = []
        
        if inventory_scan['flagged_for_review'] > 0:
            recommendations.append("Review and address flagged models requiring attention")
        
        if regime_check['ood_models'] > 0:
            recommendations.append(f"Downweight or suspend {regime_check['ood_models']} OOD models for current regime")
        
        if capital_monitor['status'] in ['warning', 'critical']:
            recommendations.append("Reduce leverage to bring capital utilization within limits")
        
        if data_sources and freshness_check['total_stale'] > 0:
            recommendations.append("Refresh stale data sources before next trading session")
        
        if inventory_scan['pending_validation'] > 0:
            recommendations.append(f"Complete validation for {inventory_scan['pending_validation']} pending models")
        
        if not recommendations:
            recommendations.append("No immediate actions required - continue monitoring")
        
        for i, rec in enumerate(recommendations, 1):
            report_lines.append(f"{i}. {rec}")
        
        report_lines.append("\n---\n")
        report_lines.append("*This report was generated automatically by the Governance Agent.*")
        report_lines.append("*Human review and approval required for all material model changes.*")
        
        return "\n".join(report_lines)
    
    def export_governance_snapshot(self,
                                  filepath: str = "governance_snapshot.json") -> None:
        """Export complete governance snapshot to JSON.
        
        Args:
            filepath: Output file path
        """
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'model_inventory': self.scan_model_inventory(),
            'registry_summary': self.model_registry.generate_summary_report()
        }
        
        with open(filepath, 'w') as f:
            json.dump(snapshot, f, indent=2)
        
        print(f"Governance snapshot exported to {filepath}")


if __name__ == "__main__":
    # Example usage
    from ml.model_registry import ModelRegistry, ModelCard
    
    # Initialize registry and agent
    registry = ModelRegistry()
    agent = GovernanceAgent(registry)
    
    # Register a sample model
    model = ModelCard(
        model_id="",
        model_name="test_model",
        model_type="panel_regression",
        version="v1.0",
        description="Test model for governance demo",
        features=["momentum", "carry"],
        target="returns",
        training_start_date="2020-01-01",
        training_end_date="2023-12-31",
        training_samples=1000,
        regime_coverage={'inflationary_growth': 0.45, 'goldilocks': 0.30},
        deployed=True
    )
    registry.register_model(model)
    
    # Generate governance report
    print("\n" + "="*60)
    report = agent.generate_governance_report(
        current_regime="inflationary_growth",
        capital_usage=850000,
        data_sources={
            'equity_prices': datetime.now() - timedelta(hours=2),
            'fx_rates': datetime.now() - timedelta(hours=1),
            'macro_data': datetime.now() - timedelta(hours=48)
        }
    )
    print(report)
    print("="*60)
