"""FRTB-Aligned Risk Engine Module

Implements Basel IV FRTB (Fundamental Review of the Trading Book) concepts:
- Expected Shortfall (ES) at 97.5% confidence over 10-day horizon
- Modellable vs Non-Modellable Risk Factor (NMRF) classification
- Output floor: max(internal model, standardized * floor_ratio)
- Risk factor eligibility testing (simplified)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.stats import norm
import yaml


class FRTBRiskEngine:
    """FRTB-aligned market risk engine with ES, NMRF handling, and output floors."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize FRTB risk engine.
        
        Args:
            config_path: Path to configuration YAML
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.es_confidence = self.config['risk_parameters']['es_confidence_level']
        self.var_horizon_days = self.config['risk_parameters']['var_horizon_days']
        self.output_floor_ratio = self.config['risk_parameters']['output_floor_ratio']
        self.nmrf_addon = self.config['risk_parameters']['nmrf_capital_addon']
        self.asset_class_params = self.config['asset_classes']
        
    def calculate_es_historical(self, 
                               returns: pd.DataFrame, 
                               weights: np.ndarray,
                               horizon_days: int = None) -> Dict[str, float]:
        """Calculate Expected Shortfall (ES) using historical simulation.
        
        ES = average of losses beyond VaR threshold
        
        Args:
            returns: DataFrame of asset returns (1-day)
            weights: Portfolio weights
            horizon_days: Risk horizon in days (default from config)
            
        Returns:
            Dictionary with ES, VaR, and related metrics
        """
        if horizon_days is None:
            horizon_days = self.var_horizon_days
        
        # Calculate portfolio returns
        portfolio_returns = returns.dot(weights)
        
        # Scale to multi-day horizon (square root of time rule)
        scaling_factor = np.sqrt(horizon_days)
        scaled_returns = portfolio_returns * scaling_factor
        
        # Calculate VaR at ES confidence level
        var_threshold = np.percentile(scaled_returns, (1 - self.es_confidence) * 100)
        
        # ES = average of tail losses beyond VaR
        tail_losses = scaled_returns[scaled_returns <= var_threshold]
        es = -tail_losses.mean() if len(tail_losses) > 0 else 0.0
        
        return {
            'es_975': es,
            'var_975': -var_threshold,
            'horizon_days': horizon_days,
            'confidence_level': self.es_confidence,
            'tail_observations': len(tail_losses),
            'scaling_factor': scaling_factor
        }
    
    def calculate_es_parametric(self,
                               expected_returns: np.ndarray,
                               covariance: pd.DataFrame,
                               weights: np.ndarray,
                               horizon_days: int = None) -> Dict[str, float]:
        """Calculate parametric ES assuming normal distribution.
        
        ES = -mu + sigma * phi(Phi^-1(alpha)) / (1 - alpha)
        where phi is PDF and Phi is CDF of standard normal
        
        Args:
            expected_returns: Expected return vector
            covariance: Covariance matrix
            weights: Portfolio weights
            horizon_days: Risk horizon
            
        Returns:
            Dictionary with parametric ES and VaR
        """
        if horizon_days is None:
            horizon_days = self.var_horizon_days
        
        # Portfolio statistics
        port_return = weights.dot(expected_returns)
        port_variance = weights.T.dot(covariance).dot(weights)
        port_std = np.sqrt(port_variance)
        
        # Scale to horizon
        scaling_factor = np.sqrt(horizon_days)
        scaled_return = port_return * horizon_days
        scaled_std = port_std * scaling_factor
        
        # Parametric VaR and ES
        z_var = norm.ppf(1 - self.es_confidence)
        var_parametric = -(scaled_return + z_var * scaled_std)
        
        # ES formula for normal distribution
        phi_z = norm.pdf(z_var)
        es_parametric = -(scaled_return - scaled_std * phi_z / (1 - self.es_confidence))
        
        return {
            'es_975_parametric': es_parametric,
            'var_975_parametric': var_parametric,
            'port_std_daily': port_std,
            'port_std_horizon': scaled_std,
            'horizon_days': horizon_days
        }
    
    def classify_risk_factors(self,
                             returns: pd.DataFrame,
                             liquidity_threshold: int = 24) -> Dict[str, List[str]]:
        """Classify risk factors as modellable or non-modellable.
        
        FRTB modellability criteria (simplified):
        - At least 24 real price observations over 12 months
        - Price data not older than 1 year
        
        Args:
            returns: DataFrame of returns by risk factor
            liquidity_threshold: Minimum observations for modellability
            
        Returns:
            Dict with 'modellable' and 'non_modellable' factor lists
        """
        modellable = []
        non_modellable = []
        
        for factor in returns.columns:
            factor_data = returns[factor].dropna()
            
            # Check observation count
            n_obs = len(factor_data)
            
            # Check if factor is marked as modellable in config
            is_modellable_config = True
            for asset_class, params in self.asset_class_params.items():
                if asset_class in factor.lower():
                    is_modellable_config = params.get('modellable', True)
                    break
            
            if n_obs >= liquidity_threshold and is_modellable_config:
                modellable.append(factor)
            else:
                non_modellable.append(factor)
        
        return {
            'modellable': modellable,
            'non_modellable': non_modellable,
            'total_factors': len(returns.columns)
        }
    
    def calculate_nmrf_addon(self,
                            returns: pd.DataFrame,
                            weights: np.ndarray,
                            nmrf_factors: List[str]) -> float:
        """Calculate capital add-on for non-modellable risk factors.
        
        Simplified NMRF charge = stress_shock * notional_exposure
        
        Args:
            returns: Full returns DataFrame
            weights: Portfolio weights
            nmrf_factors: List of non-modellable factor names
            
        Returns:
            NMRF capital add-on amount
        """
        if not nmrf_factors:
            return 0.0
        
        # Identify indices of NMRF factors
        nmrf_indices = [i for i, col in enumerate(returns.columns) if col in nmrf_factors]
        
        if not nmrf_indices:
            return 0.0
        
        # Calculate exposure to NMRF factors
        nmrf_weights = np.array([weights[i] if i < len(weights) else 0.0 for i in nmrf_indices])
        nmrf_exposure = np.abs(nmrf_weights).sum()
        
        # Apply NMRF capital add-on rate
        nmrf_capital = nmrf_exposure * self.nmrf_addon
        
        return nmrf_capital
    
    def calculate_standardized_charge(self,
                                     positions: pd.Series,
                                     asset_classes: pd.Series) -> float:
        """Calculate simplified standardized market risk charge.
        
        Standardized charge = sum(abs(position) * risk_weight)
        
        Args:
            positions: Position values by instrument
            asset_classes: Asset class labels
            
        Returns:
            Total standardized capital charge
        """
        total_charge = 0.0
        
        for idx in positions.index:
            position = abs(positions[idx])
            asset_class = asset_classes[idx]
            
            # Get standardized risk weight from config
            if asset_class in self.asset_class_params:
                risk_weight = self.asset_class_params[asset_class]['standardized_rw']
            else:
                risk_weight = 0.20  # Default fallback
            
            total_charge += position * risk_weight
        
        return total_charge
    
    def apply_output_floor(self,
                          modelled_capital: float,
                          standardized_capital: float) -> Dict[str, float]:
        """Apply Basel IV output floor to internal model capital.
        
        Final Capital = max(modelled, standardized * floor_ratio)
        
        Args:
            modelled_capital: Capital from internal ES model
            standardized_capital: Capital from standardized approach
            
        Returns:
            Dictionary with floored capital and components
        """
        floor_capital = standardized_capital * self.output_floor_ratio
        final_capital = max(modelled_capital, floor_capital)
        
        binding = "floor" if final_capital == floor_capital else "model"
        
        return {
            'modelled_capital': modelled_capital,
            'standardized_capital': standardized_capital,
            'floor_capital': floor_capital,
            'final_capital': final_capital,
            'floor_ratio': self.output_floor_ratio,
            'binding_constraint': binding
        }
    
    def compute_frtb_capital(self,
                            returns: pd.DataFrame,
                            weights: np.ndarray,
                            positions: pd.Series,
                            asset_classes: pd.Series,
                            expected_returns: Optional[np.ndarray] = None,
                            covariance: Optional[pd.DataFrame] = None) -> Dict:
        """Comprehensive FRTB capital calculation with all components.
        
        Args:
            returns: Historical returns DataFrame
            weights: Portfolio weights
            positions: Position values
            asset_classes: Asset class labels
            expected_returns: Expected return vector (optional, for parametric)
            covariance: Covariance matrix (optional, for parametric)
            
        Returns:
            Complete FRTB capital breakdown
        """
        # 1. Calculate ES (internal model)
        es_hist = self.calculate_es_historical(returns, weights)
        
        # 2. Classify risk factors
        rf_classification = self.classify_risk_factors(returns)
        
        # 3. Calculate NMRF add-on
        nmrf_addon = self.calculate_nmrf_addon(
            returns, weights, rf_classification['non_modellable']
        )
        
        # 4. Total modelled capital = ES + NMRF
        modelled_capital = es_hist['es_975'] + nmrf_addon
        
        # 5. Calculate standardized charge
        standardized_capital = self.calculate_standardized_charge(positions, asset_classes)
        
        # 6. Apply output floor
        output_floor_result = self.apply_output_floor(modelled_capital, standardized_capital)
        
        # Compile complete result
        result = {
            'es_metrics': es_hist,
            'risk_factor_classification': rf_classification,
            'nmrf_addon': nmrf_addon,
            'modelled_capital': modelled_capital,
            'standardized_capital': standardized_capital,
            'output_floor': output_floor_result,
            'final_capital_requirement': output_floor_result['final_capital']
        }
        
        return result


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Mock multi-asset returns
    assets = ['US_Equity', 'EU_Rates', 'USD_JPY_FX', 'Gold_Commodity', 'Credit_IG']
    returns = pd.DataFrame(
        np.random.normal(0.0005, 0.015, (252, 5)),
        columns=assets
    )
    
    weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
    positions = pd.Series([1000000, 800000, 500000, 300000, 200000], index=assets)
    asset_classes = pd.Series(['equities', 'rates', 'fx', 'commodities', 'credit'], index=assets)
    
    engine = FRTBRiskEngine()
    
    # Calculate comprehensive FRTB capital
    frtb_result = engine.compute_frtb_capital(
        returns=returns,
        weights=weights,
        positions=positions,
        asset_classes=asset_classes
    )
    
    print("=== FRTB Capital Calculation ===")
    print(f"\n10-Day ES (97.5%): ${frtb_result['es_metrics']['es_975']:,.2f}")
    print(f"NMRF Add-on: ${frtb_result['nmrf_addon']:,.2f}")
    print(f"Modelled Capital: ${frtb_result['modelled_capital']:,.2f}")
    print(f"Standardized Capital: ${frtb_result['standardized_capital']:,.2f}")
    print(f"\nOutput Floor Applied: {frtb_result['output_floor']['binding_constraint']}")
    print(f"Final Capital Requirement: ${frtb_result['final_capital_requirement']:,.2f}")
    
    print(f"\nRisk Factor Classification:")
    print(f"  Modellable: {len(frtb_result['risk_factor_classification']['modellable'])}")
    print(f"  Non-Modellable: {len(frtb_result['risk_factor_classification']['non_modellable'])}")
