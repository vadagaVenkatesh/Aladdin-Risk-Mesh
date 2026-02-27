"""SA-CCR Exposure Proxy Module

Simplified SA-CCR (Standardized Approach for Counterparty Credit Risk) implementation
for derivative exposure at default (EAD) calculation aligned with Basel IV.

EAD = alpha * (RC + PFE)
where:
- RC = Replacement Cost (current mark-to-market exposure)
- PFE = Potential Future Exposure (notional * supervisory factor * maturity scalar)
- alpha = 1.4 (regulatory multiplier)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import yaml


class SACCRExposureProxy:
    """Simplified SA-CCR exposure calculator for derivatives positions."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize SA-CCR proxy with configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.alpha = self.config['risk_parameters']['sa_ccr_alpha']
        self.asset_class_params = self.config['asset_classes']
        
    def calculate_replacement_cost(self, 
                                   mtm_values: pd.Series,
                                   counterparty: str = "default") -> float:
        """Calculate Replacement Cost (RC) component.
        
        RC = max(V - C, 0)
        where V = mark-to-market value, C = collateral
        
        For simplified implementation, assumes no collateral.
        
        Args:
            mtm_values: Series of mark-to-market values by instrument
            counterparty: Counterparty identifier
            
        Returns:
            Replacement cost (non-negative)
        """
        # Net positive exposure across all derivatives with counterparty
        net_mtm = mtm_values.sum()
        rc = max(net_mtm, 0)
        return rc
    
    def calculate_pfe(self,
                     notionals: pd.Series,
                     asset_classes: pd.Series,
                     maturities_years: pd.Series) -> float:
        """Calculate Potential Future Exposure (PFE) component.
        
        PFE = sum over hedging sets of (supervisory_factor * notional * maturity_factor)
        
        Args:
            notionals: Series of notional amounts by instrument
            asset_classes: Series of asset class labels (equity, rates, fx, etc.)
            maturities_years: Series of maturity in years
            
        Returns:
            Total potential future exposure
        """
        pfe_total = 0.0
        
        for idx in notionals.index:
            notional = abs(notionals[idx])
            asset_class = asset_classes[idx]
            maturity = maturities_years[idx]
            
            # Get supervisory factor from config
            if asset_class in self.asset_class_params:
                supervisory_factor = self.asset_class_params[asset_class]['supervisory_factor']
            else:
                supervisory_factor = 0.15  # Default fallback
            
            # Maturity factor: min(M, 1 year) / 1 year
            # Simplified: sqrt(min(M, 1)) for better approximation
            maturity_factor = np.sqrt(min(maturity, 1.0))
            
            pfe_instrument = supervisory_factor * notional * maturity_factor
            pfe_total += pfe_instrument
            
        return pfe_total
    
    def calculate_ead(self,
                     mtm_values: pd.Series,
                     notionals: pd.Series,
                     asset_classes: pd.Series,
                     maturities_years: pd.Series,
                     counterparty: str = "default") -> Dict[str, float]:
        """Calculate Exposure at Default (EAD) using SA-CCR formula.
        
        Args:
            mtm_values: Mark-to-market values
            notionals: Notional amounts
            asset_classes: Asset class classifications
            maturities_years: Maturities in years
            counterparty: Counterparty identifier
            
        Returns:
            Dictionary with RC, PFE, EAD components
        """
        rc = self.calculate_replacement_cost(mtm_values, counterparty)
        pfe = self.calculate_pfe(notionals, asset_classes, maturities_years)
        ead = self.alpha * (rc + pfe)
        
        return {
            'rc': rc,
            'pfe': pfe,
            'ead': ead,
            'alpha': self.alpha,
            'counterparty': counterparty
        }
    
    def aggregate_ead_by_asset_class(self,
                                    mtm_values: pd.Series,
                                    notionals: pd.Series,
                                    asset_classes: pd.Series,
                                    maturities_years: pd.Series) -> pd.DataFrame:
        """Calculate EAD metrics aggregated by asset class.
        
        Args:
            mtm_values: Mark-to-market values
            notionals: Notional amounts
            asset_classes: Asset class classifications
            maturities_years: Maturities in years
            
        Returns:
            DataFrame with EAD breakdown by asset class
        """
        results = []
        
        for asset_class in asset_classes.unique():
            mask = asset_classes == asset_class
            
            ac_mtm = mtm_values[mask]
            ac_notional = notionals[mask]
            ac_maturity = maturities_years[mask]
            ac_class = asset_classes[mask]
            
            ead_result = self.calculate_ead(ac_mtm, ac_notional, ac_class, ac_maturity, 
                                           counterparty=asset_class)
            
            results.append({
                'asset_class': asset_class,
                'total_notional': ac_notional.abs().sum(),
                'net_mtm': ac_mtm.sum(),
                'rc': ead_result['rc'],
                'pfe': ead_result['pfe'],
                'ead': ead_result['ead']
            })
            
        return pd.DataFrame(results)
    
    def calculate_capital_requirement(self,
                                     ead: float,
                                     risk_weight: float = 0.75,
                                     capital_ratio: float = 0.08) -> float:
        """Calculate regulatory capital requirement from EAD.
        
        Capital = EAD * Risk_Weight * Capital_Ratio
        
        Args:
            ead: Exposure at default
            risk_weight: Regulatory risk weight (default 75% for corporates)
            capital_ratio: Minimum capital ratio (default 8%)
            
        Returns:
            Required capital amount
        """
        return ead * risk_weight * capital_ratio


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Mock derivative portfolio
    data = {
        'instrument': ['FX_Forward_1', 'IRS_1', 'Equity_Option_1', 'Commodity_Future_1'],
        'mtm': [50000, -20000, 30000, 10000],  # Mark-to-market values
        'notional': [1000000, 5000000, 500000, 200000],
        'asset_class': ['fx', 'rates', 'equities', 'commodities'],
        'maturity_years': [0.25, 5.0, 0.5, 1.0]
    }
    
    df = pd.DataFrame(data)
    
    proxy = SACCRExposureProxy()
    
    # Calculate total EAD
    ead_result = proxy.calculate_ead(
        mtm_values=df['mtm'],
        notionals=df['notional'],
        asset_classes=df['asset_class'],
        maturities_years=df['maturity_years']
    )
    
    print("=== SA-CCR Exposure Calculation ===")
    print(f"Replacement Cost (RC): ${ead_result['rc']:,.2f}")
    print(f"Potential Future Exposure (PFE): ${ead_result['pfe']:,.2f}")
    print(f"Alpha: {ead_result['alpha']}")
    print(f"Exposure at Default (EAD): ${ead_result['ead']:,.2f}")
    
    # Calculate capital requirement
    capital_req = proxy.calculate_capital_requirement(ead_result['ead'])
    print(f"\nRegulatory Capital Requirement: ${capital_req:,.2f}")
    
    # Aggregate by asset class
    print("\n=== EAD by Asset Class ===")
    by_asset_class = proxy.aggregate_ead_by_asset_class(
        df['mtm'], df['notional'], df['asset_class'], df['maturity_years']
    )
    print(by_asset_class.to_string(index=False))
