"""
llm_agent.py

PROPRIETARY AND CONFIDENTIAL
Copyright (c) 2025 VDG Venkatesh. All Rights Reserved.

This software and associated documentation files are the proprietary 
and confidential information of VDG Venkatesh. Unauthorized copying,
modification, distribution, or use of this software, via any medium,
is strictly prohibited without express written permission.

NO LICENSE IS GRANTED. This code may not be used, reproduced, or 
incorporated into any other projects without explicit authorization.
For licensing inquiries, contact the copyright holder.

LLM Agent Layer for 2026 Macro Investing
- Natural language interface for portfolio queries
- Risk explanation and narrative generation
- Trade recommendation synthesis
- Automated report generation
"""
import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import openai


class LLMAgent:
    """
    LLM-powered agent for natural language interaction with risk mesh.
    Provides conversational interface, narrative generation, and insights.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.3,
        max_tokens: int = 1000
    ):
        """
        Initialize LLM agent.
        
        Parameters:
        -----------
        api_key : str, optional
            OpenAI API key (defaults to env var OPENAI_API_KEY)
        model : str
            Model name (gpt-4, gpt-3.5-turbo, etc.)
        temperature : float
            Sampling temperature (0.0 = deterministic, 1.0 = creative)
        max_tokens : int
            Maximum response tokens
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.conversation_history = []
        
        if self.api_key:
            openai.api_key = self.api_key
    
    def query_portfolio(
        self,
        question: str,
        portfolio_data: Dict[str, Any],
        risk_metrics: Dict[str, float]
    ) -> str:
        """
        Answer natural language questions about portfolio.
        
        Parameters:
        -----------
        question : str
            User's question (e.g., "What's my largest position?")
        portfolio_data : dict
            Portfolio holdings and weights
        risk_metrics : dict
            VaR, CVaR, Sharpe, etc.
        
        Returns:
        --------
        str : Natural language answer
        """
        context = f"""
Portfolio Summary:
{json.dumps(portfolio_data, indent=2)}

Risk Metrics:
{json.dumps(risk_metrics, indent=2)}

User Question: {question}

Provide a concise, professional answer suitable for a hedge fund investor.
        """
        
        response = self._call_llm(
            system_prompt="You are a senior portfolio manager at a global macro hedge fund. Answer questions about portfolio holdings, risk, and performance with precision.",
            user_prompt=context
        )
        
        return response
    
    def explain_risk(
        self,
        var: float,
        cvar: float,
        max_drawdown: float,
        volatility: float
    ) -> str:
        """
        Generate narrative explanation of risk metrics.
        
        Parameters:
        -----------
        var : float
            Value at Risk (95% confidence)
        cvar : float
            Conditional VaR
        max_drawdown : float
            Maximum historical drawdown
        volatility : float
            Portfolio volatility (annualized)
        
        Returns:
        --------
        str : Risk narrative
        """
        prompt = f"""
Generate a clear risk narrative for the following portfolio metrics:

- VaR (95%): {var*100:.2f}%
- CVaR (95%): {cvar*100:.2f}%
- Max Drawdown: {max_drawdown*100:.2f}%
- Volatility (annual): {volatility*100:.2f}%

Explain what these numbers mean in practical terms for a hedge fund LP. Keep it concise (3-4 sentences).
        """
        
        response = self._call_llm(
            system_prompt="You are a risk officer at a hedge fund. Explain risk metrics in clear, non-technical language.",
            user_prompt=prompt
        )
        
        return response
    
    def synthesize_trade_recommendation(
        self,
        macro_signals: Dict[str, float],
        regime: str,
        current_positions: Dict[str, float],
        market_context: str
    ) -> str:
        """
        Generate trade recommendation narrative.
        
        Parameters:
        -----------
        macro_signals : dict
            Signal strengths by asset class
        regime : str
            Current market regime (e.g., 'High Vol', 'Low Vol')
        current_positions : dict
            Current portfolio weights
        market_context : str
            Recent market developments
        
        Returns:
        --------
        str : Trade recommendation
        """
        prompt = f"""
Market Context:
{market_context}

Current Regime: {regime}

Macro Signals:
{json.dumps(macro_signals, indent=2)}

Current Positions:
{json.dumps(current_positions, indent=2)}

Provide a 2-3 paragraph trade recommendation memo. Include:
1. Market thesis
2. Suggested position adjustments
3. Risk considerations
        """
        
        response = self._call_llm(
            system_prompt="You are a senior global macro PM writing a trade recommendation for the investment committee.",
            user_prompt=prompt
        )
        
        return response
    
    def generate_daily_report(
        self,
        pnl: float,
        performance_metrics: Dict[str, float],
        top_contributors: List[tuple],
        risk_breaches: List[str],
        market_summary: str
    ) -> str:
        """
        Generate end-of-day summary report.
        
        Parameters:
        -----------
        pnl : float
            Daily P&L
        performance_metrics : dict
            Sharpe, return, vol, etc.
        top_contributors : list
            Top P&L contributors [(ticker, pnl), ...]
        risk_breaches : list
            Any risk limit breaches
        market_summary : str
            Market highlights
        
        Returns:
        --------
        str : Daily report
        """
        prompt = f"""
Generate a concise daily portfolio summary for {datetime.now().strftime('%Y-%m-%d')}:

P&L: ${pnl:,.2f}
Performance Metrics:
{json.dumps(performance_metrics, indent=2)}

Top Contributors:
{', '.join([f"{t}: ${p:,.0f}" for t, p in top_contributors])}

Risk Breaches: {', '.join(risk_breaches) if risk_breaches else 'None'}

Market Summary:
{market_summary}

Format as professional daily brief (3-4 paragraphs).
        """
        
        response = self._call_llm(
            system_prompt="You are a portfolio manager writing the daily summary for LPs and the investment team.",
            user_prompt=prompt
        )
        
        return response
    
    def explain_regime_shift(
        self,
        old_regime: str,
        new_regime: str,
        confidence: float,
        market_indicators: Dict[str, float]
    ) -> str:
        """
        Explain regime transition.
        
        Parameters:
        -----------
        old_regime : str
        new_regime : str
        confidence : float
            Model confidence in new regime
        market_indicators : dict
            Key indicators driving shift
        
        Returns:
        --------
        str : Regime shift explanation
        """
        prompt = f"""
The portfolio's regime detection model has identified a shift:

Old Regime: {old_regime}
New Regime: {new_regime}
Confidence: {confidence*100:.1f}%

Key Indicators:
{json.dumps(market_indicators, indent=2)}

Explain what this regime shift means and what portfolio adjustments might be warranted (2-3 paragraphs).
        """
        
        response = self._call_llm(
            system_prompt="You are a quantitative strategist explaining regime changes to the portfolio management team.",
            user_prompt=prompt
        )
        
        return response
    
    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> str:
        """
        Call OpenAI API.
        
        Parameters:
        -----------
        system_prompt : str
            System role/instructions
        user_prompt : str
            User query
        
        Returns:
        --------
        str : LLM response
        """
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            assistant_message = response['choices'][0]['message']['content']
            
            # Store in conversation history
            self.conversation_history.append({
                'timestamp': datetime.now().isoformat(),
                'user': user_prompt,
                'assistant': assistant_message
            })
            
            return assistant_message
            
        except Exception as e:
            return f"Error calling LLM: {str(e)}"
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Retrieve conversation history.
        
        Returns:
        --------
        list : Conversation log
        """
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []


# ========================================
# Demo
# ========================================
if __name__ == "__main__":
    agent = LLMAgent()
    
    # Mock data
    portfolio = {
        'SPY': 0.40,
        'TLT': 0.30,
        'GLD': 0.20,
        'DXY': 0.10
    }
    
    risk_metrics = {
        'VaR_95': 0.025,
        'CVaR_95': 0.035,
        'Sharpe': 1.45,
        'Max_Drawdown': -0.12
    }
    
    # Test query
    question = "What's my largest exposure and what's the tail risk?"
    answer = agent.query_portfolio(question, portfolio, risk_metrics)
    print("=" * 80)
    print("LLM Agent Demo")
    print("=" * 80)
    print(f"\nQuestion: {question}")
    print(f"\nAnswer:\n{answer}")
    print("\n" + "=" * 80)
  
