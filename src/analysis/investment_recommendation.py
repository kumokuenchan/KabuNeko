"""
Investment Recommendation System

Combines technical analysis, fundamental metrics, and ML predictions
to provide actionable investment recommendations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from datetime import datetime


class InvestmentRecommendation:
    """
    Analyzes stocks and provides investment recommendations based on
    multiple factors including technical indicators, price trends,
    and risk metrics.
    """

    @staticmethod
    def calculate_technical_score(df: pd.DataFrame) -> Dict:
        """
        Calculate technical analysis score (0-100).

        Args:
            df: DataFrame with technical indicators

        Returns:
            Dict with score and signals
        """
        signals = {
            'trend': 0,      # -2 to +2
            'momentum': 0,   # -2 to +2
            'volatility': 0, # -2 to +2
            'volume': 0,     # -1 to +1
        }
        details = []

        if df.empty or len(df) < 50:
            return {'score': 50, 'signals': signals, 'details': ['Insufficient data']}

        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        # 1. TREND ANALYSIS
        if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
            sma_20 = latest['SMA_20']
            sma_50 = latest['SMA_50']
            close = latest['Close']

            if pd.notna(sma_20) and pd.notna(sma_50):
                # Price above both SMAs = bullish
                if close > sma_20 > sma_50:
                    signals['trend'] = 2
                    details.append("✅ Strong uptrend: Price > SMA20 > SMA50")
                elif close > sma_20:
                    signals['trend'] = 1
                    details.append("⬆️ Uptrend: Price above SMA20")
                elif close < sma_20 < sma_50:
                    signals['trend'] = -2
                    details.append("❌ Strong downtrend: Price < SMA20 < SMA50")
                elif close < sma_20:
                    signals['trend'] = -1
                    details.append("⬇️ Downtrend: Price below SMA20")

        # 2. MOMENTUM ANALYSIS
        if 'RSI' in df.columns:
            rsi = latest['RSI']
            if pd.notna(rsi):
                if rsi < 30:
                    signals['momentum'] = 2
                    details.append(f"✅ RSI oversold ({rsi:.1f}): Good buy opportunity")
                elif rsi < 50:
                    signals['momentum'] = 1
                    details.append(f"⬆️ RSI moderate ({rsi:.1f}): Bullish momentum")
                elif rsi > 70:
                    signals['momentum'] = -2
                    details.append(f"❌ RSI overbought ({rsi:.1f}): Overheated")
                elif rsi > 50:
                    signals['momentum'] = -1
                    details.append(f"⬇️ RSI high ({rsi:.1f}): Bearish momentum")

        # 3. MACD ANALYSIS
        if 'MACD' in df.columns and 'MACD_signal' in df.columns:
            macd = latest['MACD']
            signal = latest['MACD_signal']
            prev_macd = prev['MACD']
            prev_signal = prev['MACD_signal']

            if pd.notna(macd) and pd.notna(signal):
                # Bullish crossover
                if macd > signal and prev_macd <= prev_signal:
                    signals['momentum'] += 1
                    details.append("✅ MACD bullish crossover detected")
                # Bearish crossover
                elif macd < signal and prev_macd >= prev_signal:
                    signals['momentum'] -= 1
                    details.append("❌ MACD bearish crossover detected")

        # 4. VOLATILITY ANALYSIS
        if 'BB_position' in df.columns:
            bb_pos = latest['BB_position']
            if pd.notna(bb_pos):
                if bb_pos < 0.2:
                    signals['volatility'] = 2
                    details.append("✅ Near lower Bollinger Band: Potential bounce")
                elif bb_pos > 0.8:
                    signals['volatility'] = -2
                    details.append("❌ Near upper Bollinger Band: Potential pullback")

        # 5. VOLUME ANALYSIS
        if 'Volume_ratio' in df.columns:
            vol_ratio = latest['Volume_ratio']
            if pd.notna(vol_ratio) and vol_ratio > 1.5:
                # High volume confirms trend
                if signals['trend'] > 0:
                    signals['volume'] = 1
                    details.append(f"✅ High volume ({vol_ratio:.1f}x) confirms uptrend")
                else:
                    signals['volume'] = -1
                    details.append(f"⚠️ High volume ({vol_ratio:.1f}x) on downtrend")

        # Calculate total score (0-100)
        total_signal = sum(signals.values())
        # Range: -9 to +8, normalize to 0-100
        score = int(((total_signal + 9) / 17) * 100)

        return {
            'score': score,
            'signals': signals,
            'details': details
        }

    @staticmethod
    def calculate_risk_metrics(df: pd.DataFrame) -> Dict:
        """
        Calculate risk metrics for the stock.

        Args:
            df: DataFrame with price data

        Returns:
            Dict with risk metrics
        """
        if df.empty or len(df) < 20:
            return {
                'volatility': 0,
                'max_drawdown': 0,
                'risk_level': 'Unknown'
            }

        # Calculate daily returns
        returns = df['Close'].pct_change().dropna()

        # Annualized volatility
        volatility = returns.std() * np.sqrt(252) * 100

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min() * 100)

        # Risk level classification
        if volatility < 20:
            risk_level = 'Low'
        elif volatility < 40:
            risk_level = 'Medium'
        else:
            risk_level = 'High'

        return {
            'volatility': round(volatility, 2),
            'max_drawdown': round(max_drawdown, 2),
            'risk_level': risk_level
        }

    @staticmethod
    def calculate_entry_exit_points(df: pd.DataFrame) -> Dict:
        """
        Suggest entry and exit price points based on support/resistance.

        Args:
            df: DataFrame with price data

        Returns:
            Dict with entry/exit suggestions
        """
        if df.empty or len(df) < 20:
            return {
                'current_price': 0,
                'entry_price': 0,
                'stop_loss': 0,
                'target_price': 0
            }

        latest = df.iloc[-1]
        current_price = latest['Close']

        # Use recent 20-day data for support/resistance
        recent = df.tail(20)

        # Support: Recent low
        support = recent['Low'].min()

        # Resistance: Recent high
        resistance = recent['High'].max()

        # ATR for stop loss (if available)
        if 'ATR' in df.columns and pd.notna(latest['ATR']):
            atr = latest['ATR']
        else:
            atr = current_price * 0.02  # Default 2%

        # Entry price: Slightly above current (conservative)
        entry_price = current_price * 0.99  # 1% below current

        # Stop loss: 2x ATR below entry
        stop_loss = entry_price - (2 * atr)

        # Target price: Resistance or 2:1 risk/reward ratio
        risk = entry_price - stop_loss
        target_price = entry_price + (risk * 2)  # 2:1 reward/risk

        return {
            'current_price': round(current_price, 2),
            'entry_price': round(entry_price, 2),
            'stop_loss': round(stop_loss, 2),
            'target_price': round(target_price, 2),
            'support': round(support, 2),
            'resistance': round(resistance, 2)
        }

    @staticmethod
    def get_recommendation(df: pd.DataFrame,
                          ml_prediction: Optional[float] = None,
                          ml_confidence: Optional[float] = None) -> Dict:
        """
        Generate comprehensive investment recommendation.

        Args:
            df: DataFrame with all indicators
            ml_prediction: Optional ML predicted price
            ml_confidence: Optional ML confidence (R² score)

        Returns:
            Dict with complete recommendation
        """
        # Calculate components
        technical = InvestmentRecommendation.calculate_technical_score(df)
        risk = InvestmentRecommendation.calculate_risk_metrics(df)
        prices = InvestmentRecommendation.calculate_entry_exit_points(df)

        # Overall score starts with technical score
        overall_score = technical['score']

        # Adjust for ML prediction if available
        ml_signal = 0
        ml_details = []

        if ml_prediction is not None and ml_confidence is not None:
            current_price = prices['current_price']
            if current_price > 0:
                predicted_return = ((ml_prediction - current_price) / current_price) * 100

                # Only trust ML if confidence is reasonable
                if ml_confidence > 0.3:  # R² > 0.3
                    if predicted_return > 5:
                        ml_signal = 10
                        ml_details.append(f"✅ AI predicts +{predicted_return:.1f}% upside (confidence: {ml_confidence:.0%})")
                    elif predicted_return > 2:
                        ml_signal = 5
                        ml_details.append(f"⬆️ AI predicts +{predicted_return:.1f}% upside (confidence: {ml_confidence:.0%})")
                    elif predicted_return < -5:
                        ml_signal = -10
                        ml_details.append(f"❌ AI predicts {predicted_return:.1f}% downside (confidence: {ml_confidence:.0%})")
                    elif predicted_return < -2:
                        ml_signal = -5
                        ml_details.append(f"⬇️ AI predicts {predicted_return:.1f}% downside (confidence: {ml_confidence:.0%})")
                else:
                    ml_details.append(f"⚠️ Low AI confidence ({ml_confidence:.0%}): Prediction unreliable")

        # Adjust overall score with ML signal
        overall_score = min(100, max(0, overall_score + ml_signal))

        # Adjust for risk
        risk_adjustment = 0
        if risk['risk_level'] == 'High':
            risk_adjustment = -10
        elif risk['risk_level'] == 'Low':
            risk_adjustment = 5

        overall_score = min(100, max(0, overall_score + risk_adjustment))

        # Generate recommendation
        if overall_score >= 70:
            recommendation = 'STRONG BUY'
            color = '🟢'
            action = 'Consider buying. Multiple positive signals detected.'
        elif overall_score >= 55:
            recommendation = 'BUY'
            color = '🟢'
            action = 'Good buying opportunity. Positive signals present.'
        elif overall_score >= 45:
            recommendation = 'HOLD'
            color = '🟡'
            action = 'Wait for clearer signals before buying or selling.'
        elif overall_score >= 30:
            recommendation = 'SELL'
            color = '🔴'
            action = 'Consider selling. Negative signals detected.'
        else:
            recommendation = 'STRONG SELL'
            color = '🔴'
            action = 'Avoid or exit position. Multiple negative signals.'

        # Combine all details
        all_details = technical['details'] + ml_details

        # Add risk warning
        if risk['risk_level'] == 'High':
            all_details.append(f"⚠️ HIGH RISK: Volatility {risk['volatility']:.1f}%, Max Drawdown {risk['max_drawdown']:.1f}%")

        return {
            'recommendation': recommendation,
            'color': color,
            'action': action,
            'overall_score': overall_score,
            'technical_score': technical['score'],
            'risk_metrics': risk,
            'price_targets': prices,
            'details': all_details,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'disclaimer': '⚠️ This is an automated analysis for educational purposes only. Not financial advice. Always do your own research and consult a financial advisor before investing.'
        }


# Convenience function
def get_investment_recommendation(df: pd.DataFrame,
                                  ml_prediction: Optional[float] = None,
                                  ml_confidence: Optional[float] = None) -> Dict:
    """
    Quick function to get investment recommendation.

    Args:
        df: DataFrame with technical indicators
        ml_prediction: Optional ML predicted price
        ml_confidence: Optional ML confidence score

    Returns:
        Dict with recommendation
    """
    return InvestmentRecommendation.get_recommendation(df, ml_prediction, ml_confidence)


if __name__ == "__main__":
    # Example usage
    from src.data.fetcher import get_stock_data
    from src.models.feature_engineering import FeatureEngineer

    # Fetch and prepare data
    df = get_stock_data('AAPL', start='2024-01-01')
    df = FeatureEngineer.prepare_features(df)

    # Get recommendation
    rec = get_investment_recommendation(df)

    print(f"\n{rec['color']} {rec['recommendation']} (Score: {rec['overall_score']}/100)")
    print(f"\nAction: {rec['action']}")
    print(f"\nPrice Targets:")
    print(f"  Current: ${rec['price_targets']['current_price']}")
    print(f"  Entry: ${rec['price_targets']['entry_price']}")
    print(f"  Stop Loss: ${rec['price_targets']['stop_loss']}")
    print(f"  Target: ${rec['price_targets']['target_price']}")
    print(f"\nRisk Level: {rec['risk_metrics']['risk_level']}")
    print(f"  Volatility: {rec['risk_metrics']['volatility']:.1f}%")
    print(f"  Max Drawdown: {rec['risk_metrics']['max_drawdown']:.1f}%")
    print(f"\nSignals:")
    for detail in rec['details']:
        print(f"  {detail}")
    print(f"\n{rec['disclaimer']}")
