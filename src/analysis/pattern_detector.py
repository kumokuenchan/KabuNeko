"""
Chart Pattern Detector

Automatically detect technical chart patterns including head & shoulders,
double tops/bottoms, triangles, flags, and other classic patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.signal import argrelextrema


class PatternDetector:
    """Detect technical chart patterns in price data"""

    @staticmethod
    def find_peaks_and_troughs(data: pd.Series, order: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find local peaks (highs) and troughs (lows) in price data

        Args:
            data: Price series
            order: How many points on each side to use for comparison

        Returns:
            Tuple of (peaks_indices, troughs_indices)
        """
        # Find local maxima (peaks)
        peaks = argrelextrema(data.values, np.greater, order=order)[0]

        # Find local minima (troughs)
        troughs = argrelextrema(data.values, np.less, order=order)[0]

        return peaks, troughs

    @classmethod
    def detect_head_and_shoulders(cls, df: pd.DataFrame, lookback: int = 50) -> Optional[Dict]:
        """
        Detect Head and Shoulders pattern (bearish reversal)

        Pattern: Left shoulder - Head - Right shoulder
        Head is higher than both shoulders
        """
        if len(df) < lookback:
            return None

        recent_data = df.tail(lookback).copy()
        prices = recent_data['Close']

        peaks, troughs = cls.find_peaks_and_troughs(prices, order=3)

        if len(peaks) < 3 or len(troughs) < 2:
            return None

        # Get last 3 peaks (potential shoulders and head)
        last_peaks = peaks[-3:]
        peak_prices = prices.iloc[last_peaks].values

        # Check if middle peak is highest (head)
        if peak_prices[1] > peak_prices[0] and peak_prices[1] > peak_prices[2]:
            # Check if shoulders are roughly equal (within 5%)
            shoulder_diff = abs(peak_prices[0] - peak_prices[2]) / peak_prices[0]

            if shoulder_diff < 0.05:
                # Calculate neckline (support level)
                neckline = prices.iloc[troughs[-2:].min()] if len(troughs) >= 2 else prices.min()

                current_price = prices.iloc[-1]
                head_price = peak_prices[1]

                # Pattern height (head to neckline)
                pattern_height = head_price - neckline

                # Target price: neckline - pattern_height
                target_price = neckline - pattern_height

                # Confidence based on pattern symmetry
                confidence = 100 - (shoulder_diff * 100)

                return {
                    'pattern': 'Head and Shoulders',
                    'type': 'Bearish Reversal',
                    'signal': 'SELL',
                    'confidence': min(confidence, 95),
                    'neckline': neckline,
                    'target_price': target_price,
                    'current_price': current_price,
                    'description': 'Head and Shoulders is a bearish reversal pattern. A breakdown below the neckline confirms the pattern.',
                    'emoji': '⬇️'
                }

        return None

    @classmethod
    def detect_inverse_head_and_shoulders(cls, df: pd.DataFrame, lookback: int = 50) -> Optional[Dict]:
        """
        Detect Inverse Head and Shoulders pattern (bullish reversal)

        Pattern: Left shoulder - Head - Right shoulder (inverted)
        Head is lower than both shoulders
        """
        if len(df) < lookback:
            return None

        recent_data = df.tail(lookback).copy()
        prices = recent_data['Close']

        peaks, troughs = cls.find_peaks_and_troughs(prices, order=3)

        if len(troughs) < 3 or len(peaks) < 2:
            return None

        # Get last 3 troughs (potential shoulders and head)
        last_troughs = troughs[-3:]
        trough_prices = prices.iloc[last_troughs].values

        # Check if middle trough is lowest (head)
        if trough_prices[1] < trough_prices[0] and trough_prices[1] < trough_prices[2]:
            # Check if shoulders are roughly equal (within 5%)
            shoulder_diff = abs(trough_prices[0] - trough_prices[2]) / trough_prices[0]

            if shoulder_diff < 0.05:
                # Calculate neckline (resistance level)
                neckline = prices.iloc[peaks[-2:].max()] if len(peaks) >= 2 else prices.max()

                current_price = prices.iloc[-1]
                head_price = trough_prices[1]

                # Pattern height (neckline to head)
                pattern_height = neckline - head_price

                # Target price: neckline + pattern_height
                target_price = neckline + pattern_height

                # Confidence based on pattern symmetry
                confidence = 100 - (shoulder_diff * 100)

                return {
                    'pattern': 'Inverse Head and Shoulders',
                    'type': 'Bullish Reversal',
                    'signal': 'BUY',
                    'confidence': min(confidence, 95),
                    'neckline': neckline,
                    'target_price': target_price,
                    'current_price': current_price,
                    'description': 'Inverse Head and Shoulders is a bullish reversal pattern. A breakout above the neckline confirms the pattern.',
                    'emoji': '⬆️'
                }

        return None

    @classmethod
    def detect_double_top(cls, df: pd.DataFrame, lookback: int = 50) -> Optional[Dict]:
        """
        Detect Double Top pattern (bearish reversal)

        Two peaks at similar price levels with a trough between them
        """
        if len(df) < lookback:
            return None

        recent_data = df.tail(lookback).copy()
        prices = recent_data['Close']

        peaks, troughs = cls.find_peaks_and_troughs(prices, order=5)

        if len(peaks) < 2 or len(troughs) < 1:
            return None

        # Get last 2 peaks
        last_peaks = peaks[-2:]
        peak_prices = prices.iloc[last_peaks].values

        # Check if peaks are at similar levels (within 3%)
        peak_diff = abs(peak_prices[0] - peak_prices[1]) / peak_prices[0]

        if peak_diff < 0.03:
            # Find trough between peaks
            trough_between = troughs[(troughs > last_peaks[0]) & (troughs < last_peaks[1])]

            if len(trough_between) > 0:
                support_level = prices.iloc[trough_between[0]]
                current_price = prices.iloc[-1]
                peak_avg = (peak_prices[0] + peak_prices[1]) / 2

                # Pattern height
                pattern_height = peak_avg - support_level

                # Target price
                target_price = support_level - pattern_height

                # Confidence
                confidence = 100 - (peak_diff * 100)

                return {
                    'pattern': 'Double Top',
                    'type': 'Bearish Reversal',
                    'signal': 'SELL',
                    'confidence': min(confidence, 95),
                    'resistance': peak_avg,
                    'support': support_level,
                    'target_price': target_price,
                    'current_price': current_price,
                    'description': 'Double Top is a bearish reversal pattern. A breakdown below support confirms the pattern.',
                    'emoji': '⬇️'
                }

        return None

    @classmethod
    def detect_double_bottom(cls, df: pd.DataFrame, lookback: int = 50) -> Optional[Dict]:
        """
        Detect Double Bottom pattern (bullish reversal)

        Two troughs at similar price levels with a peak between them
        """
        if len(df) < lookback:
            return None

        recent_data = df.tail(lookback).copy()
        prices = recent_data['Close']

        peaks, troughs = cls.find_peaks_and_troughs(prices, order=5)

        if len(troughs) < 2 or len(peaks) < 1:
            return None

        # Get last 2 troughs
        last_troughs = troughs[-2:]
        trough_prices = prices.iloc[last_troughs].values

        # Check if troughs are at similar levels (within 3%)
        trough_diff = abs(trough_prices[0] - trough_prices[1]) / trough_prices[0]

        if trough_diff < 0.03:
            # Find peak between troughs
            peak_between = peaks[(peaks > last_troughs[0]) & (peaks < last_troughs[1])]

            if len(peak_between) > 0:
                resistance_level = prices.iloc[peak_between[0]]
                current_price = prices.iloc[-1]
                trough_avg = (trough_prices[0] + trough_prices[1]) / 2

                # Pattern height
                pattern_height = resistance_level - trough_avg

                # Target price
                target_price = resistance_level + pattern_height

                # Confidence
                confidence = 100 - (trough_diff * 100)

                return {
                    'pattern': 'Double Bottom',
                    'type': 'Bullish Reversal',
                    'signal': 'BUY',
                    'confidence': min(confidence, 95),
                    'resistance': resistance_level,
                    'support': trough_avg,
                    'target_price': target_price,
                    'current_price': current_price,
                    'description': 'Double Bottom is a bullish reversal pattern. A breakout above resistance confirms the pattern.',
                    'emoji': '⬆️'
                }

        return None

    @classmethod
    def detect_ascending_triangle(cls, df: pd.DataFrame, lookback: int = 50) -> Optional[Dict]:
        """
        Detect Ascending Triangle (bullish continuation)

        Flat resistance line with rising support
        """
        if len(df) < lookback:
            return None

        recent_data = df.tail(lookback).copy()
        prices = recent_data['Close']

        peaks, troughs = cls.find_peaks_and_troughs(prices, order=4)

        if len(peaks) < 2 or len(troughs) < 2:
            return None

        # Get recent peaks and troughs
        recent_peaks = peaks[-3:] if len(peaks) >= 3 else peaks[-2:]
        recent_troughs = troughs[-3:] if len(troughs) >= 3 else troughs[-2:]

        peak_prices = prices.iloc[recent_peaks].values
        trough_prices = prices.iloc[recent_troughs].values

        # Check if peaks are flat (within 2%)
        peak_variation = (peak_prices.max() - peak_prices.min()) / peak_prices.mean()

        # Check if troughs are rising
        troughs_rising = all(trough_prices[i] < trough_prices[i+1] for i in range(len(trough_prices)-1))

        if peak_variation < 0.02 and troughs_rising:
            resistance = peak_prices.mean()
            current_price = prices.iloc[-1]

            # Pattern height (from lowest trough to resistance)
            pattern_height = resistance - trough_prices.min()

            # Target price
            target_price = resistance + pattern_height

            return {
                'pattern': 'Ascending Triangle',
                'type': 'Bullish Continuation',
                'signal': 'BUY',
                'confidence': 75,
                'resistance': resistance,
                'target_price': target_price,
                'current_price': current_price,
                'description': 'Ascending Triangle is a bullish continuation pattern. A breakout above resistance triggers an upward move.',
                'emoji': '⬆️'
            }

        return None

    @classmethod
    def detect_descending_triangle(cls, df: pd.DataFrame, lookback: int = 50) -> Optional[Dict]:
        """
        Detect Descending Triangle (bearish continuation)

        Flat support line with declining resistance
        """
        if len(df) < lookback:
            return None

        recent_data = df.tail(lookback).copy()
        prices = recent_data['Close']

        peaks, troughs = cls.find_peaks_and_troughs(prices, order=4)

        if len(peaks) < 2 or len(troughs) < 2:
            return None

        # Get recent peaks and troughs
        recent_peaks = peaks[-3:] if len(peaks) >= 3 else peaks[-2:]
        recent_troughs = troughs[-3:] if len(troughs) >= 3 else troughs[-2:]

        peak_prices = prices.iloc[recent_peaks].values
        trough_prices = prices.iloc[recent_troughs].values

        # Check if troughs are flat (within 2%)
        trough_variation = (trough_prices.max() - trough_prices.min()) / trough_prices.mean()

        # Check if peaks are declining
        peaks_declining = all(peak_prices[i] > peak_prices[i+1] for i in range(len(peak_prices)-1))

        if trough_variation < 0.02 and peaks_declining:
            support = trough_prices.mean()
            current_price = prices.iloc[-1]

            # Pattern height (from highest peak to support)
            pattern_height = peak_prices.max() - support

            # Target price
            target_price = support - pattern_height

            return {
                'pattern': 'Descending Triangle',
                'type': 'Bearish Continuation',
                'signal': 'SELL',
                'confidence': 75,
                'support': support,
                'target_price': target_price,
                'current_price': current_price,
                'description': 'Descending Triangle is a bearish continuation pattern. A breakdown below support triggers a downward move.',
                'emoji': '⬇️'
            }

        return None

    @classmethod
    def detect_all_patterns(cls, df: pd.DataFrame) -> List[Dict]:
        """
        Detect all patterns in the given data

        Args:
            df: DataFrame with OHLCV data

        Returns:
            List of detected patterns with details
        """
        patterns = []

        # Try all pattern detection methods
        detectors = [
            cls.detect_head_and_shoulders,
            cls.detect_inverse_head_and_shoulders,
            cls.detect_double_top,
            cls.detect_double_bottom,
            cls.detect_ascending_triangle,
            cls.detect_descending_triangle,
        ]

        for detector in detectors:
            try:
                pattern = detector(df)
                if pattern:
                    patterns.append(pattern)
            except Exception as e:
                # Skip patterns that fail to detect
                continue

        # Sort by confidence
        patterns.sort(key=lambda x: x.get('confidence', 0), reverse=True)

        return patterns

    @staticmethod
    def calculate_support_resistance(df: pd.DataFrame, window: int = 20) -> Dict:
        """
        Calculate support and resistance levels

        Args:
            df: DataFrame with OHLCV data
            window: Lookback window for calculation

        Returns:
            Dictionary with support and resistance levels
        """
        recent = df.tail(window * 3)

        # Calculate pivot points
        high = recent['High'].max()
        low = recent['Low'].min()
        close = recent['Close'].iloc[-1]

        pivot = (high + low + close) / 3

        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)

        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)

        return {
            'pivot': pivot,
            'resistance_1': r1,
            'resistance_2': r2,
            'resistance_3': r3,
            'support_1': s1,
            'support_2': s2,
            'support_3': s3,
            'current_price': close
        }
