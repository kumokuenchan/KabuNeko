"""
AI News Sentiment Analysis Module

Analyzes news headlines and text to determine market sentiment for stocks.
"""

import re
from typing import Dict, List, Tuple
from datetime import datetime, timedelta


class SentimentAnalyzer:
    """Analyze sentiment of financial news and text"""

    # Financial sentiment keywords
    POSITIVE_KEYWORDS = [
        'surge', 'soar', 'rally', 'jump', 'gain', 'rise', 'climb', 'profit',
        'beat', 'exceed', 'outperform', 'strong', 'growth', 'bullish', 'high',
        'upgrade', 'buy', 'positive', 'optimistic', 'breakthrough', 'record',
        'success', 'winning', 'boom', 'innovation', 'expansion', 'approved'
    ]

    NEGATIVE_KEYWORDS = [
        'fall', 'drop', 'plunge', 'tumble', 'decline', 'loss', 'miss', 'weak',
        'bearish', 'low', 'downgrade', 'sell', 'negative', 'pessimistic', 'crash',
        'failure', 'concern', 'risk', 'warning', 'cut', 'reduce', 'lawsuit',
        'investigation', 'scandal', 'bankruptcy', 'recession', 'crisis'
    ]

    INTENSIFIERS = {
        'very': 1.5,
        'extremely': 2.0,
        'highly': 1.5,
        'significantly': 1.5,
        'substantially': 1.5,
        'sharply': 1.5,
        'dramatically': 2.0,
        'massive': 2.0,
        'huge': 1.8,
        'major': 1.5
    }

    @classmethod
    def analyze_text(cls, text: str) -> Dict[str, any]:
        """
        Analyze sentiment of a single text.

        Args:
            text: Text to analyze (headline or article)

        Returns:
            Dictionary with sentiment score, label, and breakdown
        """
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)

        positive_score = 0
        negative_score = 0
        intensifier = 1.0

        for i, word in enumerate(words):
            # Check for intensifiers
            if word in cls.INTENSIFIERS:
                intensifier = cls.INTENSIFIERS[word]
                continue

            # Check sentiment
            if word in cls.POSITIVE_KEYWORDS:
                positive_score += 1 * intensifier
                intensifier = 1.0
            elif word in cls.NEGATIVE_KEYWORDS:
                negative_score += 1 * intensifier
                intensifier = 1.0
            else:
                intensifier = 1.0

        # Calculate overall sentiment (-1 to 1)
        total = positive_score + negative_score
        if total == 0:
            sentiment_score = 0
        else:
            sentiment_score = (positive_score - negative_score) / total

        # Determine sentiment label
        if sentiment_score > 0.2:
            sentiment_label = "Bullish"
            emoji = "🟢"
        elif sentiment_score < -0.2:
            sentiment_label = "Bearish"
            emoji = "🔴"
        else:
            sentiment_label = "Neutral"
            emoji = "⚪"

        return {
            'score': round(sentiment_score, 3),
            'label': sentiment_label,
            'emoji': emoji,
            'positive_count': int(positive_score),
            'negative_count': int(negative_score),
            'confidence': round(abs(sentiment_score), 3)
        }

    @classmethod
    def analyze_news_batch(cls, news_items: List[Dict]) -> Dict[str, any]:
        """
        Analyze a batch of news items.

        Args:
            news_items: List of news dictionaries with 'title' and 'published' keys

        Returns:
            Aggregated sentiment analysis
        """
        if not news_items:
            return {
                'overall_score': 0,
                'overall_label': 'Neutral',
                'overall_emoji': '⚪',
                'total_articles': 0,
                'bullish_count': 0,
                'bearish_count': 0,
                'neutral_count': 0,
                'sentiment_trend': [],
                'analyzed_news': []
            }

        analyzed_news = []
        sentiment_scores = []
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0

        for item in news_items:
            title = item.get('title', '')
            if not title:
                continue

            sentiment = cls.analyze_text(title)
            sentiment['title'] = title
            sentiment['published'] = item.get('published', '')
            sentiment['link'] = item.get('link', '')

            analyzed_news.append(sentiment)
            sentiment_scores.append(sentiment['score'])

            if sentiment['label'] == 'Bullish':
                bullish_count += 1
            elif sentiment['label'] == 'Bearish':
                bearish_count += 1
            else:
                neutral_count += 1

        # Calculate overall sentiment
        if sentiment_scores:
            overall_score = sum(sentiment_scores) / len(sentiment_scores)
        else:
            overall_score = 0

        if overall_score > 0.15:
            overall_label = "Bullish"
            overall_emoji = "🟢"
        elif overall_score < -0.15:
            overall_label = "Bearish"
            overall_emoji = "🔴"
        else:
            overall_label = "Neutral"
            overall_emoji = "⚪"

        return {
            'overall_score': round(overall_score, 3),
            'overall_label': overall_label,
            'overall_emoji': overall_emoji,
            'total_articles': len(analyzed_news),
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'neutral_count': neutral_count,
            'analyzed_news': analyzed_news,
            'confidence': round(abs(overall_score), 3)
        }

    @classmethod
    def get_trending_keywords(cls, news_items: List[Dict], top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Extract trending keywords from news.

        Args:
            news_items: List of news dictionaries
            top_n: Number of top keywords to return

        Returns:
            List of (keyword, frequency) tuples
        """
        word_freq = {}
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                     'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this',
                     'that', 'these', 'those', 'it', 'its'}

        for item in news_items:
            title = item.get('title', '')
            words = re.findall(r'\b\w+\b', title.lower())

            for word in words:
                if len(word) > 3 and word not in stopwords:
                    word_freq[word] = word_freq.get(word, 0) + 1

        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return sorted_words[:top_n]
