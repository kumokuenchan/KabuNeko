"""
Page: AI News Sentiment Analysis

Analyzes recent news headlines and provides sentiment insights for stocks using AI.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Optional
from src.data.fetcher import get_stock_data
from src.analysis.sentiment_analyzer import SentimentAnalyzer


def fetch_news_for_ticker(ticker: str) -> list:
    """
    Fetch recent news for a ticker using yfinance.

    Args:
        ticker: Stock ticker symbol

    Returns:
        List of news dictionaries
    """
    try:
        import yfinance as yf

        # Create ticker object
        stock = yf.Ticker(ticker)

        # Try to fetch news using the news property
        try:
            news = stock.news
        except AttributeError:
            # Fallback: try get_news() method if news property doesn't exist
            try:
                news = stock.get_news()
            except:
                news = []

        # Debug info
        if not news or len(news) == 0:
            st.warning(f"⚠️ No news found for **{ticker}** via yfinance API")
            st.info("""
            **Possible reasons:**
            - This ticker may not have recent news
            - yfinance API rate limiting
            - The ticker symbol may be incorrect

            **Try:**
            - A different ticker (e.g., AAPL, TSLA, MSFT, NVDA)
            - Wait a few moments and try again
            - Check that the ticker symbol is correct
            """)
            return []

        # Format news items
        formatted_news = []
        for item in news[:20]:  # Limit to 20 most recent
            try:
                # Handle new yfinance news structure (nested under 'content')
                if 'content' in item:
                    content = item['content']
                    title = content.get('title', '')
                    publisher = content.get('provider', {}).get('displayName', 'Unknown')
                    link = content.get('clickThroughUrl', {}).get('url', '#')

                    # Parse date from ISO format string
                    pub_date_str = content.get('pubDate', '')
                    if pub_date_str:
                        try:
                            from dateutil import parser
                            published_date = parser.parse(pub_date_str)
                        except:
                            # Fallback to simple parsing
                            try:
                                published_date = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
                            except:
                                published_date = datetime.now()
                    else:
                        published_date = datetime.now()

                    thumbnail = content.get('thumbnail', {}).get('resolutions', [{}])[0].get('url', '')

                else:
                    # Fallback to old structure
                    title = item.get('title', '')
                    publisher = item.get('publisher', 'Unknown')
                    link = item.get('link', '#')

                    publish_time = item.get('providerPublishTime', None)
                    if publish_time and isinstance(publish_time, int):
                        published_date = datetime.fromtimestamp(publish_time)
                    else:
                        published_date = datetime.now()

                    thumbnail = item.get('thumbnail', {}).get('resolutions', [{}])[0].get('url', '')

                # Only add if we have a title
                if title:
                    formatted_news.append({
                        'title': title,
                        'publisher': publisher,
                        'link': link,
                        'published': published_date,
                        'thumbnail': thumbnail
                    })

            except Exception as item_error:
                # Skip items that cause errors but continue processing
                continue

        if not formatted_news:
            st.warning(f"⚠️ Found {len(news)} news items but couldn't parse any of them for **{ticker}**")

        return formatted_news

    except Exception as e:
        st.error(f"❌ Error fetching news for {ticker}: {str(e)}")
        st.info("💡 Try a well-known ticker like AAPL, TSLA, or MSFT")
        import traceback
        with st.expander("🔍 Show Error Details"):
            st.code(traceback.format_exc())
        return []


def create_sentiment_gauge(sentiment_score: float, title: str = "Overall Sentiment") -> go.Figure:
    """Create a gauge chart for sentiment visualization"""

    # Map sentiment score (-1 to 1) to gauge (0 to 100)
    gauge_value = (sentiment_score + 1) * 50

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=gauge_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20}},
        delta={'reference': 50, 'increasing': {'color': "#34C759"}, 'decreasing': {'color': "#FF3B30"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': "#007AFF"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#FFE5E5'},
                {'range': [30, 70], 'color': '#FFF8E5'},
                {'range': [70, 100], 'color': '#E5FFE5'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "darkgray", 'family': "Arial"}
    )

    return fig


def create_sentiment_timeline(analyzed_news: list) -> go.Figure:
    """Create timeline chart of sentiment over time"""

    if not analyzed_news:
        return go.Figure()

    # Sort by date
    sorted_news = sorted(analyzed_news, key=lambda x: x.get('published', datetime.now()))

    dates = [item.get('published', datetime.now()) for item in sorted_news]
    scores = [item['score'] for item in sorted_news]
    titles = [item['title'][:50] + '...' if len(item['title']) > 50 else item['title']
              for item in sorted_news]
    colors = ['#34C759' if s > 0.2 else '#FF3B30' if s < -0.2 else '#8E8E93' for s in scores]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=scores,
        mode='lines+markers',
        name='Sentiment Score',
        line=dict(color='#007AFF', width=2),
        marker=dict(size=10, color=colors, line=dict(width=1, color='white')),
        text=titles,
        hovertemplate='<b>%{text}</b><br>Sentiment: %{y:.2f}<br>Date: %{x}<extra></extra>'
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Neutral")

    fig.update_layout(
        title="Sentiment Timeline",
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        template='plotly_white',
        height=400,
        hovermode='closest'
    )

    return fig


def render() -> None:
    """Render the AI news sentiment analysis page"""

    st.title("📰 AI News Sentiment Analysis")

    st.markdown("""
    Analyze recent news sentiment for stocks using AI-powered natural language processing.
    Get insights into market mood and potential price movements based on news headlines.
    """)

    # Stock input
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        ticker = st.text_input("Enter Stock Ticker", value="AAPL", placeholder="e.g., AAPL, TSLA, MSFT").upper()

    with col2:
        days_back = st.selectbox("Analysis Period", [3, 7, 14, 30], index=1)

    with col3:
        st.write("")
        st.write("")
        analyze_btn = st.button("🔍 Analyze Sentiment", type="primary")

    if analyze_btn:
        with st.spinner(f"Fetching and analyzing news for {ticker}..."):
            # Fetch news
            news_items = fetch_news_for_ticker(ticker)

            if not news_items:
                st.warning(f"⚠️ No recent news found for {ticker}")
                st.info("💡 Try a different ticker or check back later for news updates.")
                return

            # Analyze sentiment
            sentiment_result = SentimentAnalyzer.analyze_news_batch(news_items)
            analyzed_news = sentiment_result['analyzed_news']

            # Check if any articles were successfully analyzed
            if sentiment_result['total_articles'] == 0 or not analyzed_news:
                st.warning(f"⚠️ Fetched news but couldn't analyze any articles for {ticker}")
                st.info("""
                **This might happen because:**
                - News items have empty or malformed titles
                - News API returned incomplete data

                **What to try:**
                - Try a different ticker (AAPL, TSLA, MSFT, NVDA work well)
                - Wait a few moments and try again
                """)
                with st.expander("🔍 Debug Info - Raw News Data"):
                    st.json(news_items[:3])  # Show first 3 items for debugging
                return

            st.success(f"✅ Analyzed {sentiment_result['total_articles']} news articles for {ticker}")

            # === OVERALL SENTIMENT ===
            st.markdown("### 📊 Overall Market Sentiment")

            col1, col2 = st.columns([1, 2])

            with col1:
                # Sentiment gauge
                gauge_fig = create_sentiment_gauge(
                    sentiment_result['overall_score'],
                    f"{ticker} Sentiment"
                )
                st.plotly_chart(gauge_fig, use_container_width=True)

            with col2:
                # Key metrics
                st.markdown(f"""
                <div class="metric-card" style="margin-top: 2rem;">
                    <h2 style="margin: 0; color: {'#34C759' if sentiment_result['overall_score'] > 0 else '#FF3B30' if sentiment_result['overall_score'] < 0 else '#8E8E93'};">
                        {sentiment_result['overall_emoji']} {sentiment_result['overall_label']}
                    </h2>
                    <p style="font-size: 1.2rem; margin: 0.5rem 0;">
                        Sentiment Score: <strong>{sentiment_result['overall_score']:.3f}</strong>
                    </p>
                    <p style="color: #86868B; margin: 0;">
                        Based on {sentiment_result['total_articles']} recent articles
                    </p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("---")

                # Breakdown
                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    st.metric("🟢 Bullish", sentiment_result['bullish_count'])

                with col_b:
                    st.metric("⚪ Neutral", sentiment_result['neutral_count'])

                with col_c:
                    st.metric("🔴 Bearish", sentiment_result['bearish_count'])

            # === SENTIMENT TIMELINE ===
            st.markdown("### 📈 Sentiment Over Time")
            timeline_fig = create_sentiment_timeline(analyzed_news)
            st.plotly_chart(timeline_fig, use_container_width=True)

            # === TRENDING KEYWORDS ===
            st.markdown("### 🔥 Trending Keywords")

            keywords = SentimentAnalyzer.get_trending_keywords(news_items, top_n=15)

            if keywords:
                # Create bar chart of keywords
                words, frequencies = zip(*keywords)

                fig_keywords = go.Figure(data=[
                    go.Bar(
                        x=list(frequencies),
                        y=list(words),
                        orientation='h',
                        marker=dict(color='#007AFF', opacity=0.8)
                    )
                ])

                fig_keywords.update_layout(
                    title="Most Mentioned Keywords in News",
                    xaxis_title="Frequency",
                    yaxis_title="Keyword",
                    template='plotly_white',
                    height=400,
                    yaxis={'categoryorder': 'total ascending'}
                )

                st.plotly_chart(fig_keywords, use_container_width=True)

            # === NEWS FEED ===
            st.markdown("### 📰 Recent News Headlines with Sentiment")

            # Number of articles to display
            col_display, col_spacer = st.columns([1, 3])
            with col_display:
                num_articles = st.selectbox("Show articles", [5, 10, 15, 20], index=1, key="num_articles")

            display_count = min(num_articles, len(analyzed_news))
            st.markdown(f"Showing **{display_count}** of **{len(analyzed_news)}** analyzed articles")

            for idx, item in enumerate(analyzed_news[:num_articles], 1):
                sentiment_color = '#34C759' if item['score'] > 0.2 else '#FF3B30' if item['score'] < -0.2 else '#8E8E93'

                # Get article link and publisher from original news data
                article_link = item.get('link', '#')

                # Find matching original news item for publisher
                publisher = 'Unknown'
                for original in news_items:
                    if original.get('title') == item.get('title'):
                        publisher = original.get('publisher', 'Unknown')
                        break

                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid {sentiment_color}; margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between; align-items: start; gap: 1rem;">
                        <div style="flex: 1;">
                            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                                <span style="font-size: 1.2rem;">{item['emoji']}</span>
                                <span style="background: {sentiment_color}22; color: {sentiment_color};
                                            padding: 0.25rem 0.75rem; border-radius: 12px;
                                            font-weight: 600; font-size: 0.85rem;">
                                    {item['label']} {item['score']:+.2f}
                                </span>
                            </div>
                            <h4 style="margin: 0 0 0.75rem 0; line-height: 1.4;">
                                <a href="{article_link}" target="_blank"
                                   style="color: inherit; text-decoration: none; transition: color 0.2s;">
                                    {item['title']}
                                </a>
                            </h4>
                            <div style="display: flex; gap: 1.5rem; align-items: center; flex-wrap: wrap;">
                                <p style="color: #86868B; font-size: 0.9rem; margin: 0;">
                                    <strong>📅</strong> {item.get('published', 'Unknown date').strftime('%Y-%m-%d %H:%M') if isinstance(item.get('published', ''), datetime) else 'Unknown date'}
                                </p>
                                <p style="color: #86868B; font-size: 0.9rem; margin: 0;">
                                    <strong>📰</strong> {publisher}
                                </p>
                                <a href="{article_link}" target="_blank"
                                   style="color: #007AFF; text-decoration: none; font-weight: 600; font-size: 0.9rem;">
                                    Read Article →
                                </a>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # === INSIGHTS ===
            st.markdown("### 💡 AI Insights")

            # Calculate percentages safely
            if sentiment_result['total_articles'] > 0:
                bullish_pct = (sentiment_result['bullish_count'] / sentiment_result['total_articles']) * 100
                bearish_pct = (sentiment_result['bearish_count'] / sentiment_result['total_articles']) * 100
            else:
                bullish_pct = 0
                bearish_pct = 0

            if sentiment_result['overall_score'] > 0.3:
                st.success(f"""
                **Strong Bullish Sentiment Detected** 🚀

                - {bullish_pct:.0f}% of news articles are bullish
                - Overall sentiment score: {sentiment_result['overall_score']:.3f}
                - Market mood appears optimistic for {ticker}
                - Consider monitoring for confirmation in price action
                """)
            elif sentiment_result['overall_score'] < -0.3:
                st.error(f"""
                **Strong Bearish Sentiment Detected** ⚠️

                - {bearish_pct:.0f}% of news articles are bearish
                - Overall sentiment score: {sentiment_result['overall_score']:.3f}
                - Market mood appears pessimistic for {ticker}
                - Exercise caution and review risk management
                """)
            else:
                st.info(f"""
                **Neutral Market Sentiment** ⚖️

                - Mixed sentiment with no clear bias
                - Overall sentiment score: {sentiment_result['overall_score']:.3f}
                - Wait for clearer signals before making decisions
                - Monitor for sentiment shifts in coming days
                """)

            # === DISCLAIMER ===
            st.markdown("---")
            st.warning("""
            ⚠️ **Disclaimer:** Sentiment analysis is based on news headlines and does not constitute financial advice.
            Always conduct thorough research and consult with financial professionals before making investment decisions.
            Past news sentiment does not guarantee future price movements.
            """)

    else:
        # Show info when no analysis yet
        st.info("👆 Enter a stock ticker and click 'Analyze Sentiment' to get started!")

        st.markdown("""
        ### 🎯 What This Tool Does:

        - **📰 News Collection** - Fetches recent news headlines for any stock
        - **🤖 AI Analysis** - Uses natural language processing to determine sentiment
        - **📊 Visualizations** - Charts showing sentiment trends over time
        - **🔥 Keyword Tracking** - Identifies trending topics in the news
        - **💡 Actionable Insights** - AI-generated market mood analysis

        ### 📈 How to Interpret Sentiment:

        - **🟢 Bullish (Score > 0.2)** - Positive news sentiment, potential upward pressure
        - **⚪ Neutral (Score -0.2 to 0.2)** - Mixed or balanced news sentiment
        - **🔴 Bearish (Score < -0.2)** - Negative news sentiment, potential downward pressure

        ### 💡 Pro Tips:

        1. **Combine with Technical Analysis** - Use sentiment alongside chart patterns
        2. **Monitor Trend Changes** - Watch for shifts in sentiment direction
        3. **Compare with Price Action** - Check if sentiment aligns with price movements
        4. **Be Contrarian** - Sometimes extreme sentiment can signal reversals
        """)
