"""
UI Theme Module

Provides CSS styling for light and dark modes.
"""


def get_theme_css(dark_mode: bool = False) -> str:
    """Generate CSS based on current theme"""
    if dark_mode:
        return """
        <style>
            .main-header {
                font-size: 2.5rem;
                font-weight: bold;
                color: #4dabf7;
                text-align: center;
                padding: 1rem 0;
            }
            .metric-card {
                background-color: #2d3748;
                padding: 1rem;
                border-radius: 0.5rem;
                margin: 0.5rem 0;
                color: #e2e8f0;
            }
            .stAlert {
                margin-top: 1rem;
            }
            /* Dark mode overrides */
            .stApp {
                background-color: #1a202c;
                color: #e2e8f0;
            }
            .stMarkdown {
                color: #e2e8f0;
            }
            .stDataFrame {
                background-color: #2d3748;
            }
            /* News article link hover effects */
            .metric-card h4 a:hover {
                color: #007AFF !important;
            }
            .metric-card a {
                transition: all 0.2s ease;
            }
        </style>
        """
    else:
        return """
        <style>
            .main-header {
                font-size: 2.5rem;
                font-weight: bold;
                color: #1f77b4;
                text-align: center;
                padding: 1rem 0;
            }
            .metric-card {
                background-color: #f0f2f6;
                padding: 1rem;
                border-radius: 0.5rem;
                margin: 0.5rem 0;
            }
            .stAlert {
                margin-top: 1rem;
            }
            /* News article link hover effects */
            .metric-card h4 a:hover {
                color: #007AFF !important;
            }
            .metric-card a {
                transition: all 0.2s ease;
            }
        </style>
        """
