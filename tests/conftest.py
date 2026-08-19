"""
Pytest configuration and Hypothesis settings for the AI Stock Recommendation test suite.

Configures Hypothesis with a custom profile requiring a minimum of 100 examples
per property-based test, as specified in the design testing strategy.
"""

from hypothesis import settings, HealthCheck

# Register a custom Hypothesis profile with min 100 examples per property test
settings.register_profile(
    "default",
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
)

# CI profile with more examples for thorough testing
settings.register_profile(
    "ci",
    max_examples=500,
    suppress_health_check=[HealthCheck.too_slow],
)

# Load the default profile
settings.load_profile("default")
