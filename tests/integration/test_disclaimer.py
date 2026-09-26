"""Every page must tell the user the advice is not professional advice.

The first report draft listed the missing disclaimer as a required feature;
these tests keep it from silently disappearing from any page.
"""

import os

import pytest

pytestmark = pytest.mark.integration

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def client():
    from app import app
    app.config["TESTING"] = True
    return app.test_client()


@pytest.mark.parametrize("path", ["/", "/compare", "/backtest"])
def test_every_page_shows_the_disclaimer(client, path):
    response = client.get(path)
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "not financial advice" in html
    assert "qualified financial adviser" in html


def test_signal_guide_describes_what_the_model_predicts(client):
    html = client.get("/").get_data(as_text=True)
    assert "more than 10% over the next 30 trading days" in html
    assert "undervalued" not in html      # the model does no valuation


def test_indicator_explanations_ship_with_the_detail_view():
    with open(os.path.join(ROOT, "static", "app.js"), encoding="utf-8") as fh:
        script = fh.read()
    assert "INDICATOR_HINTS" in script
    for key in ("rsi:", "macd:", "volatility:", "signal:"):
        assert key in script, key
