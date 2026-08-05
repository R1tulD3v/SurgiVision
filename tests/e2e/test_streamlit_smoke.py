"""End-to-end smoke test for the Streamlit demo, driven by Playwright.

This is the browser-level verification layer (run after the unit tests each
phase). It is environment-adaptive:

- If no trained model is present, it asserts the app boots and correctly
  reports "Model not found" (the state in CI / a fresh clone).
- If a model IS present, it asserts the main controls render.

Run it explicitly (it is excluded from the default `pytest` run):

    # 1. start the app in one terminal
    python -m streamlit run streamlit_universal_demo.py \
        --server.headless true --server.port 8765

    # 2. in another terminal
    pip install pytest-playwright && python -m playwright install chromium
    APP_URL=http://127.0.0.1:8765 pytest tests/e2e -q
"""
import os
import re

from playwright.sync_api import Page, expect

APP_URL = os.environ.get("APP_URL", "http://127.0.0.1:8765")


def test_app_has_correct_title(page: Page):
    page.goto(APP_URL, wait_until="domcontentloaded")
    # Streamlit sets document.title from st.set_page_config (client-side).
    expect(page).to_have_title(re.compile("SurgiVision"), timeout=30_000)


def test_header_renders(page: Page):
    page.goto(APP_URL, wait_until="domcontentloaded")
    expect(page.get_by_text("SurgiVision").first).to_be_visible(timeout=30_000)


def test_model_state_is_reported(page: Page):
    """App must clearly report its state, whether or not a model is present."""
    page.goto(APP_URL, wait_until="domcontentloaded")
    page.wait_for_timeout(4000)  # allow Streamlit to hydrate and run main()
    body = page.locator("body").inner_text()
    if "Model not found" in body:
        # Fresh clone / CI: no checkpoint -> app reports it instead of crashing.
        assert "Model not found" in body
    else:
        # Model present: the sidebar controls should be available.
        expect(page.get_by_text("Controls").first).to_be_visible(timeout=30_000)


def test_no_uncaught_page_errors(page: Page):
    errors: list[str] = []
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    page.goto(APP_URL, wait_until="domcontentloaded")
    page.wait_for_timeout(4000)
    assert not errors, f"Uncaught JS page errors: {errors}"
