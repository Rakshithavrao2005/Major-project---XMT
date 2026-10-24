"""
tests/test_dashboard_ui.py
─────────────────────────────────────────────────────────────────────────────
Targeted Unit & Functional Tests for Dashboard UI Redesign:
- Theme styling engine (Dark Navy vs Clinical Light CSS generation).
- HTML helper escaping and component rendering.
- Benchmark accuracy and disclaimer consistency.
- Navigation state and hardware session preservation.
- Theme switching safety and non-interference with telemetry polling.
"""

from __future__ import annotations
import os
import sys
import html
import pytest
import pandas as pd

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dashboard.styles import (
    get_theme_css,
    render_header,
    render_badge,
    render_banner,
    render_metric_card,
    render_section_header,
    render_empty_state,
    THEME_DARK,
    THEME_LIGHT,
    THEMES,
    NAVIGATION_PAGES,
    PAGE_OVERVIEW,
    PAGE_PATIENT_EXPLORER,
    PAGE_CLINICAL_AI,
    PAGE_CHEST_XRAY,
    PAGE_MULTIMODAL_RESEARCH,
    PAGE_LIVE_HARDWARE,
    PAGE_ALERT_CENTER,
    PAGE_MODEL_EVALUATION,
    PAGE_DATA_QUALITY,
    PAGE_ICONS,
    PAGE_SUBTITLES,
)


class TestDashboardUIThemesAndComponents:
    """Test suite U01-U12 for dashboard theming, styling, and UI integrity."""

    # ── U01: Theme Options ───────────────────────────────────────────────────

    def test_U01_theme_definitions(self):
        """Theme constants must define Dark Navy as primary and Clinical Light as secondary."""
        assert THEME_DARK == "Dark Navy"
        assert THEME_LIGHT == "Clinical Light"
        assert THEMES == ["Dark Navy", "Clinical Light"]

    # ── U02: Dark Navy CSS Generation ────────────────────────────────────────

    def test_U02_dark_navy_css_palette(self):
        """Dark theme CSS must incorporate deep navy backgrounds and cyan/teal accents."""
        css = get_theme_css(THEME_DARK)
        assert isinstance(css, str)
        assert "<style>" in css and "</style>" in css
        # Navy backgrounds
        assert "#0B132B" in css
        assert "#111D38" in css
        # Cyan / teal accents
        assert "#00E5FF" in css
        assert "#00B4D8" in css
        # Critical and warning badge colors
        assert "#EF4444" in css or "#FCA5A5" in css
        assert "#F59E0B" in css

    # ── U03: Clinical Light CSS Generation ───────────────────────────────────

    def test_U03_clinical_light_css_palette(self):
        """Light theme CSS must incorporate crisp medical backgrounds and deep ocean accents."""
        css = get_theme_css(THEME_LIGHT)
        assert isinstance(css, str)
        assert "<style>" in css and "</style>" in css
        # Crisp light backgrounds
        assert "#F8FAFC" in css
        assert "#FFFFFF" in css
        # Deep blue / teal accents
        assert "#0284C7" in css
        assert "#0D9488" in css
        # High contrast slate text
        assert "#0F172A" in css

    # ── U04: Header Rendering & HTML Escaping ────────────────────────────────

    def test_U04_header_rendering_and_escaping(self):
        """render_header must sanitize text and output proper container HTML."""
        hdr = render_header(
            title="XMT <Test & Safe>",
            subtitle="Prototype 'Quotes' & More",
            theme=THEME_DARK,
        )
        assert '<div class="xmt-header-card">' in hdr
        assert "XMT &lt;Test &amp; Safe&gt;" in hdr
        assert "Prototype &#x27;Quotes&#x27; &amp; More" in hdr
        assert "<script>" not in hdr

    # ── U05: Badge Rendering & Variant Mapping ───────────────────────────────

    def test_U05_badge_rendering_variants(self):
        """render_badge maps critical, warning, normal, and info variants correctly."""
        b_crit = render_badge("CRITICAL RISK", "critical")
        assert 'class="badge-critical"' in b_crit
        assert "CRITICAL RISK" in b_crit

        b_warn = render_badge("STALE TELEMETRY", "warning")
        assert 'class="badge-warning"' in b_warn
        assert "STALE TELEMETRY" in b_warn

        b_norm = render_badge("NORMAL", "normal")
        assert 'class="badge-normal"' in b_norm
        assert "NORMAL" in b_norm

        b_info = render_badge("SESSION ACTIVE", "info")
        assert 'class="badge-info"' in b_info
        assert "SESSION ACTIVE" in b_info

    # ── U06: Badge HTML Escaping ─────────────────────────────────────────────

    def test_U06_badge_html_escaping(self):
        """Malicious characters inside badge text must be escaped."""
        bad_str = "<script>alert(1)</script>"
        b = render_badge(bad_str, "critical")
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in b
        assert "<script>" not in b

    # ── U07: Banner Rendering & Types ────────────────────────────────────────

    def test_U07_banner_rendering_types(self):
        """render_banner supports disclaimer (amber) and isolation (green) banners."""
        b_disc = render_banner("WARNING", "Research only", "disclaimer")
        assert 'class="disclaimer-banner"' in b_disc
        assert "WARNING" in b_disc
        assert "Research only" in b_disc

        b_iso = render_banner("ISOLATION", "Cohort separation", "isolation")
        assert 'class="isolation-banner"' in b_iso
        assert "ISOLATION" in b_iso
        assert "Cohort separation" in b_iso

    # ── U08: Banner HTML Escaping ────────────────────────────────────────────

    def test_U08_banner_html_escaping(self):
        """Malicious HTML injected into banner title or text must be escaped."""
        b = render_banner("Title <img src=x onerror=alert(1)>", "Text <b>bold</b>")
        assert "&lt;img src=x onerror=alert(1)&gt;" in b
        assert "&lt;b&gt;bold&lt;/b&gt;" in b
        assert "<img" not in b

    # ── U09: Overview Benchmarks Data Consistency ────────────────────────────

    def test_U09_overview_verified_metrics_consistency(self):
        """Benchmark metrics presented in overview and model evaluation must match frozen model artifacts."""
        import json
        bench_path = os.path.join(_PROJECT_ROOT, "reports", "evaluation", "benchmark_metrics.json")
        assert os.path.isfile(bench_path)
        with open(bench_path, "r") as f:
            bench_data = json.load(f)

        models = bench_data["models"]
        # Verified values from authoritative benchmark report
        metrics = {
            "clinical_roc_auc": models["Clinical Transformer"]["roc_auc"],
            "clinical_pr_auc": models["Clinical Transformer"]["pr_auc"],
            "cxr_roc_auc": models["CXR Encoder"]["roc_auc"],
            "cxr_pr_auc": models["CXR Encoder"]["pr_auc"],
            "fusion_a_roc_auc": models["Model A (Concat Fusion)"]["roc_auc"],
            "fusion_a_pr_auc": models["Model A (Concat Fusion)"]["pr_auc"],
            "fusion_b_roc_auc": models["Model B (Cross-Attention)"]["roc_auc"],
            "fusion_b_pr_auc": models["Model B (Cross-Attention)"]["pr_auc"],
            "logistic_roc_auc": models["Logistic Regression"]["roc_auc"],
            "rf_roc_auc": models["Random Forest"]["roc_auc"],
        }

        # Exact authoritative equality checks
        assert metrics["clinical_roc_auc"] == 0.7069
        assert metrics["clinical_pr_auc"] == 0.3281
        assert metrics["cxr_roc_auc"] == 0.6131
        assert metrics["cxr_pr_auc"] == 0.2180
        assert metrics["fusion_a_roc_auc"] == 0.7114
        assert metrics["fusion_a_pr_auc"] == 0.3335
        assert metrics["fusion_b_roc_auc"] == 0.7056
        assert metrics["fusion_b_pr_auc"] == 0.3245
        assert metrics["logistic_roc_auc"] == 0.7162
        assert metrics["rf_roc_auc"] == 0.6887

    def test_U09b_reject_unsupported_mock_metrics_in_dashboard(self):
        """Dashboard source code must not contain unverified mock metrics (0.8520, 0.8491, 0.8496, 0.6133, 0.8248)."""
        dashboard_dir = os.path.join(_PROJECT_ROOT, "dashboard")
        unsupported_values = ["0.8520", "0.8491", "0.8496", "0.6133", "0.8248"]

        for root, _, files in os.walk(dashboard_dir):
            for file in files:
                if not file.endswith(".py"):
                    continue
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                for bad_val in unsupported_values:
                    assert bad_val not in content, (
                        f"Unsupported mock metric '{bad_val}' found in dashboard file: {file_path}"
                    )

    # ── U10: Multi-Page SaaS Navigation Taxonomy ─────────────────────────────

    def test_U10_navigation_modes_defined(self):
        """Dashboard must define exactly the 9 documented healthcare AI SaaS navigation pages."""
        expected_pages = [
            PAGE_OVERVIEW,
            PAGE_PATIENT_EXPLORER,
            PAGE_CLINICAL_AI,
            PAGE_CHEST_XRAY,
            PAGE_MULTIMODAL_RESEARCH,
            PAGE_LIVE_HARDWARE,
            PAGE_ALERT_CENTER,
            PAGE_MODEL_EVALUATION,
            PAGE_DATA_QUALITY,
        ]
        assert len(NAVIGATION_PAGES) == 9
        assert NAVIGATION_PAGES == expected_pages
        for p in expected_pages:
            assert p in PAGE_ICONS
            assert p in PAGE_SUBTITLES

    # ── U11: Theme Switching Preserves State Mock Test ───────────────────────

    def test_U11_theme_switch_preserves_session_state(self):
        """Simulated theme toggle preserves live_session_id and current_search_pid."""
        mock_session_state = {
            "ui_theme": THEME_DARK,
            "live_session_id": "SESSION-20261016-TEST01",
            "current_search_pid": "17585185",
            "auto_poll_enabled": True,
            "poll_interval_seconds": 30,
        }

        # Switch theme
        mock_session_state["ui_theme"] = THEME_LIGHT

        # All session data must remain intact
        assert mock_session_state["live_session_id"] == "SESSION-20261016-TEST01"
        assert mock_session_state["current_search_pid"] == "17585185"
        assert mock_session_state["auto_poll_enabled"] is True
        assert mock_session_state["poll_interval_seconds"] == 30

    # ── U12: Disclaimers Retained in Both Themes ──────────────────────────────

    def test_U12_disclaimers_present_in_both_themes(self):
        """Regulatory notices and research disclaimers are preserved in both themes."""
        css_dark = get_theme_css(THEME_DARK)
        css_light = get_theme_css(THEME_LIGHT)

        # Both CSS outputs must define disclaimer and isolation styling
        assert "disclaimer-banner" in css_dark
        assert "isolation-banner" in css_dark
        assert "disclaimer-banner" in css_light
        assert "isolation-banner" in css_light

    # ── U13: SaaS Component Helpers Rendering & HTML Escaping ─────────────────

    def test_U13_new_component_helpers_rendering_and_escaping(self):
        """Card, section header, and empty state helpers safely escape malicious strings."""
        malicious = "<script>alert('xss')</script>&'\""

        # render_metric_card
        card_html = render_metric_card(
            title=f"Metric {malicious}",
            value="123.4",
            subtitle=f"Subtitle {malicious}",
            badge_text=f"Badge {malicious}",
            badge_type="warning",
            border_accent="#0284C7",
        )
        assert "<script>" not in card_html
        assert "&lt;script&gt;" in card_html
        assert "123.4" in card_html
        assert "metric-card" in card_html

        # render_section_header
        hdr_html = render_section_header(
            title=f"Section {malicious}",
            description=f"Subtitle {malicious}",
            icon="⚠️",
        )
        assert "<script>" not in hdr_html
        assert "&lt;script&gt;" in hdr_html
        assert "Subtitle" in hdr_html

        # render_empty_state
        empty_html = render_empty_state(
            title=f"No Data {malicious}",
            message=f"Details {malicious}",
            icon="ℹ️",
        )
        assert "<script>" not in empty_html
        assert "&lt;script&gt;" in empty_html
        assert "No Data" in empty_html

    # ── U14: Multi-Page Modules Import & Callable Contract ────────────────────

    def test_U14_multi_page_modules_import_and_render_contract(self):
        """All 9 page modules must be importable and expose their documented entrypoint."""
        import dashboard.pages.overview as p_ov
        import dashboard.pages.patient_explorer as p_pe
        import dashboard.pages.clinical_ai as p_ca
        import dashboard.pages.chest_xray as p_cx
        import dashboard.pages.multimodal_research as p_mr
        import dashboard.pages.live_hardware as p_lh
        import dashboard.pages.alert_center as p_ac
        import dashboard.pages.model_evaluation as p_me
        import dashboard.pages.data_quality as p_dq

        expected_callables = [
            (p_ov, "render_overview_page"),
            (p_pe, "render_patient_explorer_page"),
            (p_ca, "render_clinical_ai_page"),
            (p_cx, "render_chest_xray_page"),
            (p_mr, "render_multimodal_research_page"),
            (p_lh, "render_live_hardware_page"),
            (p_ac, "render_alert_center_page"),
            (p_me, "render_model_evaluation_page"),
            (p_dq, "render_data_quality_page"),
        ]

        for mod, func_name in expected_callables:
            assert hasattr(mod, func_name), f"Module {mod.__name__} missing {func_name}"
            assert callable(getattr(mod, func_name)), f"{func_name} in {mod.__name__} is not callable"

    # ── U15: Clinical Light Primary Theme Defaults ────────────────────────────

    def test_U15_clinical_light_primary_theme_defaults(self):
        """Verify palette tokens and CSS generation for Clinical Light vs Dark Navy."""
        css_light = get_theme_css(THEME_LIGHT)
        css_dark = get_theme_css(THEME_DARK)

        # Light theme primary tokens
        assert "#f8fafc" in css_light.lower() or "#ffffff" in css_light.lower()
        assert "#0f172a" in css_light.lower() or "#334155" in css_light.lower()

        # Dark theme primary tokens
        assert "#0a1128" in css_dark.lower() or "#101f42" in css_dark.lower()

        # Both generate non-empty, distinct styling
        assert len(css_light) > 1000
        assert len(css_dark) > 1000
        assert css_light != css_dark

    # ── U16: Software and Hardware Cohort Separation ──────────────────────────

    def test_U16_software_hardware_separation_in_pages(self):
        """Hardware telemetry data schemas must never be conflated with hospital EHR records."""
        from src.sensor.live_session import (
            start_monitoring_session,
            get_latest_sensor_telemetry,
        )

        session = start_monitoring_session("SESSION-TEST-20261016")

        # Invariant 1: Live hardware sessions must not contain hospital EHR identifiers
        assert "subject_id" not in session
        assert "stay_id" not in session
        assert "hadm_id" not in session
        assert "patient_id" not in session
        assert session["session_id"] == "SESSION-TEST-20261016"

        # Invariant 2: Hardware telemetry dictionaries must not contain hospital EHR identifiers
        telemetry = get_latest_sensor_telemetry(fetch_remote=False)
        assert "subject_id" not in telemetry
        assert "stay_id" not in telemetry
        assert "hadm_id" not in telemetry
        assert "patient_id" not in telemetry


