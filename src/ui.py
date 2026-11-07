"""Shared visual system for the MST research dashboard.

The theme is session-scoped so the light/dark switch works consistently across
all Streamlit pages without changing the analysis pipeline.
"""
import streamlit as st
import plotly.io as pio


def chart_template() -> str:
    """Return the Plotly template matching the active dashboard theme."""
    return "plotly_dark" if st.session_state.get("ui_dark_mode", False) else "plotly_white"


def init_page_ui(page_name: str = "Research workspace") -> None:
    """Render shared branding, theme control, and responsive dashboard styling."""
    st.session_state.setdefault("ui_dark_mode", False)

    st.sidebar.markdown(
        """
        <div class="mst-brand">
          <div class="mst-brand-mark">✦</div>
          <div>
            <div class="mst-brand-name">MST <span>Lab</span></div>
            <div class="mst-brand-caption">TEXT CLUSTERING WORKSPACE</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.toggle("Dark mode", key="ui_dark_mode", help="Switch the dashboard between light and dark themes.")
    st.sidebar.markdown(
        '<div class="mst-sidebar-note"><span class="mst-sidebar-dot"></span> DAA · Research platform</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="mst-kicker">DAA RESEARCH WORKSPACE <span>／</span> {page_name.upper()}</div>',
        unsafe_allow_html=True,
    )

    dark = bool(st.session_state["ui_dark_mode"])
    palette = {
        "APP_BG": "#0B1220" if dark else "#EEF1F6",
        "SURFACE": "#121F33" if dark else "rgba(255,255,255,0.88)",
        "SURFACE_SOLID": "#142238" if dark else "#FFFFFF",
        "SURFACE_SOFT": "#17263D" if dark else "#F7F8FB",
        "BORDER": "rgba(173,190,215,0.15)" if dark else "rgba(40,52,75,0.09)",
        "TEXT": "#F1F5FB" if dark else "#182033",
        "MUTED": "#9AAAC0" if dark else "#788397",
        "ACCENT": "#B7A0FF" if dark else "#8066D8",
        "ACCENT_SOFT": "rgba(183,160,255,0.14)" if dark else "rgba(128,102,216,0.10)",
        "SHADOW": "0 12px 32px rgba(0,0,0,0.20)" if dark else "0 12px 32px rgba(38,48,72,0.055)",
        "SIDEBAR": "#101B2D" if dark else "rgba(255,255,255,0.78)",
    }
    css = """
    <style>
      :root {
        --mst-bg: __APP_BG__;
        --mst-surface: __SURFACE__;
        --mst-surface-solid: __SURFACE_SOLID__;
        --mst-surface-soft: __SURFACE_SOFT__;
        --mst-border: __BORDER__;
        --mst-text: __TEXT__;
        --mst-muted: __MUTED__;
        --mst-accent: __ACCENT__;
        --mst-accent-soft: __ACCENT_SOFT__;
        --mst-shadow: __SHADOW__;
        --mst-sidebar: __SIDEBAR__;
      }
      html, body, [class*="css"] { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
      .stApp {
        color: var(--mst-text);
        background:
          radial-gradient(ellipse at 78% 2%, rgba(179,157,255,0.12), transparent 30%),
          radial-gradient(ellipse at 5% 25%, rgba(115,211,205,0.08), transparent 28%),
          var(--mst-bg);
      }
      [data-testid="stAppViewContainer"], [data-testid="stHeader"] { background: transparent; }
      [data-testid="stHeader"] { height: 2.5rem; }
      .main .block-container { max-width: 1500px; padding: 2.1rem 3.0rem 3.2rem; }
      h1, h2, h3, h4 { color: var(--mst-text) !important; letter-spacing: -0.035em; }
      h1 { font-size: clamp(1.85rem, 2.5vw, 2.45rem) !important; font-weight: 760 !important; margin: 0.1rem 0 0.35rem !important; }
      h2 { font-size: 1.45rem !important; font-weight: 700 !important; margin-top: 1.7rem !important; }
      h3 { font-size: 1.12rem !important; font-weight: 680 !important; }
      p, li, label, [data-testid="stMarkdownContainer"] { color: var(--mst-text); }
      [data-testid="stCaptionContainer"], .stCaption, small { color: var(--mst-muted) !important; }
      .mst-kicker { color: var(--mst-muted); font-size: 0.68rem; font-weight: 750; letter-spacing: 0.16em; margin: 0 0 0.8rem 0.08rem; }
      .mst-kicker span { color: var(--mst-accent); padding: 0 0.25rem; }
      section[data-testid="stSidebar"] { background: var(--mst-sidebar); border-right: 1px solid var(--mst-border); }
      section[data-testid="stSidebar"] > div { background: transparent; }
      section[data-testid="stSidebar"] .block-container { padding-top: 1.3rem; }
      .mst-brand { display:flex; align-items:center; gap:0.75rem; padding:0.65rem 0.35rem 1.3rem; margin-bottom:0.65rem; border-bottom:1px solid var(--mst-border); }
      .mst-brand-mark { width:2.55rem; height:2.55rem; display:flex; align-items:center; justify-content:center; border-radius:0.95rem; color:white; font-size:1.25rem; font-weight:800; background:linear-gradient(140deg,#B99CF6,#7B65D6); box-shadow:0 7px 18px rgba(128,102,216,0.22); }
      .mst-brand-name { color:var(--mst-text); font-weight:800; font-size:1.08rem; letter-spacing:-0.04em; line-height:1.2; }
      .mst-brand-name span { color:var(--mst-accent); }
      .mst-brand-caption { color:var(--mst-muted); font-size:0.56rem; letter-spacing:0.12em; font-weight:750; margin-top:0.24rem; }
      .mst-sidebar-note { border-top:1px solid var(--mst-border); margin-top:1rem; padding:0.9rem 0.4rem; color:var(--mst-muted); font-size:0.72rem; }
      .mst-sidebar-dot { display:inline-block; width:0.42rem; height:0.42rem; border-radius:50%; background:#35B98B; margin-right:0.45rem; box-shadow:0 0 0 3px rgba(53,185,139,0.12); }
      [data-testid="stMetric"] { background:var(--mst-surface); border:1px solid var(--mst-border); border-radius:1.1rem; padding:1rem 1.1rem; box-shadow:var(--mst-shadow); min-height:7rem; }
      [data-testid="stMetricLabel"] { color:var(--mst-muted) !important; font-size:0.78rem !important; font-weight:600 !important; }
      [data-testid="stMetricValue"] { color:var(--mst-text) !important; font-size:1.65rem !important; font-weight:760 !important; letter-spacing:-0.04em; }
      [data-testid="stMetricDelta"] { font-size:0.76rem !important; }
      div[data-testid="stVerticalBlockBorderWrapper"] { background:var(--mst-surface); border:1px solid var(--mst-border); border-radius:1.15rem; box-shadow:var(--mst-shadow); }
      div[data-testid="stExpander"] { background:var(--mst-surface); border:1px solid var(--mst-border); border-radius:0.95rem; }
      div[data-testid="stAlert"] { border-radius:0.95rem; border:1px solid var(--mst-border); }
      .stButton > button, [data-testid="stDownloadButton"] button { border-radius:0.8rem; border:1px solid var(--mst-border); min-height:2.55rem; font-weight:650; transition:all .15s ease; }
      .stButton > button[kind="primary"] { background:linear-gradient(135deg,#9278E6,#7660CF); border:0; color:#fff; box-shadow:0 6px 16px rgba(128,102,216,0.2); }
      .stButton > button:hover, [data-testid="stDownloadButton"] button:hover { border-color:var(--mst-accent); transform:translateY(-1px); }
      [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input, [data-testid="stSelectbox"] [data-baseweb="select"], [data-testid="stMultiSelect"] [data-baseweb="select"], [data-testid="stDateInput"] input, [data-testid="stTextArea"] textarea {
        background:var(--mst-surface-solid); color:var(--mst-text); border-radius:0.75rem;
      }
      [data-testid="stTabs"] button { font-weight:650; }
      [data-testid="stTabs"] button[aria-selected="true"] { color:var(--mst-accent) !important; }
      [data-testid="stDataFrame"], [data-testid="stTable"] { border:1px solid var(--mst-border); border-radius:0.9rem; overflow:hidden; }
      [data-testid="stPlotlyChart"] { border-radius:1rem; overflow:hidden; }
      hr { border-color:var(--mst-border) !important; }
      code { border-radius:0.35rem; }
      a { color:var(--mst-accent) !important; }
      @media (max-width: 900px) {
        .main .block-container { padding:1.3rem 1.1rem 2.2rem; }
        h1 { font-size:1.8rem !important; }
      }
    </style>
    """
    for key, value in palette.items():
        css = css.replace(f"__{key}__", value)
    st.markdown(css, unsafe_allow_html=True)
    pio.templates.default = chart_template()
