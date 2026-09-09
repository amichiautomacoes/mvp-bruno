from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st


st.set_page_config(page_title="Potencial de Votos para 2026", layout="wide")

NAVIGATION_PAGES = {
    "Pagina 1 - RaioX Votacao 2022": "pages/raiox2022.py",
    "Pagina 2 - Potencial de Votos 2026": "pages/potencial26.py",
}


def _mime_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".webp":
        return "image/webp"
    return "image/png"


def apply_background() -> None:
    bg_path = Path(__file__).resolve().parents[1] / "background" / "MGAZUL.png"
    if not bg_path.exists():
        return

    encoded = base64.b64encode(bg_path.read_bytes()).decode("utf-8")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:{_mime_type(bg_path)};base64,{encoded}");
            background-size: cover;
            background-position: center center;
            background-attachment: fixed;
        }}
        header[data-testid="stHeader"] {{
            background: transparent !important;
            height: 0 !important;
            min-height: 0 !important;
            border: 0 !important;
        }}
        [data-testid="stToolbar"] {{
            right: 0.5rem !important;
            top: 0.25rem !important;
        }}
        .stAppToolbar,
        [data-testid="stDecoration"] {{
            display: none !important;
        }}
        [data-testid="stAppViewContainer"] > .main {{
            padding-top: 0 !important;
        }}
        [data-testid="stAppViewContainer"] > .main > div {{
            padding-top: 0.35rem !important;
        }}
        [data-testid="stAppViewContainer"] > .main .block-container {{
            max-width: 1480px !important;
            padding-top: 0.5rem !important;
            padding-left: 2.2rem !important;
            padding-right: 2.2rem !important;
            padding-bottom: 2rem !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _apply_page_visual_refinement() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            color: #eaf2ff;
        }
        [data-testid="stAppViewContainer"] > .main::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            z-index: 0;
            background:
                linear-gradient(
                    100deg,
                    rgba(3, 8, 20, 0.84) 0%,
                    rgba(3, 8, 20, 0.76) 36%,
                    rgba(3, 8, 20, 0.60) 60%,
                    rgba(3, 8, 20, 0.34) 100%
                );
        }
        [data-testid="stAppViewContainer"] > .main > div {
            position: relative;
            z-index: 1;
        }
        .stApp h1 {
            font-size: 2.72rem;
            font-weight: 800;
            letter-spacing: 0.01em;
            color: #eaf2ff;
            margin-top: 0.1rem;
            margin-bottom: 0.15rem;
            line-height: 1.02;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_navigation(current_page: str) -> None:
    labels = list(NAVIGATION_PAGES.keys())
    current_label = next(
        label for label, page_path in NAVIGATION_PAGES.items() if page_path == current_page
    )
    selected_label = st.sidebar.selectbox(
        "Selecionar pagina",
        labels,
        index=labels.index(current_label),
        key=f"sidebar_navigation_{current_page}",
    )
    target_page = NAVIGATION_PAGES[selected_label]
    if target_page != current_page:
        st.switch_page(target_page)


apply_background()
_apply_page_visual_refinement()
render_sidebar_navigation("pages/potencial26.py")

st.title("Potencial de Votos para 2026")
