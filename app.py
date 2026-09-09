from __future__ import annotations

import streamlit as st


pages = [
    st.Page(
        "pages/raiox2022.py",
        title="Pagina 1 - RaioX Votacao 2022",
        default=True,
    ),
    st.Page(
        "pages/potencial26.py",
        title="Pagina 2 - Potencial de Votos 2026",
    ),
]

current_page = st.navigation(pages, position="hidden")
current_page.run()
