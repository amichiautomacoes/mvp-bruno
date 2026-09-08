from __future__ import annotations

import base64
import html
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import HfFileSystem


st.set_page_config(page_title="RaioX Votacao 2022", layout="wide")

SQ_CANDIDATO = "130001598582"
NR_CANDIDATO = "30190"
CANDIDATE_NAME = "BRUNO ARAUJO"
CANDIDATE_FULL_NAME = "BRUNO ARAUJO OLIVEIRA"
CANDIDATE_OFFICE = "DEPUTADO ESTADUAL"
CANDIDATE_PARTY = "NOVO"
CANDIDATE_FOLDER = "bruno-raiox22"
GEO_PARQUET = (
    "2022_deputado_estadual_MG_130001598582_bruno_araujo_geografico.parquet"
)
PROFILE_ELECTOR_PARQUET = (
    "2022_deputado_estadual_MG_130001598582_bruno_araujo_perfil_eleitor.parquet"
)
PHOTO_FILENAME = "FMG130001598582_div.jpg"
DEMOGRAPHIC_FILES = {
    "Gênero": "2022_deputado_estadual_MG_130001598582_bruno_araujo_genero.parquet",
    "Idade": "2022_deputado_estadual_MG_130001598582_bruno_araujo_idade.parquet",
    "Escolaridade": "2022_deputado_estadual_MG_130001598582_bruno_araujo_escolaridade.parquet",
    "Estado civil": "2022_deputado_estadual_MG_130001598582_bruno_araujo_estado_civil.parquet",
}
DEMOGRAPHIC_PREFIXES = {
    "Gênero": "votos_genero_",
    "Idade": "votos_idade_",
    "Escolaridade": "votos_escolaridade_",
    "Estado civil": "votos_estado_civil_",
}
PROFILE_ELECTOR_TITLES = {
    1: "Eleitorado de maior escolaridade",
    2: "Perfil jovem",
    3: "Adulto jovem e intermediário",
    4: "Eleitorado mais velho e popular",
}
PROFILE_JOIN_KEYS = [
    "nr_candidato",
    "nr_mesorregiao",
    "cd_ibge_municipio",
    "cd_setor_censitario",
    "cd_bairro",
    "CD_MUNICIPIO",
    "municipio",
    "bairro",
]


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
        @media (min-width: 1200px) {{
            [data-testid="stAppViewContainer"] > .main .block-container {{
                max-width: 1580px !important;
                padding-left: 2.8rem !important;
                padding-right: 2.8rem !important;
                padding-bottom: 2.5rem !important;
            }}
            div[data-testid="column"] > div {{
                height: 100%;
            }}
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
        .stApp [data-testid="stCaptionContainer"] p {
            color: #b7c7e6 !important;
            font-size: 0.95rem !important;
            margin-bottom: 1.1rem !important;
            line-height: 1.35 !important;
        }
        .candidate-photo-card,
        .candidate-info-card,
        .mapa-section-card {
            background: linear-gradient(
                145deg,
                rgba(7, 18, 36, 0.70) 0%,
                rgba(7, 18, 36, 0.52) 100%
            );
            border: 1px solid rgba(184, 208, 255, 0.24);
            border-radius: 18px;
            box-shadow: 0 18px 40px rgba(2, 9, 24, 0.42);
            backdrop-filter: blur(6px);
            -webkit-backdrop-filter: blur(6px);
        }
        .candidate-photo-card {
            padding: 0.7rem;
            min-height: 13rem;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .candidate-photo-card img {
            width: 100%;
            max-height: 12rem;
            object-fit: cover;
            border-radius: 14px;
        }
        .candidate-info-card {
            padding: 1.05rem 1.15rem;
            min-height: 13rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .candidate-info-card h3 {
            color: #f8fbff;
            font-size: 1.28rem;
            font-weight: 800;
            line-height: 1.2;
            margin: 0.28rem 0;
            letter-spacing: 0.01em;
        }
        .mapa-section-card {
            padding: 0.86rem 1rem 0.78rem 1rem;
            margin: 1.35rem 0 0.62rem 0;
        }
        .mapa-section-card--tight {
            margin-top: 0.92rem;
            margin-bottom: 0.44rem;
        }
        .mapa-section-title {
            color: #eaf2ff;
            font-size: 1.74rem;
            font-weight: 700;
            line-height: 1.06;
            letter-spacing: 0.01em;
        }
        .mapa-section-subtitle {
            margin-top: 0.17rem;
            color: #b7c7e6;
            font-size: 0.91rem;
        }
        .mapa-major-section {
            position: relative;
            overflow: hidden;
            padding: 1.08rem 1.2rem 1.02rem 1.2rem;
            margin: 1.45rem 0 0.82rem 0;
            border-radius: 20px;
            border: 1px solid rgba(191, 219, 254, 0.28);
            background:
                linear-gradient(
                    132deg,
                    rgba(12, 29, 56, 0.88) 0%,
                    rgba(9, 22, 43, 0.74) 54%,
                    rgba(8, 20, 40, 0.56) 100%
                );
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.03),
                0 22px 48px rgba(2, 9, 24, 0.46);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
        }
        .mapa-major-section::before {
            content: "";
            position: absolute;
            inset: 0 auto 0 0;
            width: 6px;
            background: linear-gradient(
                180deg,
                rgba(191, 219, 254, 0.98) 0%,
                rgba(59, 130, 246, 0.92) 40%,
                rgba(11, 31, 77, 0.94) 100%
            );
            box-shadow: 0 0 26px rgba(96, 165, 250, 0.36);
        }
        .mapa-major-section::after {
            content: "";
            position: absolute;
            top: -28%;
            right: -8%;
            width: 280px;
            height: 280px;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(96, 165, 250, 0.18) 0%, rgba(96, 165, 250, 0) 70%);
            pointer-events: none;
        }
        .mapa-major-section-title::before {
            content: "";
            position: absolute;
            top: -0.32rem;
            left: 0;
            width: 5.25rem;
            height: 2px;
            border-radius: 999px;
            background: linear-gradient(90deg, rgba(147, 197, 253, 0.95) 0%, rgba(147, 197, 253, 0.12) 100%);
            opacity: 0.9;
        }
        .mapa-major-section-title {
            position: relative;
            z-index: 1;
            color: #f8fbff;
            font-size: 2.06rem;
            font-weight: 820;
            line-height: 1.04;
            letter-spacing: 0.01em;
            padding-top: 0.18rem;
        }
        .mapa-major-section-subtitle {
            position: relative;
            z-index: 1;
            max-width: 62rem;
            margin-top: 0.36rem;
            color: #d1def7;
            font-size: 0.96rem;
            line-height: 1.45;
        }
        .mapa-kpi-card {
            background: linear-gradient(
                145deg,
                rgba(10, 23, 45, 0.72) 0%,
                rgba(10, 23, 45, 0.48) 100%
            );
            border: 1px solid rgba(184, 208, 255, 0.28);
            border-radius: 16px;
            padding: 0.82rem 0.9rem 0.72rem 0.9rem;
            box-shadow: 0 14px 32px rgba(2, 9, 24, 0.38);
            min-height: 7.8rem;
        }
        .mapa-kpi-label {
            color: #b7c7e6;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .mapa-kpi-label--wrap {
            white-space: normal;
            display: -webkit-box;
            -webkit-box-orient: vertical;
            -webkit-line-clamp: 2;
            line-clamp: 2;
            overflow: hidden;
            text-overflow: clip;
            min-height: 2.2em;
            line-height: 1.35;
        }
        .mapa-kpi-value {
            color: #f8fbff;
            font-size: 2.08rem;
            line-height: 1.04;
            font-weight: 800;
            margin-top: 0.2rem;
        }
        .mapa-kpi-caption {
            color: #b7c7e6;
            font-size: 0.8rem;
            margin-top: 0.34rem;
            line-height: 1.25;
        }
        .mapa-kpi-card--featured {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1rem;
            min-height: 0;
            padding: 0.95rem 1.05rem 0.9rem 1.05rem;
            margin-top: 0.55rem;
            margin-bottom: 1rem;
        }
        .mapa-kpi-featured-copy {
            flex: 1 1 auto;
            min-width: 0;
        }
        .mapa-kpi-value--featured {
            font-size: 2.25rem;
            line-height: 1.08;
            white-space: normal;
            text-wrap: balance;
            max-width: 16ch;
        }
        .mapa-kpi-caption--featured {
            font-size: 0.92rem;
            margin-top: 0.45rem;
        }
        .mapa-kpi-badge {
            flex: 0 0 auto;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.4rem 0.72rem;
            border-radius: 999px;
            border: 1px solid rgba(191, 219, 254, 0.26);
            background: rgba(96, 165, 250, 0.14);
            color: #dbeafe;
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            white-space: nowrap;
        }
        .perfil-eleitor-card {
            background: linear-gradient(
                145deg,
                rgba(10, 23, 45, 0.74) 0%,
                rgba(10, 23, 45, 0.50) 100%
            );
            border: 1px solid rgba(184, 208, 255, 0.28);
            border-radius: 16px;
            padding: 1rem 1.05rem 0.95rem 1.05rem;
            box-shadow: 0 18px 40px rgba(2, 9, 24, 0.42);
            backdrop-filter: blur(6px);
            -webkit-backdrop-filter: blur(6px);
            min-height: 17.2rem;
            margin-bottom: 1rem;
        }
        .perfil-eleitor-title {
            color: #f8fbff;
            font-size: 1.16rem;
            line-height: 1.18;
            font-weight: 800;
            margin-bottom: 0.72rem;
        }
        .perfil-eleitor-votes-label {
            color: #b7c7e6;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .perfil-eleitor-votes {
            color: #f8fbff;
            font-size: 2.15rem;
            line-height: 1.04;
            font-weight: 820;
            margin: 0.1rem 0 0.72rem 0;
        }
        .perfil-eleitor-row {
            display: grid;
            grid-template-columns: minmax(7rem, 0.75fr) minmax(0, 1.25fr);
            gap: 0.7rem;
            padding: 0.46rem 0;
            border-top: 1px solid rgba(184, 208, 255, 0.14);
            align-items: start;
        }
        .perfil-eleitor-row-label {
            color: #b7c7e6;
            font-size: 0.8rem;
            font-weight: 700;
        }
        .perfil-eleitor-row-value {
            color: #eaf2ff;
            font-size: 0.91rem;
            line-height: 1.25;
            font-weight: 650;
            text-align: right;
        }
        .st-key-composicao_demografica_card {
            background: linear-gradient(
                145deg,
                rgba(7, 18, 36, 0.74) 0%,
                rgba(7, 18, 36, 0.54) 100%
            );
            border: 1px solid rgba(184, 208, 255, 0.24);
            border-radius: 18px;
            padding: 1rem 1.05rem 0.78rem 1.05rem;
            box-shadow: 0 18px 40px rgba(2, 9, 24, 0.42);
            backdrop-filter: blur(6px);
            -webkit-backdrop-filter: blur(6px);
            margin-top: 0.25rem;
            margin-bottom: 1.1rem;
        }
        .composicao-demografica-title {
            color: #f8fbff;
            font-size: 1.42rem;
            line-height: 1.1;
            font-weight: 820;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-top: 0.45rem;
        }
        .st-key-composicao_demografica_card [data-testid="stPlotlyChart"] {
            background: transparent !important;
            border: 0 !important;
            box-shadow: none !important;
            backdrop-filter: none !important;
            -webkit-backdrop-filter: none !important;
            padding: 0.25rem 0 0 0 !important;
        }
        .st-key-top_cidades_mesorregioes_card {
            background: linear-gradient(
                145deg,
                rgba(7, 18, 36, 0.74) 0%,
                rgba(7, 18, 36, 0.54) 100%
            );
            border: 1px solid rgba(184, 208, 255, 0.24);
            border-radius: 18px;
            padding: 1rem 1.05rem 0.72rem 1.05rem;
            box-shadow: 0 18px 40px rgba(2, 9, 24, 0.42);
            backdrop-filter: blur(6px);
            -webkit-backdrop-filter: blur(6px);
            margin-top: 1rem;
            margin-bottom: 1.1rem;
        }
        .top-territorio-title {
            color: #f8fbff;
            font-size: 1.42rem;
            line-height: 1.1;
            font-weight: 820;
            letter-spacing: 0.01em;
            margin-top: 0.45rem;
        }
        .st-key-top_cidades_mesorregioes_card [data-testid="stPlotlyChart"] {
            background: transparent !important;
            border: 0 !important;
            box-shadow: none !important;
            backdrop-filter: none !important;
            -webkit-backdrop-filter: none !important;
            padding: 0.25rem 0 0 0 !important;
        }
        [data-testid="stPlotlyChart"] {
            background: linear-gradient(
                145deg,
                rgba(7, 18, 36, 0.72) 0%,
                rgba(7, 18, 36, 0.54) 100%
            );
            border: 1px solid rgba(184, 208, 255, 0.24);
            border-radius: 18px;
            padding: 0.56rem 0.66rem 0.3rem 0.66rem;
            box-shadow: 0 18px 40px rgba(2, 9, 24, 0.42);
            backdrop-filter: blur(6px);
            -webkit-backdrop-filter: blur(6px);
        }
        [data-testid="stPlotlyChart"] > div {
            border-radius: 14px;
            overflow: hidden;
        }
        [data-testid="stSelectbox"] label p {
            color: #c8d9f8 !important;
            font-size: 0.83rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.06em !important;
            text-transform: uppercase;
        }
        [data-testid="stSelectbox"] [data-baseweb="select"] > div {
            min-height: 3rem;
            border-radius: 12px;
            background: rgba(11, 24, 46, 0.72) !important;
            border: 1px solid rgba(184, 208, 255, 0.28) !important;
            box-shadow: 0 10px 24px rgba(2, 9, 24, 0.28);
        }
        [data-testid="stSelectbox"] [data-baseweb="select"] input,
        [data-testid="stSelectbox"] [data-baseweb="select"] div {
            color: #eaf2ff !important;
            font-weight: 600;
        }
        .st-key-bairro_treemap_card,
        .st-key-perfil_bar_card {
            background: linear-gradient(
                145deg,
                rgba(7, 18, 36, 0.72) 0%,
                rgba(7, 18, 36, 0.54) 100%
            );
            border: 1px solid rgba(184, 208, 255, 0.24);
            border-radius: 18px;
            padding: 0.72rem 0.8rem 0.58rem 0.8rem;
            box-shadow: 0 18px 40px rgba(2, 9, 24, 0.42);
            backdrop-filter: blur(6px);
            -webkit-backdrop-filter: blur(6px);
            min-height: 575px;
        }
        .st-key-bairro_treemap_card [data-testid="stPlotlyChart"],
        .st-key-perfil_bar_card [data-testid="stPlotlyChart"] {
            background: transparent !important;
            border: 0 !important;
            box-shadow: none !important;
            backdrop-filter: none !important;
            -webkit-backdrop-filter: none !important;
            padding: 0 !important;
        }
        @media (max-width: 1100px) {
            .mapa-kpi-card--featured {
                flex-direction: column;
            }
            .mapa-kpi-badge {
                white-space: normal;
            }
            .mapa-kpi-value--featured {
                max-width: none;
            }
        }
        @media (max-width: 680px) {
            .mapa-major-section-title {
                font-size: 1.55rem;
            }
            .mapa-section-title {
                font-size: 1.35rem;
            }
            .mapa-kpi-value {
                font-size: 1.76rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _format_int(value: float | int) -> str:
    return f"{int(round(float(value or 0))):,}".replace(",", ".")


def _format_percent(value: float | int) -> str:
    return f"{float(value or 0):.1f}%".replace(".", ",")


def _escape(value: object) -> str:
    return html.escape(str(value or ""))


def _build_log_colorbar_ticks(max_votes: float) -> tuple[list[float], list[str]]:
    max_v = float(max_votes or 0.0)
    if max_v <= 0:
        return [0.0], ["0"]

    ticks_raw: set[int] = {0}
    scale = 1
    while scale <= max_v:
        for mult in [1, 2, 5]:
            candidate = mult * scale
            if candidate <= max_v:
                ticks_raw.add(int(candidate))
        scale *= 10
    ticks_raw.add(int(max_v))

    tickvals = [0.0]
    ticktext = ["0"]
    for raw in sorted(ticks_raw):
        if raw <= 0:
            continue
        tickvals.append(float(np.log10(raw + 1.0)))
        ticktext.append(_format_int(raw))
    return tickvals, ticktext


def _section_header_tight(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="mapa-section-card mapa-section-card--tight">
            <div class="mapa-section-title">{_escape(title)}</div>
            <div class="mapa-section-subtitle">{_escape(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _section_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="mapa-section-card">
            <div class="mapa-section-title">{_escape(title)}</div>
            <div class="mapa-section-subtitle">{_escape(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _major_section_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="mapa-major-section">
            <div class="mapa-major-section-title">{_escape(title)}</div>
            <div class="mapa-major-section-subtitle">{_escape(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _remote_path(filename: str) -> str:
    bucket_url = os.getenv("HF_BUCKET_URL", "").strip().rstrip("/")
    if not bucket_url:
        raise RuntimeError("HF_BUCKET_URL nao foi definido.")
    return f"{bucket_url}/{CANDIDATE_FOLDER}/{filename.lstrip('/')}"


@st.cache_resource(show_spinner=False)
def _hf_fs(token: str | None) -> HfFileSystem:
    return HfFileSystem(token=token or None)


@st.cache_data(show_spinner=False)
def _load_geo_reference(geo_dir: str):
    base = Path(geo_dir)
    geojson_path = base / "geojs-31-mun.json"
    tse_path = base / "municipios_brasileiros_tse.csv"
    municipios_path = base / "municipios.csv"
    municipios_regioes_path = base / "municipios.json"

    if not geojson_path.exists() or not tse_path.exists() or not municipios_path.exists():
        return None, None, None, None

    geojson_mg = json.loads(geojson_path.read_text(encoding="utf-8"))
    df_tse = pd.read_csv(
        tse_path,
        usecols=["codigo_tse", "uf", "nome_municipio", "codigo_ibge"],
    )
    df_tse = df_tse[df_tse["uf"].astype(str).str.upper() == "MG"].copy()
    df_tse["codigo_tse"] = pd.to_numeric(df_tse["codigo_tse"], errors="coerce").astype("Int64")
    df_tse["codigo_ibge"] = pd.to_numeric(df_tse["codigo_ibge"], errors="coerce").astype("Int64")

    df_municipios = pd.read_csv(
        municipios_path,
        usecols=["codigo_ibge", "nome", "latitude", "longitude"],
    )
    df_municipios["codigo_ibge"] = pd.to_numeric(
        df_municipios["codigo_ibge"], errors="coerce"
    ).astype("Int64")

    if municipios_regioes_path.exists():
        raw_regioes = json.loads(municipios_regioes_path.read_text(encoding="utf-8"))
        df_regioes = pd.DataFrame(raw_regioes)
        expected = {
            "municipio-id",
            "municipio-nome",
            "mesorregiao-nome",
            "regiao-imediata-nome",
        }
        if expected.issubset(df_regioes.columns):
            df_regioes = df_regioes[
                [
                    "municipio-id",
                    "municipio-nome",
                    "mesorregiao-nome",
                    "regiao-imediata-nome",
                ]
            ].copy()
            df_regioes = df_regioes.rename(
                columns={
                    "municipio-id": "codigo_ibge",
                    "municipio-nome": "municipio_ibge_nome",
                    "mesorregiao-nome": "mesorregiao_nome",
                    "regiao-imediata-nome": "regiao_imediata_nome",
                }
            )
            df_regioes["codigo_ibge"] = pd.to_numeric(
                df_regioes["codigo_ibge"], errors="coerce"
            ).astype("Int64")
        else:
            df_regioes = pd.DataFrame(
                columns=[
                    "codigo_ibge",
                    "municipio_ibge_nome",
                    "mesorregiao_nome",
                    "regiao_imediata_nome",
                ]
            )
    else:
        df_regioes = pd.DataFrame(
            columns=[
                "codigo_ibge",
                "municipio_ibge_nome",
                "mesorregiao_nome",
                "regiao_imediata_nome",
            ]
        )

    return geojson_mg, df_tse, df_municipios, df_regioes


@st.cache_data(show_spinner="Carregando dados geograficos...", ttl=1800)
def load_geographic_votes(bucket_url: str, token: str | None) -> pd.DataFrame:
    if not bucket_url:
        raise RuntimeError("HF_BUCKET_URL nao foi definido.")

    path = f"{bucket_url.rstrip('/')}/{CANDIDATE_FOLDER}/rawbruto/{GEO_PARQUET}"
    fs = _hf_fs(token)
    with fs.open(path, "rb") as parquet_file:
        df = pd.read_parquet(parquet_file)

    rename_map = {
        "cd_ibge_municipio": "codigo_ibge",
        "cd_municipio": "CD_MUNICIPIO",
        "nm_municipio": "municipio",
        "nm_bairro": "bairro",
        "nr_latitude": "latitude",
        "nr_longitude": "longitude",
        "qt_votos": "votos",
        "nm_mesorregiao": "mesorregiao_nome",
    }
    df = df.rename(columns=rename_map).copy()

    required = {"CD_MUNICIPIO", "codigo_ibge", "municipio", "latitude", "longitude", "votos"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise RuntimeError(f"Colunas ausentes no parquet geografico: {', '.join(missing)}")

    df["CD_MUNICIPIO"] = df["CD_MUNICIPIO"].fillna("").astype(str).str.strip()
    df["codigo_ibge"] = pd.to_numeric(df["codigo_ibge"], errors="coerce").astype("Int64")
    df["municipio"] = df["municipio"].fillna("").astype(str).str.strip()
    df["bairro"] = (
        df["bairro"].fillna("Nao informado").astype(str).str.strip()
        if "bairro" in df.columns
        else "Nao informado"
    )
    df["mesorregiao_nome"] = (
        df["mesorregiao_nome"].fillna("Sem regiao").astype(str).str.strip()
        if "mesorregiao_nome" in df.columns
        else "Sem regiao"
    )
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["votos"] = pd.to_numeric(df["votos"], errors="coerce").fillna(0.0)
    return df.dropna(subset=["latitude", "longitude"])


@st.cache_data(show_spinner="Carregando perfil demografico...", ttl=1800)
def load_demographic_votes(bucket_url: str, token: str | None, profile: str) -> pd.DataFrame:
    if not bucket_url:
        raise RuntimeError("HF_BUCKET_URL nao foi definido.")
    filename = DEMOGRAPHIC_FILES[profile]
    path = f"{bucket_url.rstrip('/')}/{CANDIDATE_FOLDER}/rawbruto/{filename}"
    fs = _hf_fs(token)
    with fs.open(path, "rb") as parquet_file:
        df = pd.read_parquet(parquet_file)

    rename_map = {
        "cd_municipio": "CD_MUNICIPIO",
        "nm_municipio": "municipio",
        "nm_bairro": "bairro",
        "qt_votos": "votos",
        "nm_mesorregiao": "mesorregiao_nome",
    }
    df = df.rename(columns=rename_map).copy()

    for col, fallback in [
        ("CD_MUNICIPIO", ""),
        ("municipio", "Sem município"),
        ("bairro", "Não informado"),
        ("mesorregiao_nome", "Sem região"),
    ]:
        if col not in df.columns:
            df[col] = fallback
        df[col] = df[col].fillna(fallback).astype(str).str.strip()
        if fallback:
            df.loc[df[col].eq(""), col] = fallback

    prefix = DEMOGRAPHIC_PREFIXES[profile]
    value_cols = [col for col in df.columns if col.startswith(prefix)]
    if not value_cols:
        raise RuntimeError(f"Nenhuma coluna demografica encontrada para {profile}.")

    for col in value_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


@st.cache_data(show_spinner="Carregando perfis do eleitorado...", ttl=1800)
def load_profile_elector_votes(bucket_url: str, token: str | None) -> pd.DataFrame:
    if not bucket_url:
        raise RuntimeError("HF_BUCKET_URL nao foi definido.")

    path = f"{bucket_url.rstrip('/')}/{CANDIDATE_FOLDER}/rawbruto/{PROFILE_ELECTOR_PARQUET}"
    fs = _hf_fs(token)
    with fs.open(path, "rb") as parquet_file:
        df = pd.read_parquet(parquet_file)

    rename_map = {
        "cd_municipio": "CD_MUNICIPIO",
        "nm_municipio": "municipio",
        "nm_bairro": "bairro",
        "qt_votos": "votos",
        "nm_mesorregiao": "mesorregiao_nome",
    }
    df = df.rename(columns=rename_map).copy()

    required = {"perfil_eleitor", "votos_candidato"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise RuntimeError(f"Colunas ausentes no parquet de perfil do eleitor: {', '.join(missing)}")

    for col, fallback in [
        ("CD_MUNICIPIO", ""),
        ("municipio", "Sem município"),
        ("bairro", "Não informado"),
        ("mesorregiao_nome", "Sem região"),
    ]:
        if col not in df.columns:
            df[col] = fallback
        df[col] = df[col].fillna(fallback).astype(str).str.strip()
        if fallback:
            df.loc[df[col].eq(""), col] = fallback

    df["perfil_eleitor"] = pd.to_numeric(df["perfil_eleitor"], errors="coerce").astype("Int64")
    df["votos_candidato"] = pd.to_numeric(df["votos_candidato"], errors="coerce").fillna(0.0)
    return df.dropna(subset=["perfil_eleitor"])


@st.cache_data(show_spinner=False, ttl=1800)
def load_candidate_photo(bucket_url: str, token: str | None) -> bytes | None:
    if not bucket_url:
        return None
    path = f"{bucket_url.rstrip('/')}/{CANDIDATE_FOLDER}/fotos_candidatos/{PHOTO_FILENAME}"
    try:
        fs = _hf_fs(token)
        with fs.open(path, "rb") as image_file:
            return image_file.read()
    except Exception:
        return None


def render_candidate_header(photo_bytes: bytes | None) -> None:
    col_foto, col_info = st.columns([1, 4])
    with col_foto:
        if photo_bytes:
            encoded = base64.b64encode(photo_bytes).decode("utf-8")
            st.markdown(
                f"""
                <div class="candidate-photo-card">
                    <img src="data:image/jpeg;base64,{encoded}" alt="{CANDIDATE_NAME}">
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="candidate-photo-card"><span>Foto indisponivel</span></div>',
                unsafe_allow_html=True,
            )
    with col_info:
        st.markdown(
            f"""
            <div class="candidate-info-card">
                <h3>NOME: {_escape(CANDIDATE_FULL_NAME)}</h3>
                <h3>CARGO: {_escape(CANDIDATE_OFFICE)}</h3>
                <h3>PARTIDO: {_escape(CANDIDATE_PARTY)}</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )


def build_municipio_view(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["CD_MUNICIPIO", "codigo_ibge", "municipio"], as_index=False)
        .agg(
            votos=("votos", "sum"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            bairros=("bairro", "nunique"),
            mesorregiao_nome=("mesorregiao_nome", "first"),
        )
        .sort_values("votos", ascending=False)
    )
    grouped["votos_color"] = np.where(
        grouped["votos"] > 0,
        np.log10(grouped["votos"] + 1.0),
        0.0,
    )
    return grouped


def build_vote_map(
    municipio_df: pd.DataFrame,
    geojson_mg: dict,
    df_municipios_ref: pd.DataFrame,
    df_regioes_ref: pd.DataFrame,
):
    mapa_df = municipio_df.copy()
    mapa_df["codigo_ibge"] = pd.to_numeric(
        mapa_df["codigo_ibge"], errors="coerce"
    ).astype("Int64")
    mapa_df["codigo_ibge_str"] = mapa_df["codigo_ibge"].astype(str).str.zfill(7)

    geo_ids = []
    for feature in geojson_mg.get("features", []):
        props = feature.get("properties", {})
        geo_id = str(props.get("id", "")).strip()
        if geo_id:
            geo_ids.append(geo_id.zfill(7))

    malha_df = pd.DataFrame({"codigo_ibge_str": sorted(set(geo_ids))})
    malha_df["codigo_ibge"] = pd.to_numeric(
        malha_df["codigo_ibge_str"], errors="coerce"
    ).astype("Int64")

    mapa_df = malha_df.merge(
        mapa_df[
            [
                "codigo_ibge_str",
                "CD_MUNICIPIO",
                "municipio",
                "votos",
            ]
        ],
        on="codigo_ibge_str",
        how="left",
    )
    mapa_df = mapa_df.merge(df_municipios_ref, on="codigo_ibge", how="left")
    mapa_df = mapa_df.merge(
        df_regioes_ref[["codigo_ibge", "regiao_imediata_nome", "mesorregiao_nome"]],
        on="codigo_ibge",
        how="left",
    )

    mapa_df["votos"] = mapa_df["votos"].fillna(0.0).astype(float)
    mapa_df["votos_color"] = np.where(
        mapa_df["votos"] > 0,
        np.log10(mapa_df["votos"] + 1.0),
        0.0,
    )
    mapa_df["CD_MUNICIPIO"] = mapa_df["CD_MUNICIPIO"].fillna(0).astype(int)
    mapa_df["municipio_exibicao"] = (
        mapa_df["nome"]
        .fillna(mapa_df["municipio"])
        .fillna("Municipio sem voto")
    )
    max_votes = float(mapa_df["votos"].max()) if not mapa_df.empty else 0.0
    zmax = float(np.log10(max_votes + 1.0)) if max_votes > 0 else 1.0
    colorbar_tickvals, colorbar_ticktext = _build_log_colorbar_ticks(max_votes)

    fig_mg = px.choropleth(
        mapa_df,
        geojson=geojson_mg,
        locations="codigo_ibge_str",
        featureidkey="properties.id",
        color="votos_color",
        hover_name="municipio_exibicao",
        hover_data={
            "votos_color": False,
            "CD_MUNICIPIO": True,
            "mesorregiao_nome": True,
            "regiao_imediata_nome": True,
            "latitude": ":.4f",
            "longitude": ":.4f",
            "codigo_ibge_str": False,
        },
        custom_data=["CD_MUNICIPIO", "latitude", "longitude", "votos"],
        color_continuous_scale=[
            [0.00, "#FFFFFF"],
            [0.000001, "#E8F1FF"],
            [0.16, "#BFD9FF"],
            [0.42, "#60A5FA"],
            [0.70, "#2563EB"],
            [1.00, "#0B1F4D"],
        ],
        title="Concentração de votos por município (MG)",
        template="plotly_white",
        range_color=[0.0, zmax],
    )
    fig_mg.update_traces(
        marker_line_color="rgba(210,228,255,0.75)",
        marker_line_width=0.7,
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            "<span style='color:#93c5fd'>Votos:</span> %{customdata[3]:,.0f}<br>"
            "<span style='color:#93c5fd'>Cód. município (TSE):</span> %{customdata[0]}<br>"
            "<span style='color:#93c5fd'>Lat/Lon:</span> %{customdata[1]:.4f}, %{customdata[2]:.4f}<extra></extra>"
        ),
        hoverlabel={
            "bgcolor": "rgba(5,12,28,0.95)",
            "font_color": "#EAF2FF",
            "font_size": 12,
            "bordercolor": "rgba(147,197,253,0.55)",
        },
    )
    fig_mg.update_geos(
        fitbounds="locations",
        visible=False,
        bgcolor="rgba(0,0,0,0)",
    )
    fig_mg.update_layout(
        margin={"l": 6, "r": 36, "t": 52, "b": 6},
        height=560,
        coloraxis_colorbar={
            "title": {"text": "Votos", "font": {"color": "#eaf2ff"}},
            "tickvals": colorbar_tickvals,
            "ticktext": colorbar_ticktext,
            "len": 0.8,
            "thickness": 13,
            "xpad": 8,
            "x": 1.02,
            "xanchor": "left",
            "tickfont": {"color": "#b7c7e6"},
        },
        paper_bgcolor="rgba(255,255,255,0.0)",
        plot_bgcolor="rgba(255,255,255,0.0)",
        font={"color": "#eaf2ff", "family": "Segoe UI, Inter, sans-serif"},
        title={"font": {"size": 20, "color": "#eaf2ff"}},
    )
    return fig_mg


def build_bairro_treemap_view(df: pd.DataFrame, mesorregiao: str) -> pd.DataFrame:
    source = df.copy()
    if mesorregiao != "Todas as regiões":
        source = source[source["mesorregiao_nome"].astype(str).str.strip() == mesorregiao]

    source["bairro"] = source["bairro"].fillna("Não informado").astype(str).str.strip()
    source.loc[source["bairro"].eq(""), "bairro"] = "Não informado"
    source["municipio"] = source["municipio"].fillna("Sem município").astype(str).str.strip()
    source.loc[source["municipio"].eq(""), "municipio"] = "Sem município"

    return (
        source.groupby(["municipio", "bairro"], as_index=False)["votos"]
        .sum()
        .sort_values("votos", ascending=False)
        .head(500)
    )


def build_bairro_treemap_nodes(df_plot: pd.DataFrame) -> pd.DataFrame:
    municipios = (
        df_plot.groupby("municipio", as_index=False)["votos"]
        .sum()
        .sort_values("votos", ascending=False)
    )
    municipio_nodes = pd.DataFrame(
        {
            "node_id": "municipio|" + municipios["municipio"],
            "parent_id": "",
            "label": municipios["municipio"],
            "kind": "municipio",
            "municipio": municipios["municipio"],
            "bairro": "",
            "votos": municipios["votos"],
        }
    )

    bairros = df_plot.copy()
    bairro_nodes = pd.DataFrame(
        {
            "node_id": "bairro|" + bairros["municipio"] + "|" + bairros["bairro"],
            "parent_id": "municipio|" + bairros["municipio"],
            "label": bairros["bairro"],
            "kind": "bairro",
            "municipio": bairros["municipio"],
            "bairro": bairros["bairro"],
            "votos": bairros["votos"],
        }
    )
    return pd.concat([municipio_nodes, bairro_nodes], ignore_index=True)


def _build_vote_treemap(df_plot: pd.DataFrame, title: str):
    nodes = build_bairro_treemap_nodes(df_plot)
    fig = px.treemap(
        nodes,
        names="label",
        ids="node_id",
        parents="parent_id",
        values="votos",
        color="votos",
        custom_data=["kind", "municipio", "bairro", "votos"],
        color_continuous_scale=[
            [0.00, "#1b2640"],
            [0.25, "#1f3f77"],
            [0.50, "#2f7eea"],
            [0.75, "#5aa3ff"],
            [1.00, "#dbeafe"],
        ],
        title=title,
        template="plotly_white",
    )
    fig.update_traces(
        marker=dict(line=dict(width=0.8, color="rgba(255,255,255,0.65)")),
        textfont=dict(size=12),
        root_color="rgba(8,15,28,0.18)",
        branchvalues="total",
        hovertemplate="<b>%{label}</b><br><span style='color:#93c5fd'>Votos:</span> %{customdata[3]:,.0f}<extra></extra>",
    )
    fig.update_layout(
        margin={"l": 8, "r": 8, "t": 54, "b": 8},
        height=500,
        clickmode="event+select",
        paper_bgcolor="rgba(255,255,255,0.0)",
        plot_bgcolor="rgba(255,255,255,0.0)",
        font={"color": "#eaf2ff", "family": "Segoe UI, Inter, sans-serif"},
        title={"font": {"size": 16, "color": "#eaf2ff"}},
    )
    return fig


def _get_selection_points(event) -> list[dict]:
    if event is None:
        return []
    if isinstance(event, dict):
        selection = event.get("selection", {})
        points = selection.get("points", [])
        return list(points or [])
    selection = getattr(event, "selection", None)
    if selection is None:
        return []
    points = getattr(selection, "points", None)
    return list(points or [])


def extract_tree_filter(event) -> dict[str, str] | None:
    points = _get_selection_points(event)
    if not points:
        return None
    point = points[0]
    customdata = point.get("customdata") if isinstance(point, dict) else None
    node_id = str(point.get("id", "")) if isinstance(point, dict) else ""

    if customdata and len(customdata) >= 3:
        kind = str(customdata[0] or "").strip()
        municipio = str(customdata[1] or "").strip()
        bairro = str(customdata[2] or "").strip()
        if kind == "bairro" and municipio and bairro:
            return {"kind": "bairro", "municipio": municipio, "bairro": bairro}
        if kind == "municipio" and municipio:
            return {"kind": "municipio", "municipio": municipio}

    if node_id.startswith("bairro|"):
        _, municipio, bairro = node_id.split("|", 2)
        return {"kind": "bairro", "municipio": municipio, "bairro": bairro}
    if node_id.startswith("municipio|"):
        return {"kind": "municipio", "municipio": node_id.split("|", 1)[1]}
    return None


def _profile_label(raw_column: str, profile: str) -> str:
    prefix = DEMOGRAPHIC_PREFIXES[profile]
    value = raw_column.replace(prefix, "", 1).replace("_", " ").strip()
    replacements = {
        "nao": "não",
        "invalido": "inválido",
        "genero": "gênero",
        "medio": "médio",
        "fundamental": "fundamental",
        "superior": "superior",
        "le e escreve": "lê e escreve",
        "separado judicialmente": "separado judicialmente",
        "viuvo": "viúvo",
    }
    for source, target in replacements.items():
        value = value.replace(source, target)
    return value.capitalize()


def build_demographic_distribution(
    df_profile: pd.DataFrame,
    profile: str,
    mesorregiao: str,
    tree_filter: dict[str, str] | None,
) -> tuple[pd.DataFrame, str]:
    source = df_profile.copy()
    if mesorregiao != "Todas as regiões":
        source = source[source["mesorregiao_nome"].astype(str).str.strip() == mesorregiao]

    scope = mesorregiao
    if tree_filter:
        municipio = tree_filter.get("municipio", "")
        bairro = tree_filter.get("bairro", "")
        if municipio:
            source = source[source["municipio"].astype(str).str.strip() == municipio]
            scope = municipio
        if tree_filter.get("kind") == "bairro" and bairro:
            source = source[source["bairro"].astype(str).str.strip() == bairro]
            scope = f"{bairro} - {municipio}"

    prefix = DEMOGRAPHIC_PREFIXES[profile]
    value_cols = [col for col in source.columns if col.startswith(prefix)]
    distribution = pd.DataFrame(
        {
            "categoria": [_profile_label(col, profile) for col in value_cols],
            "votos": [float(source[col].sum()) for col in value_cols],
        }
    )
    distribution = distribution[distribution["votos"] > 0].copy()
    distribution = distribution.sort_values("votos", ascending=True)
    return distribution, scope


def _dominant_weighted_category(
    source: pd.DataFrame,
    value_cols: list[str],
    profile: str,
    weight_col: str = "votos_candidato",
) -> str:
    total_weight = float(source[weight_col].sum()) if weight_col in source.columns else 0.0
    if total_weight <= 0 or not value_cols:
        return "N/D"

    prefix = DEMOGRAPHIC_PREFIXES[profile]
    denominator = source[value_cols].sum(axis=1).replace(0, np.nan)
    scores = {}
    for col in value_cols:
        share = (source[col] / denominator).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        scores[_profile_label(col, profile)] = float((share * source[weight_col]).sum() / total_weight * 100)

    category, percent = max(scores.items(), key=lambda item: item[1])
    return f"{category} ({_format_percent(percent)})"


def build_profile_elector_cards(
    df_profiles: pd.DataFrame,
    demographic_sources: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    dimension_cols: dict[str, list[str]] = {}
    wide_demographics = None

    for profile, df_demo in demographic_sources.items():
        prefix = DEMOGRAPHIC_PREFIXES[profile]
        value_cols = [col for col in df_demo.columns if col.startswith(prefix)]
        join_keys = [
            key for key in PROFILE_JOIN_KEYS if key in df_profiles.columns and key in df_demo.columns
        ]
        dimension_cols[profile] = value_cols

        if not join_keys or not value_cols:
            continue

        part = df_demo[join_keys + value_cols].copy()
        if wide_demographics is None:
            wide_demographics = part
        else:
            shared_keys = [
                key for key in PROFILE_JOIN_KEYS if key in wide_demographics.columns and key in part.columns
            ]
            wide_demographics = wide_demographics.merge(part, on=shared_keys, how="outer")

    if wide_demographics is None:
        return pd.DataFrame()

    join_keys = [
        key for key in PROFILE_JOIN_KEYS if key in df_profiles.columns and key in wide_demographics.columns
    ]
    source = df_profiles.merge(wide_demographics, on=join_keys, how="left")

    rows = []
    for profile_id, group in source.groupby("perfil_eleitor"):
        profile_id_int = int(profile_id)
        row = {
            "perfil_eleitor": profile_id_int,
            "titulo": PROFILE_ELECTOR_TITLES.get(profile_id_int, f"Perfil {profile_id_int}"),
            "votos": float(group["votos_candidato"].sum()),
        }
        for profile in DEMOGRAPHIC_FILES:
            row[profile] = _dominant_weighted_category(
                group,
                dimension_cols.get(profile, []),
                profile,
            )
        rows.append(row)

    return pd.DataFrame(rows).sort_values("votos", ascending=False)


def render_profile_elector_cards(cards_df: pd.DataFrame) -> None:
    rows = [cards_df.iloc[0:2], cards_df.iloc[2:4]]
    for row_cards in rows:
        cols = st.columns(2, gap="large")
        for col, (_, card) in zip(cols, row_cards.iterrows()):
            with col:
                st.markdown(
                    f"""
                    <div class="perfil-eleitor-card">
                        <div class="perfil-eleitor-title">{_escape(card["titulo"])}</div>
                        <div class="perfil-eleitor-votes-label">Votos</div>
                        <div class="perfil-eleitor-votes">{_format_int(card["votos"])}</div>
                        <div class="perfil-eleitor-row">
                            <div class="perfil-eleitor-row-label">Gênero</div>
                            <div class="perfil-eleitor-row-value">{_escape(card["Gênero"])}</div>
                        </div>
                        <div class="perfil-eleitor-row">
                            <div class="perfil-eleitor-row-label">Idade</div>
                            <div class="perfil-eleitor-row-value">{_escape(card["Idade"])}</div>
                        </div>
                        <div class="perfil-eleitor-row">
                            <div class="perfil-eleitor-row-label">Escolaridade</div>
                            <div class="perfil-eleitor-row-value">{_escape(card["Escolaridade"])}</div>
                        </div>
                        <div class="perfil-eleitor-row">
                            <div class="perfil-eleitor-row-label">Estado civil</div>
                            <div class="perfil-eleitor-row-value">{_escape(card["Estado civil"])}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def build_demographic_heatmap_data(
    df_profiles: pd.DataFrame,
    df_demo: pd.DataFrame,
    selected_variable: str,
    profile_cards: pd.DataFrame,
) -> pd.DataFrame:
    prefix = DEMOGRAPHIC_PREFIXES[selected_variable]
    value_cols = [col for col in df_demo.columns if col.startswith(prefix)]
    join_keys = [
        key for key in PROFILE_JOIN_KEYS if key in df_profiles.columns and key in df_demo.columns
    ]
    if not join_keys or not value_cols or profile_cards.empty:
        return pd.DataFrame()

    source = df_profiles.merge(df_demo[join_keys + value_cols], on=join_keys, how="left")
    title_by_profile = dict(
        zip(profile_cards["perfil_eleitor"].astype(int), profile_cards["titulo"])
    )

    rows = []
    for profile_id, group in source.groupby("perfil_eleitor"):
        profile_id_int = int(profile_id)
        profile_title = title_by_profile.get(
            profile_id_int,
            PROFILE_ELECTOR_TITLES.get(profile_id_int, f"Perfil {profile_id_int}"),
        )
        denominator = group[value_cols].sum(axis=1).replace(0, np.nan)
        weights = group["votos_candidato"].fillna(0.0)
        for col in value_cols:
            share = (group[col] / denominator).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            rows.append(
                {
                    "perfil_eleitor": profile_id_int,
                    "perfil": profile_title,
                    "categoria": _profile_label(col, selected_variable),
                    "votos": float((share * weights).sum()),
                }
            )

    return pd.DataFrame(rows)


def _profile_heatmap_label(title: str) -> str:
    replacements = {
        "Eleitorado de maior escolaridade": "Eleitorado de maior<br>escolaridade",
        "Eleitorado mais velho e popular": "Eleitorado mais velho<br>e popular",
        "Adulto jovem e intermediário": "Adulto jovem<br>e intermediário",
        "Perfil jovem": "Perfil<br>jovem",
    }
    return replacements.get(title, title)


def build_demographic_heatmap_chart(
    heatmap_df: pd.DataFrame,
    selected_variable: str,
    profile_cards: pd.DataFrame,
):
    profile_order = profile_cards["titulo"].tolist()
    category_order = (
        heatmap_df.groupby("categoria", as_index=False)["votos"]
        .sum()
        .sort_values("votos", ascending=False)["categoria"]
        .tolist()
    )
    matrix = (
        heatmap_df.pivot_table(
            index="categoria",
            columns="perfil",
            values="votos",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(index=category_order, columns=profile_order)
        .fillna(0.0)
    )
    text_values = matrix.map(_format_int)

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=[_profile_heatmap_label(title) for title in matrix.columns],
            y=matrix.index.tolist(),
            text=text_values.values,
            texttemplate="%{text}",
            textfont={"color": "#f8fbff", "size": 12},
            colorscale=[
                [0.00, "#0a1224"],
                [0.25, "#142f55"],
                [0.55, "#2563eb"],
                [0.82, "#4f9bff"],
                [1.00, "#93c5fd"],
            ],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "<span style='color:#93c5fd'>Perfil:</span> %{x}<br>"
                "<span style='color:#93c5fd'>Votos:</span> %{z:,.0f}<extra></extra>"
            ),
            colorbar={
                "title": {"text": "Votos", "font": {"color": "#eaf2ff"}},
                "tickfont": {"color": "#b7c7e6"},
                "thickness": 13,
                "len": 0.82,
            },
        )
    )
    fig.update_layout(
        title={
            "text": f"Total de votos por perfil e {selected_variable.lower()}",
            "font": {"size": 16, "color": "#eaf2ff"},
        },
        height=430,
        margin={"l": 8, "r": 8, "t": 52, "b": 8},
        paper_bgcolor="rgba(255,255,255,0.0)",
        plot_bgcolor="rgba(255,255,255,0.0)",
        font={"color": "#eaf2ff", "family": "Segoe UI, Inter, sans-serif"},
        xaxis={
            "title": "",
            "side": "top",
            "tickfont": {"color": "#eaf2ff", "size": 11},
        },
        yaxis={
            "title": "",
            "tickfont": {"color": "#eaf2ff", "size": 12},
            "autorange": "reversed",
        },
    )
    return fig


def render_demographic_composition_card(
    df_profiles: pd.DataFrame,
    demographic_sources: dict[str, pd.DataFrame],
    profile_cards: pd.DataFrame,
) -> None:
    with st.container(key="composicao_demografica_card"):
        col_title, col_filter = st.columns([3, 1], gap="large")
        with col_title:
            st.markdown(
                '<div class="composicao-demografica-title">COMPOSIÇÃO DEMOGRÁFICA</div>',
                unsafe_allow_html=True,
            )
        with col_filter:
            selected_variable = st.selectbox(
                "Variável demográfica",
                list(DEMOGRAPHIC_FILES.keys()),
                key="composicao_demografica_variavel",
            )

        heatmap_df = build_demographic_heatmap_data(
            df_profiles,
            demographic_sources[selected_variable],
            selected_variable,
            profile_cards,
        )
        if heatmap_df.empty:
            st.warning("Sem dados para a composição demográfica selecionada.")
            return

        fig_heatmap = build_demographic_heatmap_chart(
            heatmap_df,
            selected_variable,
            profile_cards,
        )
        st.plotly_chart(fig_heatmap, width="stretch")


def _wrap_plot_label(value: object, max_chars: int = 18) -> str:
    words = str(value or "").split()
    if not words:
        return ""

    lines = []
    current = ""
    for word in words:
        next_value = f"{current} {word}".strip()
        if current and len(next_value) > max_chars:
            lines.append(current)
            current = word
        else:
            current = next_value
    if current:
        lines.append(current)
    return "<br>".join(lines)


def build_top_territory_rank(
    df_geo: pd.DataFrame,
    municipio_df: pd.DataFrame,
    selected_dimension: str,
) -> pd.DataFrame:
    if selected_dimension == "cidades":
        rank = municipio_df[["municipio", "votos"]].copy()
        rank = rank.rename(columns={"municipio": "territorio"})
    else:
        rank = (
            df_geo.groupby("mesorregiao_nome", as_index=False)["votos"]
            .sum()
            .rename(columns={"mesorregiao_nome": "territorio"})
        )

    rank["territorio"] = rank["territorio"].fillna("Não informado").astype(str).str.strip()
    rank.loc[rank["territorio"].eq(""), "territorio"] = "Não informado"
    rank["votos"] = pd.to_numeric(rank["votos"], errors="coerce").fillna(0.0)
    rank = rank.groupby("territorio", as_index=False)["votos"].sum()
    return rank.sort_values("votos", ascending=False).head(5)


def build_top_territory_bar_chart(rank_df: pd.DataFrame, selected_dimension: str):
    plot_df = rank_df.copy()
    plot_df["territorio_plot"] = plot_df["territorio"].map(_wrap_plot_label)
    y_max = float(plot_df["votos"].max()) if not plot_df.empty else 0.0

    fig = px.bar(
        plot_df,
        x="territorio_plot",
        y="votos",
        color="votos",
        color_continuous_scale=[
            [0.00, "#1b2640"],
            [0.30, "#1f3f77"],
            [0.62, "#2f7eea"],
            [1.00, "#dbeafe"],
        ],
        text=plot_df["votos"].map(_format_int),
        custom_data=["territorio", "votos"],
        title=f"Top 5 {selected_dimension}",
        template="plotly_white",
    )
    fig.update_traces(
        marker_line_color="rgba(210,228,255,0.38)",
        marker_line_width=0.8,
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "<span style='color:#93c5fd'>Votos:</span> %{customdata[1]:,.0f}<extra></extra>"
        ),
        hoverlabel={
            "bgcolor": "rgba(5,12,28,0.95)",
            "font_color": "#EAF2FF",
            "font_size": 12,
            "bordercolor": "rgba(147,197,253,0.55)",
        },
    )
    fig.update_layout(
        height=390,
        margin={"l": 8, "r": 18, "t": 52, "b": 16},
        paper_bgcolor="rgba(255,255,255,0.0)",
        plot_bgcolor="rgba(255,255,255,0.0)",
        font={"color": "#eaf2ff", "family": "Segoe UI, Inter, sans-serif"},
        title={"font": {"size": 16, "color": "#eaf2ff"}},
        xaxis={
            "title": "",
            "tickfont": {"color": "#eaf2ff", "size": 11},
            "categoryorder": "array",
            "categoryarray": plot_df["territorio_plot"].tolist(),
        },
        yaxis={
            "title": "",
            "showgrid": True,
            "gridcolor": "rgba(184,208,255,0.14)",
            "zeroline": False,
            "tickfont": {"color": "#b7c7e6"},
            "range": [0, y_max * 1.16 if y_max > 0 else 1],
        },
        coloraxis_showscale=False,
        bargap=0.28,
    )
    return fig


def render_top_cities_mesorregioes_card(
    df_geo: pd.DataFrame,
    municipio_df: pd.DataFrame,
) -> None:
    with st.container(key="top_cidades_mesorregioes_card"):
        col_title, col_filter = st.columns([3, 1], gap="large")
        with col_title:
            st.markdown(
                '<div class="top-territorio-title">Top cidades e Messoregiões</div>',
                unsafe_allow_html=True,
            )
        with col_filter:
            selected_dimension = st.selectbox(
                "Filtro",
                ["Messoregiões", "cidades"],
                key="top_cidades_mesorregioes_filtro",
            )

        rank_df = build_top_territory_rank(df_geo, municipio_df, selected_dimension)
        if rank_df.empty:
            st.warning("Sem dados para o ranking selecionado.")
            return

        fig_rank = build_top_territory_bar_chart(rank_df, selected_dimension)
        st.plotly_chart(fig_rank, width="stretch")


def build_horizontal_bar_chart(distribution: pd.DataFrame, profile: str, scope: str):
    fig = px.bar(
        distribution,
        x="votos",
        y="categoria",
        orientation="h",
        color="votos",
        color_continuous_scale=[
            [0.00, "#1b2640"],
            [0.28, "#1f3f77"],
            [0.58, "#2f7eea"],
            [1.00, "#dbeafe"],
        ],
        text=distribution["votos"].map(lambda value: _format_int(value)),
        title=f"{profile} - {scope}",
        template="plotly_white",
    )
    fig.update_traces(
        marker_line_color="rgba(210,228,255,0.38)",
        marker_line_width=0.8,
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br><span style='color:#93c5fd'>Votos:</span> %{x:,.0f}<extra></extra>",
        hoverlabel={
            "bgcolor": "rgba(5,12,28,0.95)",
            "font_color": "#EAF2FF",
            "font_size": 12,
            "bordercolor": "rgba(147,197,253,0.55)",
        },
    )
    fig.update_layout(
        height=430,
        margin={"l": 8, "r": 56, "t": 54, "b": 20},
        paper_bgcolor="rgba(255,255,255,0.0)",
        plot_bgcolor="rgba(255,255,255,0.0)",
        font={"color": "#eaf2ff", "family": "Segoe UI, Inter, sans-serif"},
        title={"font": {"size": 16, "color": "#eaf2ff"}},
        xaxis={
            "title": "",
            "showgrid": True,
            "gridcolor": "rgba(184,208,255,0.14)",
            "zeroline": False,
            "tickfont": {"color": "#b7c7e6"},
        },
        yaxis={
            "title": "",
            "tickfont": {"color": "#eaf2ff"},
        },
        coloraxis_showscale=False,
        bargap=0.28,
    )
    return fig


load_dotenv(Path(__file__).resolve().parents[1] / ".env")
apply_background()
_apply_page_visual_refinement()

st.title("RaioX Votacao 2022")
st.caption("Analises descritivas geograficas e do perfil do eleitor na ultima eleicao.")

bucket_url = os.getenv("HF_BUCKET_URL", "").strip()
hf_token = os.getenv("HF_TOKEN", "").strip() or None

try:
    df_geo = load_geographic_votes(bucket_url, hf_token)
except Exception as exc:
    st.warning(f"Sem dados para montar o diagnostico do voto. Detalhe: {exc}")
    st.stop()

if df_geo.empty:
    st.warning("Sem dados para montar o diagnostico do voto.")
    st.stop()

render_candidate_header(load_candidate_photo(bucket_url, hf_token))

df_geo["votos"] = df_geo["votos"].fillna(0.0).astype(float)
total_votos = float(df_geo["votos"].sum())

_major_section_header(
    "Mapa territorial da votacao",
    "Primeiro, lemos onde o voto se concentrou no territorio na ultima eleicao.",
)

municipio_source = build_municipio_view(df_geo)
municipio_kpi = municipio_source.sort_values("votos", ascending=False)
if municipio_kpi.empty:
    top_municipio_nome = "N/D"
    top_municipio_votos = 0.0
    total_municipios_votados = 0
else:
    top_municipio_nome = str(municipio_kpi.iloc[0]["municipio"])
    top_municipio_votos = float(municipio_kpi.iloc[0]["votos"])
    total_municipios_votados = int(municipio_kpi["municipio"].nunique())

regiao_rank = (
    df_geo.groupby("mesorregiao_nome", as_index=False)["votos"]
    .sum()
    .sort_values("votos", ascending=False)
)
if regiao_rank.empty:
    top_regiao_nome = "Sem regiao"
    top_regiao_votos = 0.0
else:
    top_regiao_nome = str(regiao_rank.iloc[0]["mesorregiao_nome"] or "Sem regiao")
    top_regiao_votos = float(regiao_rank.iloc[0]["votos"])

col_kpi_1, col_kpi_2, col_kpi_3 = st.columns(3, gap="large")
with col_kpi_1:
    st.markdown(
        f"""
        <div class="mapa-kpi-card">
            <div class="mapa-kpi-label">Total de votos</div>
            <div class="mapa-kpi-value">{_format_int(total_votos)}</div>
            <div class="mapa-kpi-caption">Base total analisada</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_kpi_2:
    st.markdown(
        f"""
        <div class="mapa-kpi-card">
            <div class="mapa-kpi-label mapa-kpi-label--wrap">Municipio mais votado</div>
            <div class="mapa-kpi-value">{_format_int(top_municipio_votos)}</div>
            <div class="mapa-kpi-caption">{_escape(top_municipio_nome)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_kpi_3:
    st.markdown(
        f"""
        <div class="mapa-kpi-card">
            <div class="mapa-kpi-label">Municipios com voto</div>
            <div class="mapa-kpi-value">{_format_int(total_municipios_votados)}</div>
            <div class="mapa-kpi-caption">Distribuicao territorial ativa</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    f"""
    <div class="mapa-kpi-card mapa-kpi-card--featured">
        <div class="mapa-kpi-featured-copy">
            <div class="mapa-kpi-label">Territorio lider</div>
            <div class="mapa-kpi-value mapa-kpi-value--featured">{_escape(top_regiao_nome)}</div>
            <div class="mapa-kpi-caption mapa-kpi-caption--featured">{_format_int(top_regiao_votos)} votos</div>
        </div>
        <div class="mapa-kpi-badge">Maior concentracao</div>
    </div>
    """,
    unsafe_allow_html=True,
)

_section_header_tight(
    "Mapa de Minas Gerais por municipio",
    "Concentracao territorial dos votos por municipio em MG.",
)

geo_dir = Path(__file__).resolve().parents[1] / "geo-mg"
geojson_mg, _df_tse_ref, df_municipios_ref, df_regioes_ref = _load_geo_reference(str(geo_dir))
if geojson_mg is None or df_municipios_ref is None or df_regioes_ref is None:
    st.warning(
        "Arquivos geograficos nao encontrados em geo-mg. Mantenha: geojs-31-mun.json, "
        "municipios_brasileiros_tse.csv, municipios.csv e municipios.json."
    )
    st.stop()

fig_mg = build_vote_map(municipio_source, geojson_mg, df_municipios_ref, df_regioes_ref)
st.plotly_chart(fig_mg, width="stretch")

render_top_cities_mesorregioes_card(df_geo, municipio_source)

_section_header("Votação por Município, Bairro e Perfil demográfico", "")

regioes_options = (
    df_geo.groupby("mesorregiao_nome", as_index=False)["votos"]
    .sum()
    .sort_values("votos", ascending=False)["mesorregiao_nome"]
    .dropna()
    .astype(str)
    .str.strip()
)
regioes_options = [regiao for regiao in regioes_options.tolist() if regiao]
regiao_selected = st.selectbox(
    "Selecione a mesorregião de MG",
    ["Todas as regiões"] + regioes_options,
    key="diag_tree_regiao",
)

treemap_municipio_bairro = build_bairro_treemap_view(df_geo, regiao_selected)
if treemap_municipio_bairro.empty:
    st.warning("Sem dados para o tree map na região selecionada.")
else:
    fig_tree_bairro = _build_vote_treemap(
        treemap_municipio_bairro,
        "Votação Município - Bairro",
    )

    col_left, col_right = st.columns(2, gap="large")
    with col_left:
        with st.container(key="bairro_treemap_card"):
            tree_event = st.plotly_chart(
                fig_tree_bairro,
                width="stretch",
                key="bairro_treemap",
                on_select="rerun",
                selection_mode="points",
            )
    with col_right:
        with st.container(key="perfil_bar_card"):
            profile_selected = st.selectbox(
                "Perfil para detalhar",
                list(DEMOGRAPHIC_FILES.keys()),
                key="perfil_bar_select",
            )
            try:
                df_profile = load_demographic_votes(bucket_url, hf_token, profile_selected)
                tree_filter = extract_tree_filter(tree_event)
                distribution, scope = build_demographic_distribution(
                    df_profile,
                    profile_selected,
                    regiao_selected,
                    tree_filter,
                )
            except Exception as exc:
                st.warning(f"Não foi possível carregar o perfil selecionado. Detalhe: {exc}")
            else:
                if distribution.empty:
                    st.warning("Sem dados para o perfil no recorte selecionado.")
                else:
                    fig_profile = build_horizontal_bar_chart(
                        distribution,
                        profile_selected,
                        scope,
                    )
                    st.plotly_chart(fig_profile, width="stretch")

_major_section_header(
    "Perfil do Eleitor",
    "Detalhamento do eleitorado por território, bairro e recortes demográficos.",
)

try:
    df_profile_elector = load_profile_elector_votes(bucket_url, hf_token)
    demographic_sources = {
        profile: load_demographic_votes(bucket_url, hf_token, profile)
        for profile in DEMOGRAPHIC_FILES
    }
    profile_cards = build_profile_elector_cards(df_profile_elector, demographic_sources)
except Exception as exc:
    st.warning(f"Não foi possível carregar os cards de Perfil do Eleitor. Detalhe: {exc}")
else:
    if profile_cards.empty:
        st.warning("Sem dados para os cards de Perfil do Eleitor.")
    else:
        render_profile_elector_cards(profile_cards)
        render_demographic_composition_card(
            df_profile_elector,
            demographic_sources,
            profile_cards,
        )
