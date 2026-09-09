from __future__ import annotations

import base64
import html
import os
import sqlite3
import struct
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import HfFileSystem


st.set_page_config(page_title="Potencial de Votos para 2026", layout="wide")

CANDIDATE_FOLDER = "bruno-raiox22"
NAVIGATION_PAGES = {
    "Pagina 1 - RaioX Votacao 2022": "pages/raiox2022.py",
    "Pagina 2 - Potencial de Votos 2026": "pages/potencial26.py",
}
CENSUS_GPKG_FILENAME = "MG_setores_CD2022.gpkg"
RAWIBGE_FILES = {
    "Idade": "df_idade_mg.parquet",
    "Genero": "df_sexo_genero_mg.parquet",
    "Escolaridade": "escolaridade.parquet",
}
SCHOOLING_COLUMNS = {
    "Total 25+": "total_25_mais",
    "Sem instrucao / fundamental incompleto": "sem_instrucao_fundamental_incompleto",
    "Fundamental completo / medio incompleto": "fundamental_completo_medio_incompleto",
    "Medio completo / superior incompleto": "medio_completo_superior_incompleto",
    "Superior completo": "superior_completo",
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
        @media (max-width: 680px) {
            .mapa-major-section-title {
                font-size: 1.55rem;
            }
        }
        .mapa-kpi-card {
            min-height: 8.4rem;
            padding: 1rem 1.05rem;
            border: 1px solid rgba(184, 208, 255, 0.24);
            border-radius: 18px;
            background: linear-gradient(
                145deg,
                rgba(7, 18, 36, 0.70) 0%,
                rgba(7, 18, 36, 0.52) 100%
            );
            box-shadow: 0 18px 40px rgba(2, 9, 24, 0.42);
            backdrop-filter: blur(6px);
            -webkit-backdrop-filter: blur(6px);
        }
        .mapa-kpi-label {
            color: #9eb6da;
            font-size: 0.78rem;
            font-weight: 760;
            letter-spacing: 0.08em;
            line-height: 1.2;
            text-transform: uppercase;
        }
        .mapa-kpi-value {
            color: #f8fbff;
            font-size: 2rem;
            font-weight: 830;
            line-height: 1.05;
            margin-top: 0.48rem;
        }
        .mapa-kpi-caption {
            color: #c7d6ee;
            font-size: 0.9rem;
            line-height: 1.32;
            margin-top: 0.44rem;
        }
        .stSelectbox label {
            color: #dbeafe !important;
            font-weight: 720 !important;
        }
        [data-testid="stPlotlyChart"] {
            padding: 0.78rem;
            border: 1px solid rgba(184, 208, 255, 0.22);
            border-radius: 18px;
            background: linear-gradient(
                145deg,
                rgba(7, 18, 36, 0.70) 0%,
                rgba(7, 18, 36, 0.46) 100%
            );
            box-shadow: 0 18px 40px rgba(2, 9, 24, 0.36);
            backdrop-filter: blur(6px);
            -webkit-backdrop-filter: blur(6px);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _major_section_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="mapa-major-section">
            <div class="mapa-major-section-title">{title}</div>
            <div class="mapa-major-section-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _escape(value: object) -> str:
    return html.escape(str(value or ""))


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


def _format_int(value: float | int) -> str:
    return f"{int(round(float(value or 0))):,}".replace(",", ".")


def _normalize_code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def _build_log_colorbar_ticks(max_value: float) -> tuple[list[float], list[str]]:
    max_v = float(max_value or 0.0)
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


@st.cache_resource(show_spinner=False)
def _hf_fs(token: str | None) -> HfFileSystem:
    return HfFileSystem(token=token or None)


def _remote_rawibge_path(bucket_url: str, filename: str) -> str:
    if not bucket_url:
        raise RuntimeError("HF_BUCKET_URL nao foi definido.")
    return f"{bucket_url.rstrip('/')}/{CANDIDATE_FOLDER}/rawibge/{filename}"


def _remote_geo_path(bucket_url: str, filename: str) -> str:
    if not bucket_url:
        raise RuntimeError("HF_BUCKET_URL nao foi definido.")
    return f"{bucket_url.rstrip('/')}/geo-mg/{filename}"


@st.cache_data(show_spinner="Carregando dados do IBGE no HF...", ttl=1800)
def load_rawibge_dataset(bucket_url: str, token: str | None, dataset: str) -> pd.DataFrame:
    filename = RAWIBGE_FILES[dataset]
    fs = _hf_fs(token)
    with fs.open(_remote_rawibge_path(bucket_url, filename), "rb") as parquet_file:
        return pd.read_parquet(parquet_file)


@st.cache_data(show_spinner="Baixando malha censitaria do IBGE no HF...", ttl=86400)
def _ensure_census_gpkg(bucket_url: str, token: str | None) -> str:
    cache_dir = Path(tempfile.gettempdir()) / "mktpolitica_geo"
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / CENSUS_GPKG_FILENAME
    if local_path.exists() and local_path.stat().st_size > 100_000_000:
        return str(local_path)

    fs = _hf_fs(token)
    fs.get(_remote_geo_path(bucket_url, CENSUS_GPKG_FILENAME), str(local_path))
    return str(local_path)


def _gpkg_wkb_offset(blob: bytes) -> int:
    if blob[:2] != b"GP":
        return 0
    flags = blob[3]
    envelope_code = (flags >> 1) & 0b111
    envelope_bytes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}.get(envelope_code, 0)
    return 8 + envelope_bytes


def _thin_ring(coords: list[list[float]], max_points: int = 72) -> list[list[float]]:
    if len(coords) <= max_points:
        return coords
    step = max(1, int(np.ceil(len(coords) / max_points)))
    thinned = coords[::step]
    if thinned[0] != coords[-1]:
        thinned.append(coords[-1])
    return thinned


def _read_wkb_geometry(blob: bytes, offset: int = 0) -> tuple[dict | None, int]:
    byte_order = blob[offset]
    endian = "<" if byte_order == 1 else ">"
    geom_type = struct.unpack_from(f"{endian}I", blob, offset + 1)[0] % 1000
    cursor = offset + 5

    if geom_type == 3:
        rings_count = struct.unpack_from(f"{endian}I", blob, cursor)[0]
        cursor += 4
        rings = []
        for _ in range(rings_count):
            points_count = struct.unpack_from(f"{endian}I", blob, cursor)[0]
            cursor += 4
            coords = []
            for _ in range(points_count):
                x, y = struct.unpack_from(f"{endian}dd", blob, cursor)
                cursor += 16
                coords.append([x, y])
            rings.append(_thin_ring(coords))
        return {"type": "Polygon", "coordinates": rings}, cursor

    if geom_type == 6:
        polygons_count = struct.unpack_from(f"{endian}I", blob, cursor)[0]
        cursor += 4
        polygons = []
        for _ in range(polygons_count):
            polygon, cursor = _read_wkb_geometry(blob, cursor)
            if polygon and polygon["type"] == "Polygon":
                polygons.append(polygon["coordinates"])
        return {"type": "MultiPolygon", "coordinates": polygons}, cursor

    return None, len(blob)


def _gpkg_geometry_to_geojson(blob: bytes) -> dict | None:
    offset = _gpkg_wkb_offset(blob)
    geometry, _cursor = _read_wkb_geometry(blob, offset)
    return geometry


@st.cache_data(show_spinner="Convertendo setores censitarios para o mapa...", ttl=86400)
def load_census_geojson(bucket_url: str, token: str | None) -> tuple[dict, pd.DataFrame]:
    gpkg_path = _ensure_census_gpkg(bucket_url, token)
    con = sqlite3.connect(gpkg_path)
    rows = con.execute(
        """
        SELECT CD_SETOR, CD_MUN, NM_MUN, SITUACAO, geom
        FROM MG_setores_CD2022
        """
    ).fetchall()
    con.close()

    features = []
    sectors = []
    for cd_setor, cd_mun, nm_mun, situacao, geom_blob in rows:
        geometry = _gpkg_geometry_to_geojson(geom_blob)
        if geometry is None:
            continue
        cd_setor_text = str(cd_setor).strip()
        features.append(
            {
                "type": "Feature",
                "id": cd_setor_text,
                "properties": {"CD_SETOR": cd_setor_text},
                "geometry": geometry,
            }
        )
        sectors.append(
            {
                "cd_setor_censitario": cd_setor_text,
                "cd_municipio": str(cd_mun).strip(),
                "municipio": str(nm_mun).strip(),
                "situacao": str(situacao).strip(),
            }
        )

    return {"type": "FeatureCollection", "features": features}, pd.DataFrame(sectors)


def build_sector_metric(df: pd.DataFrame, dataset: str, category: str) -> tuple[pd.DataFrame, str]:
    source = df.copy()
    source["cd_setor_censitario"] = _normalize_code(source["cd_setor_censitario"])

    if dataset == "Idade":
        if category != "Total 15+":
            source = source[source["faixa_etaria"].astype(str).str.strip() == category]
        source["valor"] = pd.to_numeric(source["populacao"], errors="coerce").fillna(0.0)
        label = "Populacao" if category == "Total 15+" else f"Populacao - {category}"

    elif dataset == "Genero":
        if category != "Total":
            source = source[source["sexo"].astype(str).str.strip().str.lower() == category.lower()]
        source["valor"] = pd.to_numeric(source["populacao"], errors="coerce").fillna(0.0)
        label = "Populacao" if category == "Total" else f"Populacao - {category.capitalize()}"

    else:
        value_col = SCHOOLING_COLUMNS[category]
        source["valor"] = pd.to_numeric(source[value_col], errors="coerce").fillna(0.0)
        label = f"Populacao - {category}"

    metric = source.groupby("cd_setor_censitario", as_index=False)["valor"].sum()
    return metric, label


def build_census_map(
    sectors_df: pd.DataFrame,
    geojson: dict,
    metric_df: pd.DataFrame,
    metric_label: str,
):
    mapa_df = sectors_df.merge(metric_df, on="cd_setor_censitario", how="left")
    mapa_df["valor"] = mapa_df["valor"].fillna(0.0).astype(float)
    mapa_df["valor_color"] = np.where(
        mapa_df["valor"] > 0,
        np.log10(mapa_df["valor"] + 1.0),
        0.0,
    )

    max_value = float(mapa_df["valor"].max()) if not mapa_df.empty else 0.0
    zmax = float(np.log10(max_value + 1.0)) if max_value > 0 else 1.0
    colorbar_tickvals, colorbar_ticktext = _build_log_colorbar_ticks(max_value)

    fig = px.choropleth(
        mapa_df,
        geojson=geojson,
        locations="cd_setor_censitario",
        featureidkey="properties.CD_SETOR",
        color="valor_color",
        hover_name="municipio",
        hover_data={
            "valor_color": False,
            "cd_setor_censitario": True,
            "cd_municipio": True,
            "situacao": True,
            "valor": ":,.0f",
        },
        custom_data=["cd_setor_censitario", "cd_municipio", "situacao", "valor"],
        color_continuous_scale=[
            [0.00, "#FFFFFF"],
            [0.000001, "#E8F1FF"],
            [0.18, "#BFD9FF"],
            [0.44, "#60A5FA"],
            [0.72, "#2563EB"],
            [1.00, "#0B1F4D"],
        ],
        range_color=[0.0, zmax],
        title=f"Minas Gerais por setor censitario - {metric_label}",
        template="plotly_white",
    )
    fig.update_traces(
        marker_line_color="rgba(210,228,255,0.30)",
        marker_line_width=0.18,
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            "<span style='color:#93c5fd'>Setor:</span> %{customdata[0]}<br>"
            "<span style='color:#93c5fd'>Municipio IBGE:</span> %{customdata[1]}<br>"
            "<span style='color:#93c5fd'>Situacao:</span> %{customdata[2]}<br>"
            "<span style='color:#93c5fd'>Valor:</span> %{customdata[3]:,.0f}<extra></extra>"
        ),
        hoverlabel={
            "bgcolor": "rgba(5,12,28,0.95)",
            "font_color": "#EAF2FF",
            "font_size": 12,
            "bordercolor": "rgba(147,197,253,0.55)",
        },
    )
    fig.update_geos(fitbounds="locations", visible=False, bgcolor="rgba(0,0,0,0)")
    fig.update_layout(
        margin={"l": 6, "r": 36, "t": 58, "b": 6},
        height=680,
        coloraxis_colorbar={
            "title": {"text": metric_label, "font": {"color": "#eaf2ff"}},
            "tickvals": colorbar_tickvals,
            "ticktext": colorbar_ticktext,
            "len": 0.82,
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
    return fig


load_dotenv(Path(__file__).resolve().parents[1] / ".env")
apply_background()
_apply_page_visual_refinement()
render_sidebar_navigation("pages/potencial26.py")

st.title("Potencial de Votos para 2026")
st.caption("Analises de potencial eleitoral para a proxima eleicao.")

bucket_url = os.getenv("HF_BUCKET_URL", "").strip()
hf_token = os.getenv("HF_TOKEN", "").strip() or None

_major_section_header(
    "Potencial demográfico",
    "Mapa censitario de Minas Gerais com dados demograficos do IBGE carregados da pasta rawibge no HF.",
)

dataset = st.selectbox(
    "Base demografica",
    list(RAWIBGE_FILES.keys()),
    key="potencial_demografico_dataset",
)

try:
    df_rawibge = load_rawibge_dataset(bucket_url, hf_token, dataset)
except Exception as exc:
    st.warning(f"Nao foi possivel carregar a base rawibge no HF. Detalhe: {exc}")
    st.stop()

if dataset == "Idade":
    categories = ["Total 15+"] + sorted(
        df_rawibge["faixa_etaria"].dropna().astype(str).str.strip().unique().tolist()
    )
elif dataset == "Genero":
    categories = ["Total"] + sorted(
        df_rawibge["sexo"].dropna().astype(str).str.strip().str.lower().unique().tolist()
    )
else:
    categories = list(SCHOOLING_COLUMNS.keys())

category = st.selectbox(
    "Recorte",
    categories,
    key="potencial_demografico_category",
)

try:
    census_geojson, sectors_df = load_census_geojson(bucket_url, hf_token)
except Exception as exc:
    st.warning(f"Nao foi possivel carregar a malha censitaria do HF. Detalhe: {exc}")
    st.stop()

metric_df, metric_label = build_sector_metric(df_rawibge, dataset, category)
mapped_sectors = int(metric_df["cd_setor_censitario"].nunique())
total_value = float(metric_df["valor"].sum())

col_kpi_1, col_kpi_2, col_kpi_3 = st.columns(3, gap="large")
with col_kpi_1:
    st.markdown(
        f"""
        <div class="mapa-kpi-card">
            <div class="mapa-kpi-label">Setores na malha IBGE</div>
            <div class="mapa-kpi-value">{_format_int(len(sectors_df))}</div>
            <div class="mapa-kpi-caption">Divisoes censitarias renderizadas no mapa</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_kpi_2:
    st.markdown(
        f"""
        <div class="mapa-kpi-card">
            <div class="mapa-kpi-label">Setores com dado rawibge</div>
            <div class="mapa-kpi-value">{_format_int(mapped_sectors)}</div>
            <div class="mapa-kpi-caption">{_escape(dataset)} - {_escape(category)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_kpi_3:
    st.markdown(
        f"""
        <div class="mapa-kpi-card">
            <div class="mapa-kpi-label">Total do recorte</div>
            <div class="mapa-kpi-value">{_format_int(total_value)}</div>
            <div class="mapa-kpi-caption">{_escape(metric_label)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

fig_census = build_census_map(sectors_df, census_geojson, metric_df, metric_label)
st.plotly_chart(fig_census, width="stretch")
