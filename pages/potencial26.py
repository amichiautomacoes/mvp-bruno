from __future__ import annotations

import base64
import html
import json
import os
import sqlite3
import struct
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import HfFileSystem


st.set_page_config(page_title="Potencial de Votos para 2026", layout="wide")

CANDIDATE_FOLDER = "bruno-raiox22"
OPPORTUNITY_PARQUET = (
    "2022_deputado_estadual_MG_130001598582_bruno_araujo_oportunidade_setor.parquet"
)
CENSUS_GPKG_FILENAME = "MG_setores_CD2022.gpkg"
NAVIGATION_PAGES = {
    "Pagina 1 - RaioX Votacao 2022": "pages/raiox2022.py",
    "Pagina 2 - Potencial de Votos 2026": "pages/potencial26.py",
}
CLASS_ORDER = [
    "Alta concentracao de votos e perfil parecido",
    "Concentracao media de votos e perfil parecido",
    "Pouco voto, mas perfil parecido",
    "Sem concentracao de votos e perfil diferente",
    "Fora do perfil do Bruno",
]
CLASS_COLORS = {
    "Alta concentracao de votos e perfil parecido": "#38761D",
    "Concentracao media de votos e perfil parecido": "#FACC15",
    "Pouco voto, mas perfil parecido": "#8FCE00",
    "Sem concentracao de votos e perfil diferente": "#EF4444",
    "Fora do perfil do Bruno": "#FFFFFF",
}
MUNICIPAL_BLUE_SCALE = [
    "#2986CC",
    "#2478B7",
    "#206BA3",
    "#1C5D8E",
    "#18507A",
    "#144366",
    "#103551",
]
RED_OPPORTUNITY_LABEL = "Sem concentracao de votos e perfil diferente"
MUNICIPAL_BOUNDARY_COLOR = "rgba(5, 12, 28, 0.72)"
MUNICIPAL_BOUNDARY_WIDTH = 0.9
NEIGHBORHOOD_BOUNDARY_COLOR = "rgba(234, 242, 255, 0.56)"
NEIGHBORHOOD_BOUNDARY_WIDTH = 0.48
SECTOR_BOUNDARY_COLOR = "rgba(234, 242, 255, 0.34)"
SECTOR_BOUNDARY_WIDTH = 0.18
SQLITE_IN_CHUNK_SIZE = 900
GEOGRAPHY_OPTIONS = {
    "Bairro aproximado": "bairro",
    "Setor censitario": "setor",
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


def _escape(value: object) -> str:
    return html.escape(str(value or ""))


def _format_int(value: float | int) -> str:
    return f"{int(round(float(value or 0))):,}".replace(",", ".")


def _format_score(value: float | int) -> str:
    return f"{float(value or 0) * 100:.1f}%".replace(".", ",")


def _safe_weighted_average(values: pd.Series, weights: pd.Series) -> float:
    numeric_values = pd.to_numeric(values, errors="coerce").fillna(0.0)
    numeric_weights = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    denominator = float(numeric_weights.sum())
    if denominator > 0:
        return float((numeric_values * numeric_weights).sum() / denominator)
    return float(numeric_values.mean()) if numeric_values.notna().any() else 0.0


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


def _normalize_code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


@st.cache_resource(show_spinner=False)
def _hf_fs(token: str | None) -> HfFileSystem:
    return HfFileSystem(token=token or None)


def _remote_rawpotencial_path(bucket_url: str, filename: str) -> str:
    if not bucket_url:
        raise RuntimeError("HF_BUCKET_URL nao foi definido.")
    return f"{bucket_url.rstrip('/')}/{CANDIDATE_FOLDER}/rawpotencial/{filename}"


def _remote_geo_path(bucket_url: str, filename: str) -> str:
    if not bucket_url:
        raise RuntimeError("HF_BUCKET_URL nao foi definido.")
    return f"{bucket_url.rstrip('/')}/geo-mg/{filename}"


@st.cache_data(show_spinner="Carregando oportunidades no HF...", ttl=1800)
def load_opportunity_data(bucket_url: str, token: str | None) -> pd.DataFrame:
    fs = _hf_fs(token)
    path = _remote_rawpotencial_path(bucket_url, OPPORTUNITY_PARQUET)
    with fs.open(path, "rb") as parquet_file:
        df = pd.read_parquet(parquet_file)

    required = {
        "cd_setor_censitario",
        "qt_votos_setor",
        "perfil_score",
        "classe_oportunidade_label",
        "cor_mapa",
        "cor_mapa_hex",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise RuntimeError("Colunas ausentes no parquet de oportunidade: " + ", ".join(missing))

    df = df.copy()
    df["cd_setor_censitario"] = _normalize_code(df["cd_setor_censitario"])
    for col in [
        "qt_votos_setor",
        "votos_score",
        "perfil_score",
        "perfil_score_genero",
        "perfil_score_idade",
        "perfil_score_escolaridade",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


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


def _sector_ids_for_query(sector_ids: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        sorted({str(sector_id).strip() for sector_id in sector_ids if str(sector_id).strip()})
    )


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


def _geometry_as_multipolygon(geometry: dict) -> list:
    if geometry.get("type") == "Polygon":
        return [geometry.get("coordinates", [])]
    if geometry.get("type") == "MultiPolygon":
        return geometry.get("coordinates", [])
    return []


def _aggregate_to_neighborhoods(
    mapa_df: pd.DataFrame,
    census_geojson: dict,
) -> tuple[dict, pd.DataFrame]:
    df = mapa_df.copy()
    df["nm_bairro_principal"] = df["nm_bairro_principal"].fillna("Fora da base")
    df["nm_municipio_exibicao"] = df["nm_municipio_exibicao"].fillna("Municipio")
    df["bairro_map_id"] = (
        df["cd_municipio_ibge"].astype(str).str.strip()
        + "::"
        + df["nm_bairro_principal"].astype(str).str.strip().str.upper()
    )

    geometry_by_sector = {
        str(feature.get("properties", {}).get("CD_SETOR", "")).strip(): feature.get("geometry")
        for feature in census_geojson.get("features", [])
    }
    polygons_by_bairro: dict[str, list] = {}
    for row in df[["bairro_map_id", "cd_setor_censitario"]].itertuples(index=False):
        geometry = geometry_by_sector.get(str(row.cd_setor_censitario).strip())
        if geometry:
            polygons_by_bairro.setdefault(row.bairro_map_id, []).extend(
                _geometry_as_multipolygon(geometry)
            )

    group_cols = [
        "bairro_map_id",
        "cd_municipio_ibge",
        "nm_municipio_exibicao",
        "nm_bairro_principal",
    ]
    score_cols = [
        "perfil_score",
        "perfil_score_genero",
        "perfil_score_idade",
        "perfil_score_escolaridade",
        "votos_score",
    ]
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["qt_votos_setor"] = float(group["qt_votos_setor"].sum())
        row["setores_no_bairro"] = int(group["cd_setor_censitario"].nunique())
        row["cd_setores_exemplo"] = ", ".join(
            group["cd_setor_censitario"].astype(str).sort_values().head(6).tolist()
        )
        for col in score_cols:
            row[col] = _safe_weighted_average(group[col], group["qt_votos_setor"])
        row["concentracao_votos"] = (
            group.sort_values("qt_votos_setor", ascending=False)["concentracao_votos"].iloc[0]
        )
        row["titulo_perfil_dominante"] = (
            group.sort_values("qt_votos_setor", ascending=False)["titulo_perfil_dominante"].iloc[0]
        )
        class_votes = group.groupby("classe_oportunidade_label", dropna=False)[
            "qt_votos_setor"
        ].sum()
        row["classe_oportunidade_label"] = str(class_votes.idxmax())
        row["cor_mapa_hex"] = CLASS_COLORS.get(row["classe_oportunidade_label"], "#FFFFFF")
        rows.append(row)

    bairros_df = pd.DataFrame(rows)
    features = []
    for bairro_id, polygons in polygons_by_bairro.items():
        if not polygons:
            continue
        features.append(
            {
                "type": "Feature",
                "id": bairro_id,
                "properties": {"BAIRRO_ID": bairro_id},
                "geometry": {"type": "MultiPolygon", "coordinates": polygons},
            }
        )

    return {"type": "FeatureCollection", "features": features}, bairros_df


@st.cache_data(show_spinner=False, ttl=86400)
def load_census_sector_count(bucket_url: str, token: str | None) -> int:
    gpkg_path = _ensure_census_gpkg(bucket_url, token)
    con = sqlite3.connect(gpkg_path)
    total = int(con.execute("SELECT COUNT(*) FROM MG_setores_CD2022").fetchone()[0])
    con.close()
    return total


@st.cache_data(show_spinner="Convertendo setores de oportunidade para o mapa...", ttl=86400)
def load_census_geojson(
    bucket_url: str,
    token: str | None,
    sector_ids: tuple[str, ...],
) -> tuple[dict, pd.DataFrame]:
    gpkg_path = _ensure_census_gpkg(bucket_url, token)
    requested_ids = _sector_ids_for_query(sector_ids)
    if not requested_ids:
        return {"type": "FeatureCollection", "features": []}, pd.DataFrame(
            columns=[
                "cd_setor_censitario",
                "cd_municipio_ibge",
                "municipio_ibge",
                "situacao_setor_ibge_malha",
            ]
        )

    con = sqlite3.connect(gpkg_path)
    rows = []
    for start in range(0, len(requested_ids), SQLITE_IN_CHUNK_SIZE):
        chunk = requested_ids[start : start + SQLITE_IN_CHUNK_SIZE]
        placeholders = ",".join("?" for _ in chunk)
        rows.extend(
            con.execute(
                f"""
                SELECT CD_SETOR, CD_MUN, NM_MUN, SITUACAO, geom
                FROM MG_setores_CD2022
                WHERE CAST(CD_SETOR AS TEXT) IN ({placeholders})
                """,
                chunk,
            ).fetchall()
        )
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
                "cd_municipio_ibge": str(cd_mun).strip(),
                "municipio_ibge": str(nm_mun).strip(),
                "situacao_setor_ibge_malha": str(situacao).strip(),
            }
        )

    return {"type": "FeatureCollection", "features": features}, pd.DataFrame(sectors)


@st.cache_data(show_spinner=False)
def load_municipal_geojson(geo_dir: str) -> dict:
    path = Path(geo_dir) / "geojs-31-mun.json"
    if not path.exists():
        raise RuntimeError("Arquivo geo-mg/geojs-31-mun.json nao encontrado.")
    return json.loads(path.read_text(encoding="utf-8"))


def _municipal_score_by_sector(mapa_df: pd.DataFrame) -> dict[str, float]:
    if mapa_df.empty:
        return {}

    scored = mapa_df.copy()
    scored["cd_municipio_ibge"] = scored["cd_municipio_ibge"].astype(str).str.strip().str.zfill(7)
    scored["is_red_sector"] = scored["classe_oportunidade_label"].eq(RED_OPPORTUNITY_LABEL)
    by_city = scored.groupby("cd_municipio_ibge", dropna=False).agg(
        setores_classificados=("cd_setor_censitario", "nunique"),
        setores_vermelhos=("is_red_sector", "sum"),
    )
    by_city["score_municipal"] = (
        by_city["setores_classificados"] - (by_city["setores_vermelhos"] * 2)
    ).clip(lower=0)
    max_score = float(by_city["score_municipal"].max())
    if max_score <= 0:
        by_city["score_normalizado"] = 0.0
    else:
        by_city["score_normalizado"] = by_city["score_municipal"] / max_score
    return by_city["score_normalizado"].to_dict()


def _municipal_colorscale() -> list[list[object]]:
    max_index = len(MUNICIPAL_BLUE_SCALE) - 1
    return [[index / max_index, color] for index, color in enumerate(MUNICIPAL_BLUE_SCALE)]


def _municipal_fill_trace(municipal_geojson: dict, mapa_df: pd.DataFrame) -> go.Choropleth:
    municipal_scores = _municipal_score_by_sector(mapa_df)
    locations = []
    names = []
    scores = []
    for feature in municipal_geojson.get("features", []):
        properties = feature.get("properties") or {}
        geo_id = str(properties.get("id", "")).strip()
        if geo_id:
            location = geo_id.zfill(7)
            locations.append(location)
            names.append(
                str(properties.get("name") or properties.get("description") or "Municipio")
            )
            scores.append(float(municipal_scores.get(location, 0.0)))

    return go.Choropleth(
        geojson=municipal_geojson,
        locations=locations,
        z=scores,
        text=names,
        featureidkey="properties.id",
        zmin=0,
        zmax=1,
        colorscale=_municipal_colorscale(),
        marker_line_color=MUNICIPAL_BOUNDARY_COLOR,
        marker_line_width=MUNICIPAL_BOUNDARY_WIDTH,
        opacity=0.68,
        showscale=False,
        hovertemplate=(
            "<b>%{text}</b><br>"
            "<span style='color:#93c5fd'>Intensidade territorial:</span> %{z:.0%}"
            "<extra></extra>"
        ),
        hoverlabel={
            "bgcolor": "rgba(5,12,28,0.95)",
            "font_color": "#EAF2FF",
            "font_size": 12,
            "bordercolor": "rgba(147,197,253,0.55)",
        },
        showlegend=False,
        name="Municipios",
    )


def build_opportunity_map(
    sectors_df: pd.DataFrame,
    census_geojson: dict,
    opportunity_df: pd.DataFrame,
    municipal_geojson: dict,
    geography: str,
):
    sectors = sectors_df.copy()
    sectors["cd_setor_censitario"] = _normalize_code(sectors["cd_setor_censitario"])
    opportunity = opportunity_df.copy()
    opportunity["cd_setor_censitario"] = _normalize_code(opportunity["cd_setor_censitario"])

    keep_cols = [
        "cd_setor_censitario",
        "nm_municipio",
        "nm_bairro_principal",
        "bairros_no_setor",
        "qt_votos_setor",
        "votos_score",
        "perfil_score",
        "perfil_score_genero",
        "perfil_score_idade",
        "perfil_score_escolaridade",
        "concentracao_votos",
        "classe_oportunidade_label",
        "cor_mapa",
        "cor_mapa_hex",
        "perfil_eleitor_dominante",
        "titulo_perfil_dominante",
    ]
    keep_cols = [col for col in keep_cols if col in opportunity.columns]

    mapa_df = sectors.merge(opportunity[keep_cols], on="cd_setor_censitario", how="left")
    mapa_df["classe_oportunidade_label"] = mapa_df["classe_oportunidade_label"].fillna(
        "Fora do perfil do Bruno"
    )
    mapa_df["cor_mapa"] = mapa_df["cor_mapa"].fillna("branco")
    mapa_df["cor_mapa_hex"] = mapa_df["cor_mapa_hex"].fillna("#FFFFFF")
    mapa_df["nm_municipio_exibicao"] = (
        mapa_df["nm_municipio"].fillna(mapa_df["municipio_ibge"]).fillna("Municipio")
    )
    mapa_df["nm_bairro_principal"] = mapa_df["nm_bairro_principal"].fillna("Fora da base")
    mapa_df["bairros_no_setor"] = mapa_df["bairros_no_setor"].fillna("Fora da base")
    mapa_df["concentracao_votos"] = mapa_df["concentracao_votos"].fillna("sem votos")
    mapa_df["titulo_perfil_dominante"] = mapa_df["titulo_perfil_dominante"].fillna("N/D")
    mapa_df["perfil_eleitor_dominante"] = mapa_df["perfil_eleitor_dominante"].fillna(0)
    for col in [
        "qt_votos_setor",
        "votos_score",
        "perfil_score",
        "perfil_score_genero",
        "perfil_score_idade",
        "perfil_score_escolaridade",
    ]:
        if col not in mapa_df.columns:
            mapa_df[col] = 0.0
        mapa_df[col] = pd.to_numeric(mapa_df[col], errors="coerce").fillna(0.0)

    if geography == "bairro":
        plot_geojson, plot_df = _aggregate_to_neighborhoods(mapa_df, census_geojson)
        locations_col = "bairro_map_id"
        feature_id_key = "properties.BAIRRO_ID"
        hover_name = "nm_bairro_principal"
        custom_data = [
            "nm_municipio_exibicao",
            "setores_no_bairro",
            "cd_setores_exemplo",
            "qt_votos_setor",
            "perfil_score",
            "perfil_score_genero",
            "perfil_score_idade",
            "perfil_score_escolaridade",
            "concentracao_votos",
            "titulo_perfil_dominante",
            "classe_oportunidade_label",
        ]
        title = "Oportunidade eleitoral por bairro aproximado"
        hovertemplate = (
            "<b>%{hovertext}</b><br>"
            "<span style='color:#93c5fd'>Municipio:</span> %{customdata[0]}<br>"
            "<span style='color:#93c5fd'>Setores agrupados:</span> %{customdata[1]}<br>"
            "<span style='color:#93c5fd'>Exemplos de setores:</span> %{customdata[2]}<br>"
            "<span style='color:#93c5fd'>Votos no bairro:</span> %{customdata[3]:,.0f}<br>"
            "<span style='color:#93c5fd'>Similaridade:</span> %{customdata[4]:.1%}<br>"
            "<span style='color:#93c5fd'>Genero:</span> %{customdata[5]:.1%} | "
            "<span style='color:#93c5fd'>Idade:</span> %{customdata[6]:.1%} | "
            "<span style='color:#93c5fd'>Escolaridade:</span> %{customdata[7]:.1%}<br>"
            "<span style='color:#93c5fd'>Concentracao:</span> %{customdata[8]}<br>"
            "<span style='color:#93c5fd'>Perfil dominante:</span> %{customdata[9]}<br>"
            "<span style='color:#93c5fd'>Classe:</span> %{customdata[10]}<extra></extra>"
        )
        marker_line_width = NEIGHBORHOOD_BOUNDARY_WIDTH
        marker_line_color = NEIGHBORHOOD_BOUNDARY_COLOR
    else:
        plot_geojson = census_geojson
        plot_df = mapa_df
        locations_col = "cd_setor_censitario"
        feature_id_key = "properties.CD_SETOR"
        hover_name = "nm_municipio_exibicao"
        custom_data = [
            "cd_setor_censitario",
            "nm_bairro_principal",
            "bairros_no_setor",
            "qt_votos_setor",
            "perfil_score",
            "perfil_score_genero",
            "perfil_score_idade",
            "perfil_score_escolaridade",
            "concentracao_votos",
            "titulo_perfil_dominante",
            "classe_oportunidade_label",
        ]
        title = "Oportunidade eleitoral por setor censitario"
        hovertemplate = (
            "<b>%{hovertext}</b><br>"
            "<span style='color:#93c5fd'>Setor:</span> %{customdata[0]}<br>"
            "<span style='color:#93c5fd'>Bairro principal:</span> %{customdata[1]}<br>"
            "<span style='color:#93c5fd'>Bairros no setor:</span> %{customdata[2]}<br>"
            "<span style='color:#93c5fd'>Votos no setor:</span> %{customdata[3]:,.0f}<br>"
            "<span style='color:#93c5fd'>Similaridade:</span> %{customdata[4]:.1%}<br>"
            "<span style='color:#93c5fd'>Genero:</span> %{customdata[5]:.1%} | "
            "<span style='color:#93c5fd'>Idade:</span> %{customdata[6]:.1%} | "
            "<span style='color:#93c5fd'>Escolaridade:</span> %{customdata[7]:.1%}<br>"
            "<span style='color:#93c5fd'>Concentracao:</span> %{customdata[8]}<br>"
            "<span style='color:#93c5fd'>Perfil dominante:</span> %{customdata[9]}<br>"
            "<span style='color:#93c5fd'>Classe:</span> %{customdata[10]}<extra></extra>"
        )
        marker_line_width = SECTOR_BOUNDARY_WIDTH
        marker_line_color = SECTOR_BOUNDARY_COLOR

    fig = px.choropleth(
        plot_df,
        geojson=plot_geojson,
        locations=locations_col,
        featureidkey=feature_id_key,
        color="classe_oportunidade_label",
        color_discrete_map=CLASS_COLORS,
        category_orders={"classe_oportunidade_label": CLASS_ORDER},
        hover_name=hover_name,
        custom_data=custom_data,
        title=title,
        template="plotly_white",
    )
    fig.update_traces(
        marker_line_color=marker_line_color,
        marker_line_width=marker_line_width,
        hovertemplate=hovertemplate,
        hoverlabel={
            "bgcolor": "rgba(5,12,28,0.95)",
            "font_color": "#EAF2FF",
            "font_size": 12,
            "bordercolor": "rgba(147,197,253,0.55)",
        },
    )
    fig.add_trace(_municipal_fill_trace(municipal_geojson, mapa_df))
    fig.data = (fig.data[-1],) + fig.data[:-1]
    fig.update_geos(fitbounds="locations", visible=False, bgcolor="rgba(0,0,0,0)")
    fig.update_layout(
        height=720,
        margin={"l": 6, "r": 10, "t": 58, "b": 6},
        paper_bgcolor="rgba(255,255,255,0.0)",
        plot_bgcolor="rgba(255,255,255,0.0)",
        font={"color": "#eaf2ff", "family": "Segoe UI, Inter, sans-serif"},
        title={"font": {"size": 20, "color": "#eaf2ff"}},
        legend={
            "title": {"text": "Territorios"},
            "font": {"color": "#eaf2ff", "size": 12},
            "bgcolor": "rgba(5,12,28,0.64)",
            "bordercolor": "rgba(184,208,255,0.20)",
            "borderwidth": 1,
        },
    )
    return fig, mapa_df, plot_df


load_dotenv(Path(__file__).resolve().parents[1] / ".env")
apply_background()
_apply_page_visual_refinement()
render_sidebar_navigation("pages/potencial26.py")

st.title("Potencial de Votos para 2026")
st.caption("Mapa de oportunidades eleitorais com visualizacao por bairro aproximado ou setor.")

bucket_url = os.getenv("HF_BUCKET_URL", "").strip()
hf_token = os.getenv("HF_TOKEN", "").strip() or None

_major_section_header(
    "Mapa de oportunidade territorial",
    "A visualizacao por bairro agrupa setores censitarios pelo bairro principal da base. A visao por setor continua disponivel para detalhe fino.",
)

geography_label = st.segmented_control(
    "Granularidade do mapa",
    options=list(GEOGRAPHY_OPTIONS.keys()),
    default="Bairro aproximado",
)
geography = GEOGRAPHY_OPTIONS[geography_label or "Bairro aproximado"]

try:
    opportunity_df = load_opportunity_data(bucket_url, hf_token)
    opportunity_sector_ids = tuple(
        sorted(opportunity_df["cd_setor_censitario"].dropna().astype(str).str.strip().unique())
    )
    total_census_sectors = load_census_sector_count(bucket_url, hf_token)
    census_geojson, sectors_df = load_census_geojson(
        bucket_url,
        hf_token,
        opportunity_sector_ids,
    )
    municipal_geojson = load_municipal_geojson(str(Path(__file__).resolve().parents[1] / "geo-mg"))
except Exception as exc:
    st.warning(f"Nao foi possivel carregar o mapa de oportunidade. Detalhe: {exc}")
    st.stop()

fig_opportunity, mapa_df, plot_df = build_opportunity_map(
    sectors_df,
    census_geojson,
    opportunity_df,
    municipal_geojson,
    geography,
)

classified_territories = int(len(plot_df))
total_sectors = int(total_census_sectors)
total_votes = float(opportunity_df["qt_votos_setor"].sum())
avg_similarity = float(opportunity_df["perfil_score"].mean()) if not opportunity_df.empty else 0.0
territory_label = "Bairros no mapa" if geography == "bairro" else "Setores classificados"
territory_caption = (
    "Agrupados pelo bairro principal dos setores"
    if geography == "bairro"
    else "Territorios conectados ao perfil de voto"
)

col_kpi_1, col_kpi_2, col_kpi_3, col_kpi_4 = st.columns(4, gap="large")
with col_kpi_1:
    st.markdown(
        f"""
        <div class="mapa-kpi-card">
            <div class="mapa-kpi-label">Setores na malha IBGE</div>
            <div class="mapa-kpi-value">{_format_int(total_sectors)}</div>
            <div class="mapa-kpi-caption">Cobertura territorial completa de MG</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_kpi_2:
    st.markdown(
        f"""
        <div class="mapa-kpi-card">
            <div class="mapa-kpi-label">{_escape(territory_label)}</div>
            <div class="mapa-kpi-value">{_format_int(classified_territories)}</div>
            <div class="mapa-kpi-caption">{_escape(territory_caption)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_kpi_3:
    st.markdown(
        f"""
        <div class="mapa-kpi-card">
            <div class="mapa-kpi-label">Votos analisados</div>
            <div class="mapa-kpi-value">{_format_int(total_votes)}</div>
            <div class="mapa-kpi-caption">Total usado na matriz de oportunidade</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_kpi_4:
    st.markdown(
        f"""
        <div class="mapa-kpi-card">
            <div class="mapa-kpi-label">Similaridade media</div>
            <div class="mapa-kpi-value">{_format_score(avg_similarity)}</div>
            <div class="mapa-kpi-caption">Genero, idade e escolaridade combinados</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.plotly_chart(fig_opportunity, width="stretch")
