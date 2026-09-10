from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "geo-mg" / "bairros"
MASTER_DIR = OUT_ROOT / "geoparquet"
GEOJSON_DIR = OUT_ROOT / "geojson_simplificado"


def _slug(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "sem_nome"


def _required_imports():
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise SystemExit(
            "Dependencia ausente: instale geopandas, pyogrio, shapely e pyarrow "
            "para preparar malhas de bairros."
        ) from exc
    return gpd


def _pick_column(columns: list[str], requested: str | None, candidates: list[str]) -> str:
    if requested:
        if requested not in columns:
            raise SystemExit(f"Coluna informada nao encontrada: {requested}")
        return requested
    normalized = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    raise SystemExit(
        "Nao foi possivel identificar coluna. Informe explicitamente uma das opcoes "
        f"com base nestas colunas: {', '.join(columns)}"
    )


def _standardize_neighborhoods(args: argparse.Namespace):
    gpd = _required_imports()

    source = Path(args.source).resolve()
    if not source.exists():
        raise SystemExit(f"Arquivo de origem nao encontrado: {source}")

    bairros = gpd.read_file(source)
    if bairros.empty:
        raise SystemExit(f"Malha sem registros: {source}")

    municipio_col = _pick_column(
        list(bairros.columns),
        args.municipio_col,
        ["cd_municipio_ibge", "CD_MUN", "CD_GEOCMU", "cod_mun", "geocodigo"],
    )
    bairro_col = _pick_column(
        list(bairros.columns),
        args.bairro_col,
        ["nm_bairro", "NM_BAIRRO", "bairro", "nome", "name"],
    )

    keep = bairros[[municipio_col, bairro_col, bairros.geometry.name]].copy()
    keep = keep.rename(
        columns={
            municipio_col: "cd_municipio_ibge",
            bairro_col: "nm_bairro",
            bairros.geometry.name: "geometry",
        }
    )
    keep["cd_municipio_ibge"] = (
        keep["cd_municipio_ibge"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(7)
    )
    keep["nm_bairro"] = keep["nm_bairro"].astype(str).str.strip()
    keep["fonte"] = args.fonte
    keep["ano"] = args.ano
    keep = keep.set_geometry("geometry")
    if keep.crs is None:
        keep = keep.set_crs(args.input_crs)
    keep = keep.to_crs("EPSG:4326")
    keep = keep[~keep.geometry.is_empty & keep.geometry.notna()].copy()
    keep["geometry"] = keep.geometry.make_valid()

    if args.simplify_tolerance > 0:
        keep["geometry"] = keep.geometry.simplify(
            args.simplify_tolerance,
            preserve_topology=True,
        )

    return keep


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Padroniza malha de bairros e gera GeoParquet mestre + GeoJSON leve por municipio."
    )
    parser.add_argument("source", help="Arquivo de entrada: shp, gpkg, geojson, kml etc.")
    parser.add_argument("--municipio-col", help="Coluna com codigo IBGE de municipio.")
    parser.add_argument("--bairro-col", help="Coluna com nome do bairro.")
    parser.add_argument("--fonte", default="nao_informada", help="Fonte da malha.")
    parser.add_argument("--ano", default="", help="Ano ou versao da malha.")
    parser.add_argument("--input-crs", default="EPSG:4326", help="CRS usado se a origem nao informar.")
    parser.add_argument(
        "--simplify-tolerance",
        type=float,
        default=0.00008,
        help="Tolerancia de simplificacao em graus depois da reprojecao para EPSG:4326.",
    )
    args = parser.parse_args()

    bairros = _standardize_neighborhoods(args)
    MASTER_DIR.mkdir(parents=True, exist_ok=True)
    GEOJSON_DIR.mkdir(parents=True, exist_ok=True)

    master_path = MASTER_DIR / "bairros_mg.parquet"
    bairros.to_parquet(master_path, compression="zstd", index=False)

    written = []
    for municipio, group in bairros.groupby("cd_municipio_ibge"):
        municipio_name = _slug(group["cd_municipio_ibge"].iloc[0])
        out_path = GEOJSON_DIR / f"{municipio}_{municipio_name}.geojson"
        payload = json.loads(group.to_json(drop_id=True))
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        written.append(out_path)

    print(f"geoparquet={master_path}")
    print(f"geojsons={len(written)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
