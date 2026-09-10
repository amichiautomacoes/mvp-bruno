# Malhas territoriais de bairros

Esta pasta guarda as malhas oficiais ou semioficiais de bairros usadas pelo mapa de potencial.

## Estrutura recomendada

```text
geo-mg/bairros/
|-- README.md
|-- fontes/
|-- geoparquet/
`-- geojson_simplificado/
```

- `fontes/`: arquivos originais recebidos de prefeituras ou fontes publicas.
- `geoparquet/`: base mestre padronizada e comprimida.
- `geojson_simplificado/`: cache leve para o Streamlit/Plotly, separado por municipio.

## Regra de uso no app

Quando existir malha oficial para um municipio, o app deve usar essa geometria de bairro.
Quando nao existir, o fallback continua sendo o bairro aproximado por setores censitarios.

