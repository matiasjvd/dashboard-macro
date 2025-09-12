# Dashboard Macro por País (Streamlit)

Este proyecto trae el dashboard desde `quant-allocation/dashboard/app.py` y lo adapta para correr en esta carpeta.

## Estructura
- `app.py`: aplicación principal Streamlit
- `services/`: servicios auxiliares (noticias)
- `scripts/`: scripts utilitarios para extraer/actualizar data desde otra carpeta
- `data/`: aquí se guarda `macro_data.db` (o archivos CSV/Parquet si eliges ese flujo)
- `.streamlit/secrets.toml`: variables de API (Finnhub/Perplexity) y configuración

## Requisitos
- Python 3.10+
- Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Correr local
```bash
streamlit run app.py
```

## Deploy en Streamlit Cloud
1) Subí este repo a GitHub (sin `scripts/set_env.sh` ni `.streamlit/secrets.toml`, ya están ignorados).
2) Antes de deploy, asegura un archivo de datos en `data/macro_data.parquet` (está permitido en git para Cloud):
   - Opción A: extrae subset aquí:
     ```bash
     python scripts/extract_data.py \
       --source-db "/Users/matias/Desktop/Proyectos/quant-allocation/data/macro_data.db" \
       --countries "Argentina,Chile" \
       --metrics "CPI,Unemployment Rate,Index Price" \
       --start 2015-01-01 --end 2025-12-31 \
       --format parquet
     ```
   - Opción B: copia tu parquet:
     ```bash
     python scripts/refresh_db.py --source "/ruta/a/macro_data.parquet" --target parquet
     ```
3) Commit y push (incluyendo `data/macro_data.parquet`).
4) En Streamlit Cloud:
   - Configura en “Secrets” (UI de Streamlit Cloud) el contenido de `.streamlit/secrets.example.toml` con tus claves.
   - No es necesario subir `.streamlit/secrets.toml` real.
5) Setea el comando principal: `streamlit run app.py`.

Notas:
- La app detecta `data/macro_data.parquet` automáticamente en Cloud.
- Si no subís datos, la app mostrará el aviso para generar/traer el subset.

## Flujo de datos
1) NO se versiona la BBDD completa. Puedes traer datos puntuales desde la otra carpeta con:

```bash
python scripts/extract_data.py --source-db \
  "/Users/matias/Desktop/Proyectos/quant-allocation/data/macro_data.db" \
  --countries "Argentina,Chile" \
  --metrics "CPI,Unemployment Rate,Index Price" \
  --start 2015-01-01 --end 2025-12-31 \
  --format sqlite
```

- `--format sqlite`: guarda/actualiza `data/macro_data.db` aquí (mismo esquema esperado por el dashboard)
- Alternativamente, usa `--format parquet` o `--format csv`. El `app.py` soporta SQLite y Parquet por defecto.
- La ejecución es manual (no automática), ideal para despliegue en Streamlit Cloud.

2) Para Streamlit Cloud, sube los archivos resultantes a `data/` en el repo (p.ej. `macro_data.db` o `macro_data.parquet`).

## Variables y secretos
No se versionan llaves. Usa variables de entorno o `.streamlit/secrets.toml` local (ignorado por git).

- Ejemplo de secrets: `.streamlit/secrets.example.toml` (cópialo a `.streamlit/secrets.toml` y completa claves)
- Ejemplo de entorno: `scripts/set_env_example.sh` (cópialo a `scripts/set_env.sh`, edítalo y carga con `source scripts/set_env.sh`)

Si no configuras las keys, la app funciona pero sin briefing de Perplexity y con noticias limitadas.

## Refrescar data (reemplazo directo)
Para copiar el archivo de la otra carpeta y reemplazar el local:

```bash
python scripts/refresh_db.py \
  --source "/Users/matias/Desktop/Proyectos/quant-allocation/data/macro_data.db" \
  --target sqlite
```

O si usas Parquet:

```bash
python scripts/refresh_db.py \
  --source "/Users/matias/Desktop/Proyectos/quant-allocation/data/macro_data.parquet" \
  --target parquet
```

## Notas
- El dashboard autodetecta si existe `data/macro_data.db` o `data/macro_data.parquet`.
- La actualización de datos es manual vía `scripts/extract_data.py` o `scripts/refresh_db.py`.