import os
import requests
import pandas as pd
from typing import List

# Finnhub-based news fetcher (basic)

# Mapeo de países a códigos ISO de 2 letras para Finnhub
COUNTRY_CODE_MAP = {
    "United States": "US",
    "USA": "US",
    "Estados Unidos": "US",
    "Brazil": "BR",
    "Brasil": "BR",
    "Mexico": "MX",
    "México": "MX",
    "Argentina": "AR",
    "Chile": "CL",
    "Colombia": "CO",
    "Peru": "PE",
    "Perú": "PE",
    "Canada": "CA",
    "Canadá": "CA",
    "United Kingdom": "GB",
    "UK": "GB",
    "Reino Unido": "GB",
    "Germany": "DE",
    "Alemania": "DE",
    "France": "FR",
    "Francia": "FR",
    "Italy": "IT",
    "Italia": "IT",
    "Spain": "ES",
    "España": "ES",
    "Japan": "JP",
    "Japón": "JP",
    "China": "CN",
    "India": "IN",
    "Australia": "AU",
    "South Korea": "KR",
    "Corea del Sur": "KR",
    "Russia": "RU",
    "Rusia": "RU",
    "Turkey": "TR",
    "Turquía": "TR",
    "South Africa": "ZA",
    "Sudáfrica": "ZA",
}

def get_country_code(country: str) -> str:
    """Obtiene el código ISO de 2 letras para un país."""
    # Buscar en el mapeo (case-insensitive)
    country_clean = country.strip()
    for key, code in COUNTRY_CODE_MAP.items():
        if key.lower() == country_clean.lower():
            return code
    # Si no se encuentra, intentar usar las primeras 2 letras en mayúsculas
    return country_clean[:2].upper() if len(country_clean) >= 2 else "US"

def fetch_news_finnhub(country: str, api_key: str | None = None, max_items: int = 50) -> List[dict]:
    api_key = api_key or os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        return []
    
    # Obtener código del país
    country_code = get_country_code(country)
    
    try:
        # Usar el endpoint de noticias por país de Finnhub
        # Formato: /news?category=general&minId=0
        # Para noticias específicas del país, usamos el parámetro de búsqueda
        url = f"https://finnhub.io/api/v1/news?category=general"
        headers = {"X-Finnhub-Token": api_key}
        
        # Intentar primero con noticias generales y filtrar
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        items = r.json() if isinstance(r.json(), list) else []
        
        # Filtrar por país en el contenido
        filtered_items = []
        country_keywords = [country.lower(), country_code.lower()]
        
        for item in items:
            title = (item.get("headline") or item.get("title") or "").lower()
            summary = (item.get("summary") or item.get("description") or "").lower()
            source = (item.get("source") or "").lower()
            
            # Verificar si el país está mencionado en título, resumen o fuente
            if any(keyword in title or keyword in summary or keyword in source for keyword in country_keywords):
                filtered_items.append(item)
        
        # Si no hay suficientes noticias filtradas, intentar con company news del país
        if len(filtered_items) < 5:
            # Buscar noticias de mercado del país usando el endpoint de company news
            # con símbolos de índices principales del país
            market_symbols = {
                "US": "^GSPC",  # S&P 500
                "BR": "^BVSP",  # Bovespa
                "MX": "^MXX",   # IPC Mexico
                "AR": "MERV",   # Merval
                "GB": "^FTSE",  # FTSE 100
                "DE": "^GDAXI", # DAX
                "FR": "^FCHI",  # CAC 40
                "JP": "^N225",  # Nikkei
                "CN": "000001.SS", # Shanghai
            }
            
            symbol = market_symbols.get(country_code)
            if symbol:
                try:
                    company_url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={(pd.Timestamp.now() - pd.Timedelta(days=14)).strftime('%Y-%m-%d')}&to={pd.Timestamp.now().strftime('%Y-%m-%d')}"
                    r2 = requests.get(company_url, headers=headers, timeout=15)
                    if r2.status_code == 200:
                        company_items = r2.json() if isinstance(r2.json(), list) else []
                        filtered_items.extend(company_items[:max_items - len(filtered_items)])
                except Exception:
                    pass
        
        return filtered_items[:max_items] if filtered_items else items[:max_items]
    except Exception:
        return []


def filter_news_by_country(items: List[dict], country: str) -> List[dict]:
    """
    Filtro adicional por país (ya no es tan necesario porque fetch_news_finnhub 
    ya filtra, pero se mantiene para compatibilidad y refinamiento adicional).
    """
    if not items:
        return []
    
    # Obtener código del país y keywords
    country_code = get_country_code(country)
    keywords = [country.strip().casefold(), country_code.casefold()]
    
    # Agregar variaciones del nombre del país
    country_variations = {
        "Brazil": ["brazil", "brasil", "brazilian"],
        "Brasil": ["brazil", "brasil", "brazilian"],
        "Mexico": ["mexico", "méxico", "mexican"],
        "México": ["mexico", "méxico", "mexican"],
        "United States": ["united states", "usa", "us", "american"],
        "Argentina": ["argentina", "argentinian", "argentino"],
    }
    
    country_clean = country.strip()
    for key, variations in country_variations.items():
        if key.lower() == country_clean.lower():
            keywords.extend(variations)
            break
    
    out = []
    for it in items:
        title = (it.get("headline") or it.get("title") or "").casefold()
        summary = (it.get("summary") or it.get("description") or "").casefold()
        source = (it.get("source") or "").casefold()
        
        # Verificar si alguna keyword está en título, resumen o fuente
        if any(keyword in title or keyword in summary or keyword in source for keyword in keywords):
            out.append(it)
    
    # Si no hay resultados filtrados, devolver los items originales
    return out if out else items