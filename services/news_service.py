import os
import requests
from typing import List

# Finnhub-based news fetcher (basic)

def fetch_news_finnhub(country: str, api_key: str | None = None, max_items: int = 50) -> List[dict]:
    api_key = api_key or os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        return []
    # Use category or query by country name; Finnhub supports /news?category=
    # Here we fallback to general news and filter by country later.
    try:
        url = "https://finnhub.io/api/v1/news?category=general"
        headers = {"X-Finnhub-Token": api_key}
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        items = r.json() if isinstance(r.json(), list) else []
        return items[:max_items]
    except Exception:
        return []


def filter_news_by_country(items: List[dict], country: str) -> List[dict]:
    if not items:
        return []
    # simple heuristic: title or summary contains country name (casefold)
    key = (country or "").strip().casefold()
    if not key:
        return items
    out = []
    for it in items:
        title = (it.get("headline") or it.get("title") or "").casefold()
        summary = (it.get("summary") or it.get("description") or "").casefold()
        if key in title or key in summary:
            out.append(it)
    return out or items