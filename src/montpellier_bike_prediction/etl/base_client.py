# src/montpellier_bike_traffic/etl/base_client.py

from __future__ import annotations

from typing import Any, Optional

import requests


class BaseAPIClient:
    """
    Classe de base pour les clients d'API HTTP.

    Elle gère :
    - la session HTTP
    - la construction des URL
    - la gestion de base des erreurs
    """

    BASE_URL: str = ""

    def __init__(self, base_url: Optional[str] = None, session: Optional[requests.Session] = None) -> None:
        self.base_url = base_url or self.BASE_URL
        self.session = session or requests.Session()

    def _build_url(self, endpoint: str) -> str:
        if endpoint.startswith("http"):
            return endpoint
        return f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"

    def get(self, endpoint: str, params: Optional[dict[str, Any]] = None, timeout: int = 30) -> Any:
        """
        Effectue un GET sur l'endpoint donné et renvoie le JSON décodé.
        Lève une exception en cas de code HTTP != 200.
        """
        url = self._build_url(endpoint)
        resp = self.session.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
