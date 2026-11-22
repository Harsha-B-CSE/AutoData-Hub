import json
import hashlib
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

class PipelineCache:
    """
    Manages content-hash and parameter-hash caching to enable smart skipping
    of redundant processing steps.
    """
    _cache_index: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def generate_cache_key(asset_sha256: str, engine_type: str, parameters: Dict[str, Any]) -> str:
        serialized_params = json.dumps(parameters, sort_keys=True)
        key_input = f"{asset_sha256}:{engine_type}:{serialized_params}"
        return hashlib.sha256(key_input.encode("utf-8")).hexdigest()

    @classmethod
    def get(cls, cache_key: str) -> Optional[Dict[str, Any]]:
        return cls._cache_index.get(cache_key)

    @classmethod
    def set(cls, cache_key: str, result_metadata: Dict[str, Any]):
        cls._cache_index[cache_key] = result_metadata

    @classmethod
    def should_skip(cls, asset_sha256: str, engine_type: str, parameters: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        cache_key = cls.generate_cache_key(asset_sha256, engine_type, parameters)
        cached = cls.get(cache_key)
        if cached:
            out_path = cached.get("output_path")
            if out_path and Path(out_path).exists():
                return True, cached
        return False, None
