import jax
import sys
from pathlib import Path


def configure_cache(cache_id):
    """configure JAX cache (must be called before importing JAX classes)"""
    cache_dir = Path(sys.modules["__main__"].__file__).parent / f".jax_cache/{cache_id}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)

    print(f"✅ JAX cache configured: {cache_dir}")
