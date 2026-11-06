"""Técnicas de PDI (escala de cinza). Importar o pacote registra todas as técnicas no registry."""
from laringopdi.pdi import enhance, filters  # noqa: F401
from laringopdi.pdi.registry import (DEFAULTS, TECHNIQUES, letterbox, parse, pipeline_hash,  # noqa: F401
                                     resolve_params, run_pipeline)
