"""Técnicas de PDI. Importar o pacote registra todas as técnicas no registry."""
from laringopdi.pdi import color, enhance, filters, geometry, texture  # noqa: F401
from laringopdi.pdi.registry import DEFAULTS, TECHNIQUES, letterbox, parse, pipeline_hash, run_pipeline  # noqa: F401
