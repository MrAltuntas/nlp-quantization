from datetime import datetime

_GENERATED: set[str] = set()


def generate_run_id(cfg) -> str:
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M")
    base = (
        f"{cfg.model.display_name}__{cfg.quantization.type}__"
        f"{cfg.quantization.bit_width}bit__{timestamp}"
    )
    if base not in _GENERATED:
        _GENERATED.add(base)
        return base

    millis = now.microsecond // 1000
    candidate = f"{base}_{millis}"
    while candidate in _GENERATED:
        millis = (millis + 1) % 1000
        candidate = f"{base}_{millis}"
    _GENERATED.add(candidate)
    return candidate
