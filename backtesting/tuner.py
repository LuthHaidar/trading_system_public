import itertools
from typing import Any, Dict, Iterable, List, Tuple


def _parse_scalar(token: str) -> Any:
    lowered = token.lower()
    if lowered in {'true', 'false'}:
        return lowered == 'true'

    try:
        if '.' in token:
            return float(token)
        return int(token)
    except ValueError:
        return token


def _expand_range_token(token: str) -> List[Any]:
    """
    Expand range syntax start:end[:step] for numeric values (inclusive of end).

    Examples:
      10:30:10 -> [10, 20, 30]
      0.1:0.3:0.1 -> [0.1, 0.2, 0.3]
      5:1:-2 -> [5, 3, 1]
    """
    parts = [part.strip() for part in token.split(':')]
    if len(parts) not in {2, 3} or any(part == '' for part in parts):
        return [_parse_scalar(token)]

    start = _parse_scalar(parts[0])
    end = _parse_scalar(parts[1])

    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return [_parse_scalar(token)]

    if len(parts) == 3:
        step = _parse_scalar(parts[2])
        if not isinstance(step, (int, float)):
            raise ValueError(f"Invalid range step '{parts[2]}' in token '{token}'")
    else:
        step = 1 if end >= start else -1

    if step == 0:
        raise ValueError(f"Range step cannot be zero in token '{token}'")

    direction = 1 if step > 0 else -1
    if (end - start) * direction < 0:
        raise ValueError(
            f"Range step sign does not move from start to end in token '{token}'"
        )

    values: List[Any] = []
    current = float(start)
    end_f = float(end)
    step_f = float(step)

    epsilon = 1e-12
    if direction > 0:
        while current <= end_f + epsilon:
            values.append(round(current, 12))
            current += step_f
    else:
        while current >= end_f - epsilon:
            values.append(round(current, 12))
            current += step_f

    cast_to_int = all(isinstance(v, int) for v in (start, end, step))
    if cast_to_int:
        return [int(v) for v in values]
    return values


def parse_sweep_values(raw: str) -> List[Any]:
    """Parse comma-separated sweep values into typed Python values."""
    values: List[Any] = []
    for token in raw.split(','):
        token = token.strip()
        if token == '':
            continue
        values.extend(_expand_range_token(token))
    return values


def set_nested_value(target: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set nested dict value from dotted key path."""
    parts = dotted_key.split('.')
    cursor = target
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def build_strategy_overrides(sweep_definitions: Iterable[str]) -> List[Dict[str, Any]]:
    """Build cartesian product of strategy overrides from --sweep definitions."""
    parsed: List[Tuple[str, List[Any]]] = []
    for definition in sweep_definitions:
        if '=' not in definition:
            raise ValueError(f"Invalid sweep definition '{definition}'. Expected key=v1,v2")
        key, raw_values = definition.split('=', 1)
        key = key.strip()
        values = parse_sweep_values(raw_values)
        if not key or not values:
            raise ValueError(f"Invalid sweep definition '{definition}'. Expected key=v1,v2")
        parsed.append((key, values))

    if not parsed:
        return [{}]

    all_overrides: List[Dict[str, Any]] = []
    keys = [k for k, _ in parsed]
    for combo in itertools.product(*(vals for _, vals in parsed)):
        override: Dict[str, Any] = {}
        for key, value in zip(keys, combo):
            set_nested_value(override, key, value)
        all_overrides.append(override)
    return all_overrides


def rank_results(results: List[Dict[str, Any]], metric: str, top_n: int = 5) -> List[Dict[str, Any]]:
    """Return top-n results sorted by chosen metric descending."""

    def metric_value(row: Dict[str, Any]) -> float:
        return float(row.get('metrics', {}).get(metric, float('-inf')))

    ordered = sorted(results, key=metric_value, reverse=True)
    return ordered[:top_n]
