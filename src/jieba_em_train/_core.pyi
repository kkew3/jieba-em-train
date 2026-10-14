def compute_values(
    hat_logprobs: list[float],
    dag_name: str,
    offset: int = 0,
    sz: int = 0,
) -> tuple[list[float], float, float]: ...
def compute_values_parallel(
    hat_logprobs: list[float],
    dag_name: str,
    offsets: list[int],
    sizes: list[int],
) -> tuple[list[float], float, float]: ...
