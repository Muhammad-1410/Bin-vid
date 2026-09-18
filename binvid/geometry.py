from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GridSpec:
    cols: int
    rows: int
    cell_w: int
    cell_h: int
    out_w: int
    out_h: int


def compute_grid(
    src_w: int,
    src_h: int,
    cols: int,
    cell_w: int,
    cell_h: int,
) -> GridSpec:
    """Compute character grid dimensions and output pixel size from source geometry.

    Corrects row count for non-square character cell aspect ratios:
        rows = round(cols * (src_h / src_w) * (cell_w / cell_h))

    Ensures both out_w and out_h are even numbers (as required for H.264 yuv420p)
    by decrementing cols or rows until both dimensions are even without padding.
    """
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"Source dimensions must be positive, got src_w={src_w}, src_h={src_h}.")
    if cols <= 0 or cell_w <= 0 or cell_h <= 0:
        raise ValueError(f"Grid parameters must be positive, got cols={cols}, cell_w={cell_w}, cell_h={cell_h}.")

    rows = max(1, round(cols * (src_h / src_w) * (cell_w / cell_h)))
    out_w = cols * cell_w
    out_h = rows * cell_h

    if (cols * cell_w) % 2:
        cols -= 1
    if (rows * cell_h) % 2:
        rows -= 1
    out_w, out_h = cols * cell_w, rows * cell_h

    return GridSpec(
        cols=cols,
        rows=rows,
        cell_w=cell_w,
        cell_h=cell_h,
        out_w=out_w,
        out_h=out_h,
    )


def fit_cols_to_width(
    src_w: int,
    src_h: int,
    target_out_w: int,
    cell_w: int,
    cell_h: int,
) -> int:
    """Solve for the cols value that gets the output width closest to target_out_w."""
    if target_out_w <= 0:
        raise ValueError(f"target_out_w must be positive, got {target_out_w}.")
    if src_w <= 0 or src_h <= 0 or cell_w <= 0 or cell_h <= 0:
        raise ValueError("All dimensions and cell sizes must be positive.")

    initial_cols = max(1, round(target_out_w / cell_w))
    best_cols = initial_cols
    best_diff = float("inf")

    # Search nearby candidates to account for even-dimension constraints
    search_range = range(max(1, initial_cols - 10), initial_cols + 11)
    for candidate in search_range:
        spec = compute_grid(src_w, src_h, candidate, cell_w, cell_h)
        diff = abs(spec.out_w - target_out_w)
        if diff < best_diff or (diff == best_diff and abs(candidate - initial_cols) < abs(best_cols - initial_cols)):
            best_diff = diff
            best_cols = candidate

    return best_cols
