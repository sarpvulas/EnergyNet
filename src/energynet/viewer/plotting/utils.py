from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def week_lines(
    df_idx: Union[pd.Series, pd.DatetimeIndex],
    y_true: Union[pd.Series, np.ndarray],
    y_pred: np.ndarray,
    target: str,
    out_dir: Union[str, Path] = "plots",
    dpi: int = 150,
) -> None:
    """
    Save one PNG per 7‑day window with a single line chart:
    actual vs. predicted (no heat‑maps, no extras).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    df = (
        pd.DataFrame({"actual": np.asarray(y_true), "pred": y_pred}, index=df_idx)
        .sort_index()
    )

    for w, (_, wk) in enumerate(df.groupby(pd.Grouper(freq="7D", label="left"))):
        if wk.empty:
            continue

        plt.figure(figsize=(12, 4), dpi=dpi)
        plt.plot(wk.index, wk["actual"], label="actual", linewidth=1.2)
        plt.plot(wk.index, wk["pred"], label="pred", linewidth=1.2, alpha=0.75)
        plt.title(f"{target} – test week {w}")
        plt.ylabel("kWh")
        plt.legend(frameon=False)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / f"{target}_week{w}.png")
        plt.close()





