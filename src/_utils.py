import pandas as pd
import numpy as np
from pathlib import Path


def gen_dummy(data_path, patient_id, new_panel):
    data_path = Path(data_path)
    parent_dir = data_path.parent.parent
    panel_dir = data_path.parent
    df = pd.read_excel(data_path, index_col=0)
    df = df.loc[df.index[df.index != "Patient No."], df.columns[df.columns != "Patient No."]]

    for c in df.columns:
        df[c] = np.random.random(df.shape[0]) * max(df[c]) * 2
        if "cells" in c:
            df[c] = df[c].astype(int)

    df.loc[:, "Patient No."] = f"DM{patient_id}"
    df.loc["Patient No.", :] = f"DM{patient_id}"
    if new_panel is not None:
        panel_dir = (parent_dir / new_panel)
        panel_dir.mkdir(parents=True, exist_ok=True)
        df = df.rename(index={idx: f"{new_panel}_celltype_{i}" for i, idx in enumerate(df.index)})
    df.to_excel(panel_dir / f"DM{patient_id} quantification.T.xlsx")
