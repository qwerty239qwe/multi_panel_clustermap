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
        df = df.rename(index={idx: f"{new_panel}_celltype_{i}" for i, idx in enumerate(df.index) if "Patient No" not in idx})
    if Path(panel_dir / f"DM{patient_id} quantification.T.xlsx").is_file():
        return
    df.to_excel(panel_dir / f"DM{patient_id} quantification.T.xlsx")


def gen_metadata(panel_dir, choose_from = None):
    panel_dir = Path(panel_dir)
    sample_names = [fn.stem.split(" ")[0] for fn in panel_dir.glob("*.xlsx")]
    if choose_from is None:
        choose_from = {"treatment": ["a", "b", "c", "d"],
                       "sex": ["F", "M"]}

    metadata = pd.DataFrame({k : np.random.choice(v, len(sample_names)) for k, v in choose_from.items()},
                            index=sample_names)

    return metadata


def gen_cluster_metadata(panel_dir="./test_data", choose_from = None):

    panel_dir = Path(__file__).parent.parent / Path(panel_dir)
    cell_names = set()
    cell_to_panel = {}
    for subdir in panel_dir.iterdir():
        for fn in subdir.rglob("*.xlsx"):
            cells = pd.read_excel(fn, index_col=0).index
            cell_names |= set(cells)
            cell_to_panel.update({c: subdir.stem for c in cells})

    if choose_from is None:
        choose_from = {"property": ["a", "b", "c", "d"],}

    metadata = pd.DataFrame({"property": np.random.choice(choose_from["property"], len(cell_names))},
                            index=list(cell_names))
    metadata["panel"] = metadata.index.to_series().map(cell_to_panel)

    return metadata


if __name__ == "__main__":
    gen_cluster_metadata()