import json

from incident_risk.config import ROOT


def cell(kind, text):
    base = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        base.update(outputs=[], execution_count=None)
    return base


cells = [
    cell("markdown", "# EDA (train period only)\nSYNTHETIC data. Run top to bottom."),
    cell("code", "from incident_risk import eda\neda.main()"),
    cell(
        "code",
        "from IPython.display import Image, display\nimport glob\n"
        "for f in sorted(glob.glob('../reports/eda/*.png')):\n    display(Image(f))",
    ),
    cell(
        "code", "import pandas as pd\npd.read_csv('../reports/eda/univariate_auc.csv', index_col=0)"
    ),
]
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
(ROOT / "notebooks").mkdir(exist_ok=True)
(ROOT / "notebooks" / "01_eda.ipynb").write_text(json.dumps(nb, indent=1))
print("wrote notebooks/01_eda.ipynb")
