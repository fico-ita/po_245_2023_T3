import pandas as pd  # noqa: INP001, D100

from fico.technicalindicators import bollingerbandssignal

data = pd.read_csv("data/data.csv")
close = data["close"]

df = bollingerbandssignal(close, 50, 1)  # noqa: PD901

df.head()

df.loc[(df.side_short == 1) | (df.side_long == 1)].head()
