import pandas as pd  # noqa: INP001, D100

from fico.technicalindicators import bollingerbands

data = pd.read_csv("data/data.csv")
close = data["close"]

mean_ewm, upper_band, lower_band = bollingerbands(close, 50, 2)

print(mean_ewm)
print(upper_band)
print(lower_band)
