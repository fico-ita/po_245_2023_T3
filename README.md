# FICO T3

FICO is a multidisciplinary interest group in Computational Finance and Systematic
Investments operating.

## Project

On this repository, you will find the code for the **Meta-Labeling Application** for financial

## Usage

### Installation

```bash
# Activate the virtual environment
cd project_folder
poetry shell

# Add the project packages
poetry add git+https://github.com/fico-ita/po_245_2023_T3.git

# Install the project packages
poetry install
```

### Requirements

Python 3.11 or higher is required. The project requires the following packages:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

### Example

```python
>>> import pandas as pd
>>> from fico.technicalindicators import bollingerbandssignal
>>> close = pd.read_csv("data/data.csv")["close"]
>>> df = bollingerbandssignal(close, 50, 1)
>>> df.head()
   close   ewm_mean      upper      lower  label  side_long  side_short
0  74.09  74.090000        NaN        NaN    NaN        NaN         NaN
1  77.03  75.589400  77.668294  73.510506    NaN        NaN         NaN
2  78.06  76.446090  78.493703  74.398477    NaN        NaN         NaN
3  79.91  77.364705  79.785135  74.944274   -1.0        NaN         NaN
4  82.22  78.414970  81.465249  75.364691   -1.0        NaN         NaN
```

## Documentation

The documentation is available on [GitHub](
    http://127.0.0.1:8000/
    )

## License

[Apache License 2.0](LICENSE)

## Citation

Since this project is research based, it is important to cite it in your work.
To cite this project, use the following reference:

### BibTeX
```bibtex
@misc{lima2023finance,
    author = {Lima, R. D. C.},
    title = {Finance Machine Learning: Meta-Labeling Application},
    year = {2023},
    DOI = {10.5281/zenodo.0000000},
    publisher = {Zenodo},
    url = {https://doi.org/10.5281/zenodo.0000000}
}
```
### APA
```text
Lima, R.D.C.(2023), Finance Machine Learning: Meta-Labeling Application.
Zenodo. https://doi.org/10.5281/zenodo.0000000
```
