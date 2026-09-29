# Regression Benchmark and Forecasting

[![Read the write-up](https://img.shields.io/badge/read-the%20write--up-c2603d)](https://o-2wice.github.io/regression-benchmark-and-forecasting/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/O-2wice/regression-benchmark-and-forecasting/blob/main/notebooks/regression-benchmark-and-forecasting.ipynb)
![Library](https://img.shields.io/badge/library-scikit--learn-f89939)
![Models](https://img.shields.io/badge/models-linear%20%C2%B7%20ridge%20%C2%B7%20forest%20%C2%B7%20SVR-2f6f8f)
![Data](https://img.shields.io/badge/data-synthetic%20%2B%20weather%201940--45-2f6f8f)
![Best R²](https://img.shields.io/badge/R²%20forecast-0.687-c2603d)
![Worst R²](https://img.shields.io/badge/R²%20under%20noise-−24.4-c2603d)

Six regression models pushed until they break, then asked to forecast real
temperatures. The synthetic half is built so the correct answer is known at
every point, which means a wrong prediction can be proved wrong rather than
argued about. The real half removes that guarantee and keeps the same models.

**[Read the write-up](https://o-2wice.github.io/regression-benchmark-and-forecasting/)**
for the reasoning. It opens with a scroll-driven chart that raises the noise
level while four models are refitted at each step, so the collapse happens on
screen rather than in a table.

## The headline

Polynomial regression is the only model in the project to score a perfect
**1.000**, and the only one to score **−24.4**. Same model, same curve, noise
added. A negative R² means the model does worse than ignoring the input and
always answering with the average.

| σ (noise) | Linear | Polynomial | Ridge | Forest |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.888 | **1.000** | 1.000 | 0.986 |
| 20 | 0.316 | −8.426 | 0.257 | 0.441 |
| 75 | −0.775 | **−24.360** | −1.025 | −0.739 |

Ridge never holds a best score anywhere in the project and never records a
disaster either. That trade is the argument.

## What is in it

| Stage | What it establishes |
| --- | --- |
| Three analytical functions | Linear regression scores **−0.019** on a rippled parabola it can see perfectly well, because the shape is symmetric and the best straight line through it is flat |
| Feature engineering | Adding `sin x`, `cos x`, `x²`, `x³` moves polynomial regression from 0.891 to 1.000 and moves random forest by −0.001 |
| Noise sweep | Eight noise levels, four models refitted at each; the fit that was exact degrades fastest |
| Correlated inputs | Every model's MSE **improves** while every model's R² collapses below zero, from identical predictions |
| Uninformative features | A pure-noise column is handed a coefficient of −32.5, next to the real signal's 55.9 |
| SelectKBest vs PCA | Both refuse to help, because redundancy was not the binding problem |
| Noise-aware pipeline | Bayesian Ridge and RidgeCV start lower at 0.839 and end at 0.467 rather than −0.917 |
| Rolling-window forecasting | Daily weather, 1940–45, trained on 1940–44 and tested on 1945 |
| Hyperparameter search | `GridSearchCV` makes it fractionally worse, which is reported rather than buried |

## Forecasting results

Sensor 22508 near Honolulu. The previous *W* days become the inputs, the next
day becomes the target, and the window slides forward one day at a time so the
model never sees a period it will be tested on.

| Model | R² | MSE |
| --- | ---: | ---: |
| Linear Regression | **0.687** | 0.736 |
| Polynomial (deg 2) | 0.680 | 0.754 |
| Random Forest | 0.649 | 0.825 |
| SVR (RBF) | 0.398 | 1.416 |

Linear regression wins here having lost on the synthetic curve. Tomorrow's
temperature mostly resembles today's, and that relationship is close to a
straight line, so the flexible models have no hidden structure to find and
their flexibility only gives them more ways to be wrong.

Across five `TimeSeriesSplit` folds Bayesian Ridge averages about 0.73, with
folds ranging from 0.628 to 0.795. The spread is the more useful number: it
says the answer depends on which stretch of the war you test on.

## Running it

```bash
pip install -r requirements.txt
jupyter lab notebooks/regression-benchmark-and-forecasting.ipynb
```

Everything runs on a laptop CPU in a few minutes. There is nothing to download
by hand and no credentials anywhere: the weather archive is fetched over
`urllib` from a public URL and unpacked with `zipfile`, so the notebook behaves
the same on Windows, macOS, Linux and Colab.

## Repository layout

| Path | Purpose |
| --- | --- |
| [notebooks/regression-benchmark-and-forecasting.ipynb](notebooks/regression-benchmark-and-forecasting.ipynb) | The project notebook, executed end to end |
| [index.qmd](index.qmd) | Source of the write-up |
| [assets/](assets/) | Figures, themes, and the noise-sweep data behind the scrolling chart |
| [scripts/](scripts/) | Exports the sweep data from the notebook's own models |
| [requirements.txt](requirements.txt) | Pinned dependency set the notebook was executed against |

The downloaded archive, generated figures and local environments are not
tracked.

## Notes

The project is deliberately weighted towards failure. A benchmark that only
reports where models succeed teaches you which model to pick once; watching the
same model reach 1.000 and −24.4 on the same curve teaches you what the score
is actually measuring.

Built on Lab #8 of the Data Science Lab course, Politecnico di Torino.

## License

[MIT](LICENSE). The weather archive keeps its own terms.
