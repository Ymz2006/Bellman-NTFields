# Sharp features + midpoint loss

## 22×22

| spec | seeds | success without → with midpoint | with midpoint vs octave + midpoint | midpoint gain | uphill without → with | minima without → with |
|---|---|---|---|---|---|---|
| `octave` | 3 | 46.3 → 54.9 % | +0.0 | +8.6 | 2.2 → 0.5 % | 0.77 → 0.21 |
| `Agtri@allx2` | 3 | 41.9 → 44.2 % | -10.7 | +2.3 | 1.3 → 1.1 % | 1.02 → 0.89 |
| `Agtri@topx8` | 1 | 32.4 → 35.7 % | -26.0 | +3.3 | 14.6 → 14.7 % | 7.81 → 8.93 |
| `Apow16@axis` | 1 | 39.2 → 44.6 % | -17.1 | +5.4 | 3.1 → 2.4 % | 1.44 → 1.00 |
| `Apowmix@all` | 3 | 29.2 → 32.3 % | -22.6 | +3.1 | 18.9 → 17.9 % | 8.74 → 7.40 |
| `Asc@new128` | 3 | 48.6 → 58.1 % | +3.2 | +9.5 | 1.4 → 0.5 % | 0.52 → 0.21 |
| `Rpowmix@half` | 3 | 32.4 → 35.0 % | -20.0 | +2.6 | 13.7 → 12.7 % | 4.61 → 4.95 |
| `Rtri@all` | 3 | 38.8 → 41.5 % | -13.4 | +2.7 | 2.6 → 1.8 % | 3.18 → 2.23 |
