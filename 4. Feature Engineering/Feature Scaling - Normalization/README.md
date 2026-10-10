# Normalization (Feature Scaling)

## MinMax Scaling
```python
- Formula : X = (x - xm) / (xM - xm), where xm = minimum and xM = maximum
- Minimum becomes 0, maximum becomes 1, so the output is always bounded in [0, 1]
- Use when a fixed range is needed and the data has no big outliers (min and max are sensitive to outliers)
- sklearn class : MinMaxScaler

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

s = pd.Series([10, 20, 30, 40, 50])
manual = (s - s.min()) / (s.max() - s.min())
print(manual.tolist())

scaled = MinMaxScaler().fit_transform(s.to_frame())
print(scaled.ravel().tolist())

OUTPUT :
[0.0, 0.25, 0.5, 0.75, 1.0]
[0.0, 0.25, 0.5, 0.75, 1.0]
```

## Mean Normalization
```python
- Formula : X = (x - mean) / (xM - xm)
- Only the numerator changes to the mean, the denominator stays max minus min
- Output is centered around 0 and lies roughly in [-1, 1], NOT in [0, 1]
- Rarely used in practice, and sklearn has no ready-made class for it

import pandas as pd

s = pd.Series([10, 20, 30, 40, 50])
mean_norm = (s - s.mean()) / (s.max() - s.min())
print(mean_norm.tolist())

OUTPUT :
[-0.5, -0.25, 0.0, 0.25, 0.5]
```

## MaxAbs Scaling
```python
- Formula : X = x / |max absolute value of the data|
- Divides by the largest absolute value, so the range is [-1, 1]
- Keeps the sign and keeps zeros as zeros, so it is good for sparse data
- It does not center the data
- sklearn class : MaxAbsScaler

import pandas as pd
from sklearn.preprocessing import MaxAbsScaler

t = pd.Series([-50, -20, 0, 10, 40])
manual = t / t.abs().max()
print(manual.tolist())

scaled = MaxAbsScaler().fit_transform(t.to_frame())
print(scaled.ravel().tolist())

OUTPUT :
[-1.0, -0.4, 0.0, 0.2, 0.8]
[-1.0, -0.4, 0.0, 0.2, 0.8]
```

## Robust Scaling
```python
- Formula : X = (x - median) / IQR, where IQR = Q3 - Q1 (75th percentile minus 25th percentile)
- Median and IQR depend on rank, not on extreme values, so outliers do not distort the scale for the other points
- Median becomes 0, and the middle 50% of the data spans one unit of width
- It does NOT remove outliers, the outlier stays far away but the normal points keep a healthy spread
- Output is not bounded to any fixed range
- Use when the data has outliers
- sklearn class : RobustScaler

import pandas as pd
from sklearn.preprocessing import RobustScaler

r = pd.Series([1, 2, 3, 4, 100])
med = r.median()
q1 = r.quantile(0.25)
q3 = r.quantile(0.75)
iqr = q3 - q1
print(med, q1, q3, iqr)

manual = (r - med) / iqr
print(manual.tolist())

scaled = RobustScaler().fit_transform(r.to_frame())
print(scaled.ravel().tolist())

minmax = (r - r.min()) / (r.max() - r.min())
print([round(v, 3) for v in minmax.tolist()])

OUTPUT :
3.0 2.0 4.0 2.0
[-1.0, -0.5, 0.0, 0.5, 48.5]
[-1.0, -0.5, 0.0, 0.5, 48.5]
[0.0, 0.01, 0.02, 0.03, 1.0]
```