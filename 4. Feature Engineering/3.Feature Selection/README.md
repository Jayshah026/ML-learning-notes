## Feature Selection

```python
- Feature selection means keeping only the existing features that actually help the model, and dropping the rest - the features themselves are not changed or combined, just filtered.
- The main reason is that irrelevant or redundant features add noise and can hurt model performance (overfitting) - reduced storage/computation is just a side benefit, not the goal.
- A simple relevance check is correlation with the target - a feature with near-zero correlation (like a random ID) is a strong candidate to drop.

import pandas as pd

df = pd.DataFrame({
    'Experience': [1, 2, 3, 4, 5],
    'Random_ID': [104, 101, 105, 102, 103],
    'Salary': [30000, 32000, 34000, 36000, 38000]
})
print(df.corr()['Salary'])

df_selected = df.drop(columns=['Random_ID'])
print(df_selected)

OUTPUT :
Experience    1.0
Random_ID    -0.1
Salary        1.0
Name: Salary, dtype: float64

   Experience  Salary
0           1   30000
1           2   32000
2           3   34000
3           4   36000
4           5   38000
```
