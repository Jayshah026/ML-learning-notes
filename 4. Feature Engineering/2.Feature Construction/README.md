## Feature Construction

```python
- Feature construction means creating a new, more useful feature by combining or deriving it from existing features, using domain knowledge.
- It isn't limited to summing columns - it can also be a ratio, a difference, or extracting part of existing data (e.g. pulling month out of a date column).
- The new feature captures something the raw columns don't show directly on their own.

import pandas as pd

df = pd.DataFrame({
    'College': ['A', 'B'],
    'Mechanical': [120, 90],
    'Civil': [80, 70],
    'CSE': [150, 130],
    'IT': [100, 60]
})
df['Total_BTech_Students'] = df[['Mechanical', 'Civil', 'CSE', 'IT']].sum(axis=1)
print(df)

OUTPUT :
  College  Mechanical  Civil  CSE   IT  Total_BTech_Students
0       A         120     80  150  100                   450
1       B          90     70  130   60                   350
```