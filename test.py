import numpy as np
from scipy import stats
rng = np.random.default_rng()
x = stats.norm.rvs(size=500, random_state=rng)
comma_separated = ', '.join(map(str, x))
print(comma_separated)
res = stats.cramervonmises(
    [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 12, 10, 8, 5, 6, 3], 'norm')
print(res)
