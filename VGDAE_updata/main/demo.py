import numpy as np
import scipy
a=np.array([[2,5,8,9],[2,5,8,9]])
a=np.average(a, axis=0)
print(a.shape[0])
cor = np.zeros((a.shape[0], a.shape[0]))
for i in range(a.shape[0]):
    for j in range(a.shape[0]):
        cof = scipy.stats.pearsonr(a[i], a[j])[0]
        cor[i][j] = cof
        print(cof)