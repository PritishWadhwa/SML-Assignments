# %%
from tqdm import tqdm
import numpy as np
from numpy.linalg import eig
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.style.use('dark_background')

# %% [markdown]
# ### Part c

# %%
A = np.array([[2, 6], [5, 1]]).T
A

# %%
M = np.mean(A.T, axis=1)
M

# %%
C = A - M
C

# %%
V = np.cov(C.T)
V

# %%
values, vectors = eig(V)

# %%
values, vectors

# %%
Y = np.dot(vectors.T, C)
Y

# %%
newMat = (np.dot(vectors, Y) + M).T
newMat

# %%
mse = np.mean(np.square(newMat - A.T))
mse

# %% [markdown]
# Yes my calculation match with the code, my calculation gives a perfect 0 mse while my code gives mse which is very very close to 0. The small difference is because of the way numbers are stored in python.

# %% [markdown]
# ### Part d
# assuming d < n

# %%
d = 10
n = 1000

# %%
mean = np.random.rand(d)
mat = np.random.rand(d, d)
cov = np.dot(mat, mat.T)

# %%
X = np.random.multivariate_normal(mean, cov, n).T

# %%
X.shape

# %% [markdown]
# ### Part e

# %%


def getEigenvectors(X):
    """
    Returns the eigenvectors of the input matrix X
    """
    C = X-X.mean(axis=1)[:, np.newaxis]
    V = np.cov(C)
    vals, vecs = eig(V)
    list = []
    for i in range(len(vals)):
        list.append((vals[i], vecs[i]))
    list = sorted(list, reverse=True)
    eigenvectors = [i[1] for i in list]
    eigenvectors = np.array(eigenvectors)
    return eigenvectors


# %% [markdown]
# ### Part f

# %%
Xc = X-X.mean(axis=1)[:, np.newaxis]
U = getEigenvectors(X)
Y = np.matmul(U.T, Xc)

# %%
newMat = (np.matmul(U, Y) + X.mean(axis=1)[:, np.newaxis])

# %%
mse = np.mean((newMat - X) ** 2)
mse

# %% [markdown]
# ### Part g

# %%

# %%
mseList = []
Xc = X-X.mean(axis=1)[:, np.newaxis]
U = getEigenvectors(X)
for i in tqdm(range(1, d+1)):
    Up = U[:, :i]
    Yp = np.matmul(Up.T, Xc)
    newMat = (np.matmul(Up, Yp) + X.mean(axis=1)[:, np.newaxis])
    mseList.append(np.mean((newMat - X) ** 2))

# %%
plt.plot(mseList, '-o', label='MSE')
plt.xlabel('Number of Principle Components')
plt.ylabel('MSE')
plt.title('MSE vs Number of Principle Components')
plt.legend()
plt.show()
