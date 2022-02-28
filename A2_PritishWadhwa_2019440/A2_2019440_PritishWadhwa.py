# %%
from scipy.stats import multivariate_normal
from numpy.linalg import eig
from tqdm import tqdm
from scipy.stats import bernoulli
import numpy as np
import matplotlib.pyplot as plt
import math
plt.style.use('ggplot')
plt.style.use('dark_background')

# %% [markdown]
# ### Part a

# %%
class1 = bernoulli.rvs(p=[0.5, 0.8], size=(100, 2), random_state=10)
class2 = bernoulli.rvs(p=[0.9, 0.2], size=(100, 2), random_state=10)
np.random.seed(1)
np.random.shuffle(class1)
np.random.shuffle(class2)

# %% [markdown]
# ### Part b

# %%
class1Train = class1[:50]
class1Test = class1[50:]

# %%


def getBernoulliMLE(data):
    """
    Returns the MLE of the Bernoulli distribution for the given data
    """
    sum1 = 0
    sum2 = 0
    for i in data:
        sum1 += i[0]
        sum2 += i[1]
    return [sum1/len(data), sum2/len(data)]


# %%
class1_mu = getBernoulliMLE(class1Train)

# %%
class1_mu

# %%
mu1 = []
mu2 = []
for i in range(1, 51):
    mu = getBernoulliMLE(class1Train[:i])
    mu1.append(mu[0])
    mu2.append(mu[1])
plt.plot(np.arange(1, 51), mu1, 'r', label='mu1')
plt.plot(np.arange(1, 51), mu2, 'b', label='mu2')
plt.xlabel('Number of samples')
plt.ylabel('Mean')
plt.legend()
plt.title('Mean of Bernoulli distribution')
plt.show()

# %% [markdown]
# ### Part c

# %%
class2Train = class2[:50]
class2Test = class2[50:]

# %%
class2_mu = getBernoulliMLE(class2Train)

# %%
class2_mu

# %%
mu1 = []
mu2 = []
for i in range(1, 51):
    mu = getBernoulliMLE(class2Train[:i])
    mu1.append(mu[0])
    mu2.append(mu[1])
plt.plot(np.arange(1, 51), mu1, 'r', label='mu1')
plt.plot(np.arange(1, 51), mu2, 'b', label='mu2')
plt.xlabel('Number of samples')
plt.ylabel('Mean')
plt.legend()
plt.title('Mean of Bernoulli distribution')
plt.show()

# %% [markdown]
# ### Part d

# %%
plt.scatter(class1Train[:, 0], class1Train[:, 1], color='r')
plt.title('Class 1 Training Samples')
plt.show()

# %%
plt.scatter(class2Train[:, 0], class2Train[:, 1], color='b')
plt.title('Class 2 Training Samples')
plt.show()

# %% [markdown]
# ### Part e

# %%


def getDiscriminant(point, mu):
    """
    Returns the discriminant value for a given point and mean
    """
    return (point[0]*math.log(mu[0]) + (1-point[0])*math.log(1-mu[0]) + point[1]*math.log(mu[1]) + (1-point[1])*math.log(1-mu[1]))


# %%
class1Correct = 0
class2Correct = 0
for i in class1Test:
    g1 = getDiscriminant(i, class1_mu)
    g2 = getDiscriminant(i, class2_mu)
    if g1 >= g2:
        class1Correct += 1
for i in class2Test:
    g1 = getDiscriminant(i, class1_mu)
    g2 = getDiscriminant(i, class2_mu)
    if g1 <= g2:
        class2Correct += 1

# %%
class1Correct, class2Correct
# %%
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
