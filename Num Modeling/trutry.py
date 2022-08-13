'''
print(__doc__)

# Author: Nelle Varoquaux <nelle.varoquaux@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_random_state

n = 10
x = np.arange(n)
rs = check_random_state(0)
y = rs.randint(-50, 50, size=(n,)) + 50. * np.log1p(np.arange(n))


# #############################################################################
# Fit IsotonicRegression and LinearRegression models

ir = IsotonicRegression()

y_ = ir.fit_transform(x, y)

lr = LinearRegression()
lr.fit(x[:, np.newaxis], y)  # x needs to be 2d for LinearRegression

# #############################################################################
# Plot result

segments = [[[i, y[i]], [i, y_[i]]] for i in range(n)]
lc = LineCollection(segments, zorder=0)
lc.set_array(np.ones(len(y)))
lc.set_linewidths(np.full(n, 0.5))

fig = plt.figure()
print(x)
print(y)
print(y_)
plt.plot(x, y, 'r.', markersize=12)
plt.plot(x, y_, 'g.-', markersize=12)
plt.plot(x, lr.predict(x[:, np.newaxis]), 'b-')
plt.gca().add_collection(lc)
plt.legend(('Data', 'Isotonic Fit', 'Linear Fit'), loc='lower right')
plt.title('Isotonic regression')
plt.show()
'''
# import packages
import numpy as np
import matplotlib.pyplot as plt
import gurobipy
from pystoned import CNLS
from pystoned.plot import plot2d
from pystoned.constant import CET_ADDI, FUN_PROD, OPT_LOCAL, RTS_VRS

'''
# set seed
np.random.seed(0)

# generate DMUs: DGP
x = np.sort(np.random.uniform(low=1, high=10, size=50))
u = np.abs(np.random.normal(loc=0, scale=0.7, size=50))
y_true = 3 + np.log(x)
y = y_true - u

# define the CNLS model
model = CNLS.CNLS(y, x, z=None, cet = CET_ADDI, fun = FUN_PROD, rts = RTS_VRS)
# solve the model
model.optimize(OPT_LOCAL)

# display the residuals
model.display_residual()

# plot CNLS frontier
plot2d(model, x_select=0, label_name="CNLS", fig_name='CNLS_frontier')
'''
'''
A = np.array([[1, -1, 0], [0, 1, -1]])  #fixed for u=r3
p = 10*np.random.rand(1,3)
b = np.zeros(len(p[0])-1)
alpha = 0.01
u=np.array([0,0,0])

Aisotonic = np.identity(len(p[0]))
for i in range(0, (len(p[0])-1)):
    Aisotonic[i, i+1] = -1
Aisotonic=Aisotonic[0:(len(p[0])-1),0:len(p[0])]
print(Aisotonic)
'''



####Isotonic Working
p = 10 * np.random.rand(1, 7)
b = np.zeros((len(p[0])-1, 1))
alpha = 0.001
xpoints = np.arange(len(p[0]))

Aisotonic = np.identity(len(p[0]))
for i in range(0, (len(p[0]) - 1)):
    Aisotonic[i, i + 1] = -1
Aisotonic = Aisotonic[0:(len(p[0]) - 1), 0:len(p[0])]


def projection(Aisotonic, b, p, alpha):
    beta = len(b)
    jeyp = len(p[0])
    lamda = np.zeros((beta, 1))
    uo = np.zeros((jeyp, 1))
    u = uo + 1
    while np.linalg.norm(u - uo, 2) > 1e-6:
        uo = u
        u = np.transpose(p) - np.dot(np.transpose(Aisotonic), lamda)
        lamda = lamda + alpha * (np.dot(Aisotonic, u) - b)
        # now we want to project the lamba on positive part
        lamda0 = np.append(lamda, 0 * lamda, axis=1)
        lamda = np.amax(lamda0, axis=1).reshape(beta,1)
        #print(lamda)
    return u

xpoints=np.transpose(xpoints)

print(projection(Aisotonic, b, p, alpha))
plt.plot(xpoints,np.transpose(p),'o')
plt.plot(xpoints,projection(Aisotonic, b, p, alpha),'o-')
plt.show()


#### Convex
pconvex = 10 * np.random.rand(1, 7)
b = np.zeros((len(pconvex[0])-2, 1))
alpha = 0.001
xpoints = np.arange(len(pconvex[0]))

Aconvex = np.identity(len(pconvex[0]))
for i in range(0, len(pconvex[0]) - 2):
    Aconvex[i, i] = -0.5
    Aconvex[i, i + 1] = 1
    Aconvex[i, i + 2] = -0.5
Aconvex = Aconvex[0:len(pconvex[0]) - 2, 0:len(pconvex[0])]

'''
def projection(Aconvex, b, pconvex, alpha):
    beta = len(b)
    jeyp = len(pconvex[0])
    lamda = np.zeros((beta, 1))
    uo = np.zeros((jeyp, 1))
    u = uo + 1
    while np.linalg.norm(u - uo, 2) > 1e-06:
        uo = u
        u = np.transpose(pconvex) - np.dot(np.transpose(Aconvex), lamda)
        lamda = lamda + alpha * (np.dot(Aconvex, u) - b)
        # now we want to project the lamba on positive part
        lamda0 = np.append(lamda, 0 * lamda, axis=1)
        lamda = np.amax(lamda0, axis=1).reshape(beta,1)
        #print(lamda)
    return u
'''

# Convex - Application to statistics
xconvex = np.array([50, 30, 25, 22, 27, 30, 26, 32, 28, 26, 21, 16, 8, 4]).reshape((-1, 1))
yconvex = np.array([3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10])
theta = np.zeros(12)
for r in range(1, 13):
    theta[r - 1] = (yconvex[r + 1] - yconvex[r]) / (yconvex[r + 1] - yconvex[r - 1])
    Aconvex = np.identity(14)
    for i in range(0, 12):
        Aconvex[i, i] = -theta[i]
        Aconvex[i, i + 1] = 1
        Aconvex[i, i + 2] = theta[i] - 1
    Aconvex = Aconvex[0:12, 0:14]

Io = np.array([[1, -1, 0, 0], [0, 1, -1, 0], [0, 0, 1, -1]])
be = np.array([0, 0, 0])
pe = np.array([8, 1, 7, 6])

A = np.array([[5, 4], [6, 10]])
b = np.array([3, 2])
p = np.array([0, 1])
alpha = 0.01


#def projection(A, b, p, alpha):
beta = len(b)
jeyp = len(p)
lamda=np.zeros(beta)
uo = np.zeros(jeyp)
u = uo + 1
while np.linalg.norm(u - uo, 2) > 1e-6:
        #print(lamda)
        lamda = lamda + alpha * (lamda*np.amax(u-1))
        lamda0 = np.array([lamda, 0 * lamda])
        lamda = np.amax(lamda0, axis=0)
        uo = u
        u = p - lamda
#    return u


xregression = np.array([1, 2, 3, 4])
print(projection(A, b, p, alpha))
print(projection(Io, be, pe, alpha))
Athree = np.array([[-1 / 2, 1, -1 / 2, 0], [0, -1 / 2, 1, -1 / 2]])
ufour = np.array([0, 0, 0, 0])
pfour = np.array([8, 1, 7, 2])
print(projection(Athree, b, pfour, alpha))
plt.plot(xregression, projection(Athree, b, pfour, alpha))
plt.plot(xregression, projection(Io, be, pe, alpha))
plt.plot(xregression, pfour, 'o')
plt.show()

# linear regression
from sklearn.linear_model import LinearRegression

x = np.array([50, 30, 25, 22, 27, 30, 26, 32, 28, 26, 21, 16, 8, 4]).reshape((-1, 1))
y = np.array([3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10])
model = LinearRegression()
model.fit(x, y)
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)
new_model = LinearRegression().fit(x, y.reshape((-1, 1)))
print('intercept:', new_model.intercept_)
print('slope:', new_model.coef_)
plt.plot(x, new_model.coef_ * x + new_model.intercept_)
plt.plot(x, y, 'o')
plt.show()
# piso = np.array([50,32,30,30,28,27,26,26,25,22,21,16,8,4])
# piso = np.array([10,9.5,9,8.5,8,7.5,7,6.5,6,5.5,5,4.5,4,3.5])

Aiso=np.identity(14)
for i in range (0, 13):
    Aiso[i,i+1]=-1
Aiso=Aiso[0:13,0:14]
biso=np.zeros(13)
print(projection(Aiso, biso , piso , 0.2))

# print(z)
# Isotonic Regression

yisot = -np.array([50, 30, 25, 22, 27, 30, 26, 32, 28, 26, 21, 16, 8, 4])
xisot = np.array([3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]).reshape((-1, 1))

ir = IsotonicRegression()
y_ = -ir.fit_transform(xisot, yisot)
yisot = -yisot
print(y_)
plt.plot(xisot, y_, 'o-')
plt.plot(xisot, yisot, 'o')
plt.show()
'''
# x = np.array([50,25,30,21,16,8,4])
# y = np.array([3.5,4.5,6,8,9,9.5,10])
# rc = conreg(x,y,convex=TRUE)
# lines(rc, col = 2)