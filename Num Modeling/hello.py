import numpy as np
import matplotlib.pyplot as plt
import time

global p
p = 10


# fonction
def f(x, y):
    # return (x-1)**2+p*(y-1)**2
    # return (x-1)**2+p*(x**2-y)**2 # Rosenbrock
    return np.exp(-np.sqrt(x ** 2 + y ** 2)) + np.sqrt(x ** 2 + y ** 2)  # convexe


# gradient
def df(x, y):
    # return np.array([2*(x-1),2*p*(y-1)])
    # return np.array([2*(x-1)+4*p*(x**2-y)*x,-p*2*(x**2-y)])
    return np.array([(x / np.sqrt(x ** 2 + y ** 2)) * (1 - np.exp(-np.sqrt(x ** 2 + y ** 2))),
                     (y / np.sqrt(x ** 2 + y ** 2)) * (1 - np.exp(-np.sqrt(x ** 2 + y ** 2)))])


# Hessian (Newton)
def ddf(x, y):
    # return np.array([[2,0],[0,2*p]])
    # return np.array([[2+4*p*((2*x)*x+(x**2-y)),-p*2*2*x],[-4*p*x,2*p]])
    return np.array([[(y ** 2 / (x ** 2 + y ** 2) ** (3 / 2)) * (1 - np.exp(-(np.sqrt(x ** 2 + y ** 2)))) + (
                x ** 2 / (x ** 2 + y ** 2)) * np.exp(-np.sqrt(x ** 2 + y ** 2)),
                      -(y ** 2 / (x ** 2 + y ** 2) ** (3 / 2)) * (1 - np.exp(-(np.sqrt(x ** 2 + y ** 2)))) + (
                                  y ** 2 / (x ** 2 + y ** 2)) * np.exp(-np.sqrt(x ** 2 + y ** 2))], [
                         -(y ** 2 / (x ** 2 + y ** 2) ** (3 / 2)) * (1 - np.exp(-(np.sqrt(x ** 2 + y ** 2)))) + (
                                     y ** 2 / (x ** 2 + y ** 2)) * np.exp(-np.sqrt(x ** 2 + y ** 2)),
                         (x ** 2 / (x ** 2 + y ** 2) ** (3 / 2)) * (1 - np.exp(-(np.sqrt(x ** 2 + y ** 2)))) + (
                                     y ** 2 / (x ** 2 + y ** 2)) * np.exp(-np.sqrt(x ** 2 + y ** 2))]])


# initial point
xini = -1;
yini = 2

# Plot
delta = 0.025
xx = np.arange(-1, 1.5, delta)
yy = np.arange(-0.5, 2, delta)
X, Y = np.meshgrid(xx, yy)
Z = f(X, Y)

fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z, 20)
ax.clabel(CS, inline=1, fontsize=10)
ax.set_title('Contours')

# Gradient with fixed step

# x=xini; y=yini # initialisation
# maxiter=5000 # sécurité
# g=df(x,y) # gradient
# eps=np.linalg.norm(g,2)*1e-7 # arrêt
# alpha=0.1 # pas
# xh=np.array([x]); yh=np.array([y])
# ti=time.time()
# while np.linalg.norm(g,2)>eps and xh.size<maxiter:
#    u=[x,y]-alpha*g
#    x=u[0]; y=u[1]
#    xh=np.concatenate((xh,np.array([x]))); yh=np.concatenate((yh,np.array([y])))
#    g=df(x,y)
#
# print(xh.size-1)
# print([x,y])
# print(time.time()-ti)
# plt.plot(xh,yh,'r-')

# Gradient with Armijo line search

x = xini;
y = yini  # initialisation
maxiter = 5000  # sécurité
g = df(x, y)  # gradient
eps = np.linalg.norm(g, 2) * 1e-7  # arrêt
alpha0 = 1  # pas initial
epsa = 0.5  # coef Armijo
xh = np.array([x]);
yh = np.array([y])
ti = time.time()
while np.linalg.norm(g, 2) > eps and xh.size < maxiter:
    Jtest = f(x, y) + 1
    alpha = alpha0
    while Jtest > f(x, y) - epsa * alpha * np.linalg.norm(g, 2) ** 2:
        utest = [x, y] - alpha * g
        xtest = utest[0];
        ytest = utest[1]
        Jtest = f(xtest, ytest)
        alpha = alpha / 2
    x = xtest;
    y = ytest
    xh = np.concatenate((xh, np.array([x])));
    yh = np.concatenate((yh, np.array([y])))
    g = df(x, y)

print(xh.size - 1)
print([x, y])
print(time.time() - ti)
plt.plot(xh, yh, 'r--')
plt.show()

# Newton

# x=xini; y=yini # initialisation
# maxiter=5000 # sécurité
# g=df(x,y) # gradient
# eps=np.linalg.norm(g,2)*1e-7 # arrêt
# xh=np.array([x]); yh=np.array([y])
# ti=time.time()
# while np.linalg.norm(g,2)>eps and xh.size<maxiter and x**2+y**2<1e6:
#    u=[x,y]
#    u=u-np.linalg.solve(ddf(x,y),g)
#    x=u[0]; y=u[1]
#    xh=np.concatenate((xh,np.array([x]))); yh=np.concatenate((yh,np.array([y])))
#    g=df(x,y)
#
# print(xh.size-1)
# print([x,y])
# print(time.time()-ti)
# plt.plot(xh,yh,'g-')

# Quasi-Newton

# x=xini; y=yini # initialisation
# maxiter=5000 # sécurité
# g=df(x,y) # gradient
# eps=np.linalg.norm(g,2)*1e-7 # arrêt
# xh=np.array([x]); yh=np.array([y])
# S=np.identity(2)
# ti=time.time()
# while np.linalg.norm(g,2)>eps and xh.size<maxiter and x**2+y**2<1e6:
#    d=-np.matmul(S,g)
#    Jtest=f(x,y)+1
#    alpha=alpha0
#    while Jtest>f(x,y):
#        utest=[x,y]+alpha*d
#        xtest=utest[0]; ytest=utest[1]
#        Jtest=f(xtest,ytest)
#        alpha=alpha/2
#    delta=np.array([xtest-x,ytest-y])
#    gamma=df(xtest,ytest)-df(x,y)
#    # DFP
#    #S=S+np.reshape(np.kron(delta,delta),(2,2))/np.dot(delta,gamma)-np.reshape(np.kron(np.dot(S,gamma),np.dot(S,gamma)),(2,2))/np.dot(gamma,np.dot(S,gamma))
#    # BFGS
#    S=S+(1+np.dot(gamma,np.dot(S,gamma))/np.dot(delta,gamma))*np.reshape(np.kron(delta,delta),(2,2))/np.dot(delta,gamma)-(np.dot(np.reshape(np.kron(delta,gamma),(2,2)),S)+np.dot(S,np.reshape(np.kron(gamma,delta),(2,2))))/np.dot(delta,gamma)
#    x=xtest; y=ytest
#    xh=np.concatenate((xh,np.array([x]))); yh=np.concatenate((yh,np.array([y])))
#    g=df(x,y)
#
# print(xh.size-1)
# print([x,y])
# print(time.time()-ti)
# plt.plot(xh,yh,'g--')

##############################
## Uzawa
# def projection(A,b,p,alpha):
#    m=len(b)
#    n=len(p)
#    lam=np.zeros(m)
#    uo=np.zeros(n)
#    u=uo+1
#    while np.linalg.norm(u-uo,2)>1e-6:
#        print(lam)
#        lam=lam+alpha*(-np.dot(A,np.dot(np.transpose(A),lam))+np.dot(A,p)-b)
#        lam0=np.array([lam,0*lam])
#        lam=np.amax(lam0,axis=0)
#        uo=u
#        u=p-np.dot(np.transpose(A),lam)
#    return u
#