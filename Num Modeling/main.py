import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import time
import cobs
from matplotlib.collections import LineCollection
from sklearn.isotonic import IsotonicRegression
from timeit import default_timer as timer

# import time

'''
global p
p=10

def f(x, y):
   # return (x-1)**2+p*(y-1)**2 # Quadratic
    return (x-1)**2+p*(x**2-y)**2 # Rosenbrock
    # return np.exp(-np.sqrt(x ** 2 + y ** 2)) + np.sqrt(x ** 2 + y ** 2)  # Radial

# gradient
def df(x, y):
   #return np.array([2*(x-1),2*p*(y-1)])
    return np.array([2*(x-1)+4*p*(x**2-y)*x,-p*2*(x**2-y)])
   # return np.array([(x / np.sqrt(x ** 2 + y ** 2)) * (1 - np.exp(-np.sqrt(x ** 2 + y ** 2))),
    #               (y / np.sqrt(x ** 2 + y ** 2)) * (1 - np.exp(-np.sqrt(x ** 2 + y ** 2)))])


# Hessian (Newton)
def ddf(x, y):
   #return np.array([[2,0],[0,2*p]])
   return np.array([[2+4*p*((2*x)*x+(x**2-y)),-p*2*2*x],[-4*p*x,2*p]])
    # return np.array([[(y ** 2 / (x ** 2 + y ** 2) ** (3 / 2)) * (1 - np.exp(-(np.sqrt(x ** 2 + y ** 2)))) + (
     #          x ** 2 / (x ** 2 + y ** 2)) * np.exp(-np.sqrt(x ** 2 + y ** 2)),
      #                -(y ** 2 / (x ** 2 + y ** 2) ** (3 / 2)) * (1 - np.exp(-(np.sqrt(x ** 2 + y ** 2)))) + (
       #                           y ** 2 / (x ** 2 + y ** 2)) * np.exp(-np.sqrt(x ** 2 + y ** 2))], [
        #                 -(y ** 2 / (x ** 2 + y ** 2) ** (3 / 2)) * (1 - np.exp(-(np.sqrt(x ** 2 + y ** 2)))) + (
         #                            y ** 2 / (x ** 2 + y ** 2)) * np.exp(-np.sqrt(x ** 2 + y ** 2)),
          #               (x ** 2 / (x ** 2 + y ** 2) ** (3 / 2)) * (1 - np.exp(-(np.sqrt(x ** 2 + y ** 2)))) + (
           #                          y ** 2 / (x ** 2 + y ** 2)) * np.exp(-np.sqrt(x ** 2 + y ** 2))]])



# initial point
xi = 1.1;
yi = 1.2;

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

x=xi; y=yi # initialisation
maxiter=5000 # sécurité
#eps=np.linalg.norm(g,2)*1e-7 # arrêt
xh=np.array([x]); yh=np.array([y])
#ti=time.time()
c=0;
deltax=np.linalg.solve(ddf(x,y),df(x,y))
u=np.array([x,y])
'''

'''
while (deltax.any()>(1e-7) and c<=1000):
    c=c+1
    u=[x,y]
    deltax=np.linalg.solve(ddf(x,y),df(x,y))
    u = u - deltax
    x = u[0]
    y = u[1]
    xh = np.concatenate((xh, np.array([x]))); yh = np.concatenate((yh, np.array([y])))
    g=df(x,y)
print(u)
print(deltax)
print(c)
print(xh)
print(yh)
plt.plot(xh,yh,'g-')
plt.show()
print(df)
print(ddf)
a=0;
l=0;
delta=np.array([1,2])
H=np.identity(2)
'''

'''
while (np.abs(delta[0])>1e-9 and np.abs(delta[1])>1e-9 and a<=2000):
    a=a+1
    jac=df(x,y)
    d = -np.dot(H, jac)
    stepq=2 #random value to initialize
    floop=f(x,y)+.1 #to be able to enter the loop
    while floop>f(x,y):
        uloop=[x,y]+stepq*d
        xloop=uloop[0]
        yloop=uloop[1]
        floop=f(xloop,yloop)
        stepq=stepq/2
    gamma=df(xloop,yloop)-df(x,y)
    delta=np.array([xloop-x,yloop-y])
    dt=np.transpose(delta)
    gt=np.transpose(gamma)
   # H=H+np.dot(delta,dt)/np.dot(dt,gamma)-np.dot(np.dot(H,gamma,gt),H)/np.dot(gt,H,gamma)
   # H=H-(np.dot(delta,gt,H)+np.dot(H,gamma,dt))/(np.dot(dt,gamma))
    H=H+(1+np.dot(gt,H,gamma)/np.dot(dt,gamma))*(np.dot(delta,dt)/np.dot(dt,gamma)) -(np.dot(delta,np.dot(gt,H))) +np.dot(H,np.dot(gamma,dt))/(np.dot(dt,gamma))
    x=xloop
    y=yloop
    xh= np.concatenate((xh, np.array([x])))
    yh = np.concatenate((yh, np.array([y])))
    jac=df(x,y)
print(H)
plt.plot(xh,yh,'g-')
plt.show()
print(xloop,yloop)
'''
'''
# Uzawa
#start = timer()
#ti = time.time()
def projection(A, b, p, alpha):
    beta = len(b)
    jeyp = len(p[0])
    lamda = np.zeros((beta, 1))
    uo = np.zeros((jeyp, 1))
    u = uo + 1
    iter=0
    while np.linalg.norm(u - uo, 2) > 1e-6:
        uo = u
        u = np.transpose(p) - np.dot(np.transpose(A), lamda)
        lamda = lamda + alpha * (np.dot(A, u) - b)
        # now we want to project the lamba on positive part
        lamda0 = np.append(lamda, 0 * lamda, axis=1)
        lamda = np.amax(lamda0, axis=1).reshape(beta,1)
        iter=iter+1
        #print(iter)
    return u
'''
'''
#end = timer()
#infinity input
#p_inf = np.array([[1.88],[8.85],[1.39],[4.22],[9.83]])
#p_inf = np.transpose(p_inf)
p_inf = np.random.rand(1, 5)
b_inf = np.ones((len(p_inf[0]), 1))
A_inf=np.identity(len(p_inf[0]))
#print('p_inf', p_inf)

# alpha upper limit
Bi=np.dot(A_inf, np.transpose(A_inf))
eigeninf=LA.eigvals(Bi).max()
limit_inf=2/(abs(eigeninf))
alpha = 1.4
xpoints = np.arange(len(p_inf[0]))

u1=projection(A_inf,b_inf,p_inf,alpha)
u2=projection(-A_inf,b_inf,p_inf,alpha)
u_int=np.transpose(np.concatenate((u1, u2), axis=1))
u_int=np.absolute(u_int)
#print('u_int first row', u_int[0,:])
#print('u_int second row', u_int[1,:])

for i in range (0,(len(p_inf[0]))) :
    if u_int[0,i]>1.0001:
        u_int[0,i]=0
for i in range (0,(len(p_inf[0]))) :
    if u_int[1,i]>1.0001:
       u_int[1,i]=0
#print('unique solution',u_int)
#print(time.time() - ti)
#print('final unique solution',u_int.max(axis=0))
#print(end - start) # Time in seconds
plt.plot(xpoints,np.transpose(p_inf),'o')
plt.plot(xpoints,u_int.max(axis=0),'o-')
plt.xlabel("element i")
plt.ylabel("u-p")
plt.show()
'''
'''
# Start of Application:Regression
# Isotonic
p = 10 * np.random.rand(1, 5)
#print('p',p)
#p = np.array([[8.63600419],[9.18778716],[3.18309338],[8.02354045],[5.1053322]])
#p = np.array([[0.54863857], [5.52507966], [9.7165155], [8.78835412], [2.66371649]])
p=np.array([[3.66166666], [4.67405793], [4.36214204], [4.73485713], [5.87080719], [2.07742866]])
p = np.transpose(p)
b = np.zeros((len(p[0])-1, 1))
alpha = 0.5
xpoints = np.arange(len(p[0]))

Aisotonic = np.identity(len(p[0]))
for i in range(0, (len(p[0]) - 1)):
    Aisotonic[i, i + 1] = -1
Aisotonic = Aisotonic[0:(len(p[0]) - 1), 0:len(p[0])]

# alpha upper limit
Bi=np.dot(Aisotonic, np.transpose(Aisotonic))
eigeniso=LA.eigvals(Bi).max()
#print('eigeniso',eigeniso)

#print('limit isotonic ')
limit_iso=2/(abs(eigeniso))
#print('limit_iso',limit_iso)

xpoints=np.transpose(xpoints)

#print(p)
print(projection(Aisotonic, b, p, alpha))
plt.plot(xpoints,np.transpose(p),'o')
plt.plot(xpoints,projection(Aisotonic, b, p, alpha),'o-')
plt.xlabel("element i")
plt.ylabel("u-p")
plt.show()
'''
'''
# Convex
#pconvex = 10 * np.random.rand(1, 6)
pconvex=np.array([[3.66166666], [4.67405793], [4.36214204], [4.73485713], [5.87080719], [2.07742866]])
pconvex = np.transpose(pconvex)
#print(pconvex)
b = np.zeros((len(pconvex[0])-2, 1))
alpha_c = 0.3
xpoints = np.arange(len(pconvex[0]))

Aconvex = np.identity(len(pconvex[0]))
for i in range(0, len(pconvex[0]) - 2):
    Aconvex[i, i] = -0.5
    Aconvex[i, i + 1] = 1
    Aconvex[i, i + 2] = -0.5
Aconvex = Aconvex[0:len(pconvex[0]) - 2, 0:len(pconvex[0])]

xpoints=np.transpose(xpoints)

# alpha upper limit
Bc=np.dot(Aconvex, np.transpose(Aconvex))
eigenc=LA.eigvals(Bc).max()

#print('limit convex')
limit_convex=2/(abs(eigenc))
#print(limit_convex)

print(projection(Aconvex, b, pconvex, alpha_c))
plt.plot(xpoints,np.transpose(pconvex),'o')
plt.plot(xpoints,projection(Aconvex, b, pconvex, alpha_c),'o-')
plt.xlabel("element i")
plt.ylabel("u-p")
plt.show()
'''
'''
#application to statistics.
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
#plt.plot(x, new_model.coef_ * x + new_model.intercept_)
#plt.plot(x, y, 'o')
#plt.xlabel("No x of sold honey glasses")
#plt.ylabel("Price p(x) per glass")
#plt.show()
'''
'''
# Isotonic Regression

yisot = -np.array([3.5, 7.0, 4.0, 6.0, 7.5, 5.5, 6.5, 8.0, 4.5, 5.0, 8.5, 9, 9.5, 10])
xisot = np.array([50, 32, 30, 30, 28, 27, 26, 26, 25, 22, 21, 16, 8, 4]).reshape((-1, 1))

ir = IsotonicRegression()
y_ = -ir.fit_transform(xisot, yisot)
yisot = -yisot
print(y_)
#plt.plot(xisot, y_, '-')
#plt.plot(xisot, yisot, 'o')
#plt.xlabel("No x of sold honey glasses")
#plt.ylabel("Price p(x) per glass")
#plt.show()


#non-uniform x data (not concave, check last zoom recorded)

xconvex = np.array([50, 32, 30, 30, 28, 27, 26, 26, 25, 22, 21, 16, 8, 4]).reshape((-1, 1))
yconvex = np.array([3.5, 7.0, 4.0, 6.0, 7.5, 5.5, 6.5, 8.0, 4.5, 5.0, 8.5, 9, 9.5, 10])
p_stat_convex = np.transpose(yconvex)
#print('statcon', len(yconvex[0]))
b_stat = np.zeros((len(p_stat_convex)-1, 1))
theta = np.zeros((len(p_stat_convex)-2,1))
alpha_c = 0.4

for r in range(1, len(yconvex)-1):
    theta[r - 1] = (xconvex[r + 1] - xconvex[r]) / (xconvex[r + 1] - xconvex[r - 1])
    Aconvex_stat = np.identity(len(yconvex))
    for i in range(0, (len(yconvex)-2)):
        Aconvex_stat[i, i] = -theta[i]
        Aconvex_stat[i, i + 1] = 1
        Aconvex_stat[i, i + 2] = theta[i] - 1
    Aconvex_stat = Aconvex_stat[0:(len(yconvex)-2), 0:len(yconvex)]
#mention that we tried this with uniform data and got all thetas=0.5
#print(theta)
yconvex = np.array([3.5, 7.0, 4.0, 6.0, 7.5, 5.5, 6.5, 8.0, 4.5, 5.0, 8.5, 9, 9.5, 10]).reshape(1,-1)
bul = np.zeros((len(yconvex[0])-2, 1))
print(projection(Aconvex_stat, bul, yconvex, alpha_c))
#x_stat=np.transpose(np.arange(len(yconvex[0])))
plt.plot(xisot, yisot, 'o', label = 'Observations')
plt.plot(xisot, y_, '-', label = 'Isotonic Model')
plt.plot(x, new_model.coef_ * x + new_model.intercept_, label = 'Linear Model')
#plt.plot(x, y, 'o')
plt.plot(xconvex, projection(Aconvex_stat, bul, yconvex, alpha_c), '-', label = 'Convex Model')
#plt.plot(xconvex, np.transpose(yconvex), 'o')
plt.xlabel("No x of sold honey glasses")
plt.ylabel("Price p(x) per glass")
plt.legend()
plt.show()
print('convex', projection(Aconvex_stat, bul, yconvex, alpha_c))
print('iso', y_)
print('linear', new_model.coef_ * x + new_model.intercept_)'''
