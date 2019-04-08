
# coding: utf-8

# In[1]:


from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from scipy.optimize import minimize


# In[2]:


domain = Sphere(Point(0,0),1)
mesh = Mesh("sphere2.xml")
plot(mesh)


# In[3]:


V = FunctionSpace(mesh, "CG", 1)


# In[4]:


# Define variational formulation
u = TrialFunction(V)
v = TestFunction(V)


# In[5]:


# Solve the problem



def func(data):
    dt = 0.5
    rho, R, r = data
    t = dt
    counter = 1
    alpha = 0.01
    T = 50
    theta = 0.5
    indata = Expression("pow(R-sqrt(x[0]*x[0]+x[1]*x[1]),2)+x[2]*x[2]<=r*r?rho:0",degree=1,R=R,r=r,rho=rho)
    
    def boundary(x, on_boundary):
        return on_boundary

    u_old = Function(V)
    u_old.assign(indata)

    f = 0.0

    # Set up Dirichlet boundary condition,(Strongly)
    gD = Constant(0.0)
    # gD = u_exact
    bc = DirichletBC(V, gD, boundary)

    a = u*v*dx + dt*theta*alpha * inner(grad(u), grad(v))*dx
    L = (u*v*dx - dt*(1-theta)*alpha*inner(grad(u), grad(v))*dx)*u_old

    # A, b = assemble_system(a, L, bc)

    A = assemble(a)
    b = assemble(L)
    bc.apply(A, b)
    
    # Define solution
    w = Function(V)
    #define initial mass
    u_initial = Function(V)
    u_initial = interpolate(indata, V)

    M = (u_initial - u_old) * dx
    mass = assemble(M)
    loss_array = []

    while t<T:

        solve(A, w.vector(), b)
        u_old.assign(w)
        mass = assemble(M)
        b = assemble(L)
        bc.apply(b)
        if (t == 5 or t == 7 or t == 30):
            loss_array.append(mass)
        counter +=1
        t += dt
    
    F = (loss_array[0]-10)**2 + (loss_array[1]-15)**2 + (loss_array[2]-30)**2
    #F = loss_array
    return F


# In[ ]:


#a = func(20,0.5,0.1)
#print(a)


# In[6]:


data = [20.0, 0.5, 0.1]
res = minimize(func, data, method = 'nelder-mead',options = {'xtol':1e-3,'disp':True})
print(res)

