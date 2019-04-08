
# coding: utf-8

# In[1]:


from dolfin import *
from mshr import *
import matplotlib.pyplot as plt


# In[2]:


domain = Circle(Point(0,0),1)
mesh = Mesh("circle3.xml")
plot(mesh)


# In[3]:


V = FunctionSpace(mesh, "CG", 1)


# In[4]:


# Define variational formulation
u = TrialFunction(V)
v = TestFunction(V)


# In[6]:


alpha = 0.01
R = 0.5
r = 0.3
rho = 10
T = 30
dt = mesh.hmin()
theta = 0.5

indata = Expression("pow(R-sqrt(x[0]*x[0]+x[1]*x[1]),2)<=r*r?rho:0",degree=1,R=R,r=r,rho=rho)


# In[7]:


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


# In[8]:


# Define solution
w = Function(V)


# In[9]:


# Solve the problem
t = dt
file = File("visual/2d/2d.pvd")
file << u_old
counter = 1

while t<T:
    
    solve(A, w.vector(), b)
    u_old.assign(w)
    if counter == 1:
        plot(w)
    b = assemble(L)
    bc.apply(b)
    if (counter % 10 == 0):
        file << u_old
    counter +=1
    t += dt

