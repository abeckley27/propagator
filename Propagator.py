# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 19:21:21 2018

wavefunction propagator

@author: Aidan Beckley
"""

import math
import time
import numpy as np
import scipy as scp
import scipy.integrate
import functools
from matplotlib import pyplot
from matplotlib import animation

def diff(func,x):
    dx = x[1] - x[0]
    output = [0]

    for i in range(len(x)-1):
        output.append((func[i+1] - func[i]) / dx)

    output[0] = output[1]   # close enough
    return output

def H(func,x,b,t):

    d1 = diff(func,x)
    d2 = diff(d1,x)
    output = np.zeros(len(x))

    for i in range(len(x)):
        output[i] = (-0.5*d2[i] + 0.5*np.power(x[i],2)*func[i] )

    return output

def matrix_element(phi_i,phi_j,x,L,b,t):
    # calculates <phi_i | H | phi_j>
    H_phi_j = H(phi_j,x,b,t)
    f = phi_i * H_phi_j

    return scp.integrate.simps(f,x)

def overlap(phi_i,phi_j,x,L):
    # calculates <phi_i | phi_j>
    f = []
    for k in range(len(x)):
        f.append(phi_i[k] * phi_j[k])

    return scp.integrate.simps(f,x)

def generate_basis(n,L,x):
    return np.sqrt(2.0 / L) * np.sin(n*math.pi*(x+L/2) / L)

def generate_wavefunction(x,a):
    N = np.power((1.0 / math.pi),0.25)
    psi = N * np.exp(-0.5*np.power(x-a,2))
    return psi

t0 = time.time()

#Parameters:
L = 10
a = 1
b = 0.1
n_basis = 20
step = 0.005
dt = 0.01
t_initial = 0
t_final = 20

# execution
space = []
i = -1*L/2
while i <= (L/2)+(2 * step):
    space.append(round(i,4)) # change this if you change step :P
    i += step

x = tuple(space)
# only tuples can be plotted

lst_bases = []
for j in range(n_basis):
    lst_bases.append([])
    for k in x:
        lst_bases[-1].append(generate_basis(j+1,L,k))

basis = []
for j in range(n_basis):
    basis.append(tuple(lst_bases[j]))

print "Constructed basis in %.3f seconds" %(time.time() - t0)

#pyplot.plot(x,basis[0],x,basis[1])

# testing
#y1 = []
#for j in x:
#    y1.append(j**2)
#y = tuple(y1)
#z = []
#for k in x:
#    z.append(math.sin(k))

# Build H_mn
# This is now only an initial hamiltonian

H_mn = np.zeros((n_basis,n_basis))
for m in range(n_basis):
    for n in range(n_basis):
        H_mn[m][n] = matrix_element(basis[m],basis[n],x,L,b,0)

E = np.linalg.eig(H_mn)[0]  # Does this preserve the order ?
#print "Energy eigenvalues: "
#for k in E[:]:
#    print "%.6f" %(k-0.5)

print "Calculated Hamiltonian matrix in %.3f seconds" %(time.time() - t0)

# Generate Psi(x,0)
trial_fn = []
for k in x:
    trial_fn.append(generate_wavefunction(k,a))

# Calculate initial overlap integrals
c_n0 = []
for j in range(n_basis):
    c_n0.append(overlap(basis[j],trial_fn,x,L))

# Express Psi(x,0) as a linear combination of basis functions
# useful as a test of accuracy
approx = []
for j in range(len(x)):
    subt = 0
    for k in range(n_basis):
        subt += c_n0[k] * basis[k][j]
    approx.append(subt)

z2 = tuple(approx)
#pyplot.plot(x,z2)
#pyplot.title("Initial (trial) wavefunction")
avg_x = []
for k in range(len(x)):
    avg_x.append(trial_fn[k] * x[k] * trial_fn[k])

print "Error in initial position:"
print abs(a - sum(avg_x) * step)



print "Calculated initial overlap and wavefunction in %.3f seconds" %(time.time() - t0)

c_n = []
c_n.append(c_n0)
t_lst = [0]
P = []


def func(t,y,n):
# Each value of k corresponds to the differential equation for c_k
# Each value of m corresponds to the mth term of the sum 
#i.e. <phi_k | H | phi_m>    
# The additional argument, n, is the number of basis functions
    output = []
    
    for k in range(n):
        term = 0
        for m in range(n):
            term += y[m] * matrix_element(basis[k], basis[m], x, L, b, t)
            #y[m] is c_m(t). The function is passed a new y at each time step.
        output.append(-1j*term)
    
    return output

def jac(t,y,n):
    J = []
    
    for k in range(n):
        jrow = []
        for m in range(n):
            jrow.append(-1j * matrix_element(basis[k], basis[m], x, L, b, t))
        J.append(jrow)
            
    return J

r = scp.integrate.ode(func, jac).set_integrator('zvode', method='bdf')
r.set_initial_value(c_n0, t_initial)
r.set_f_params(n_basis)
r.set_jac_params(n_basis)

while r.successful() and r.t < t_final:
    t_lst.append(r.t+dt)
    c_n.append(r.integrate(r.t+dt))
    amplitude = np.dot(np.array(c_n0),  np.array(c_n[-1]))
    P.append(np.power(np.absolute(amplitude),2))

print "Propagated system of ODEs in %.3f seconds" %(time.time() - t0)

t_tup = tuple(t_lst)[:-1]
pyplot.plot(t_tup,P)
pyplot.title("Survival Probability")
pyplot.show()
pyplot.cla()

def show_evolution(ts):
    Psi_t = []
    for j in range(len(x)):
        subt = 0
        for k in range(n_basis):
            subt += c_n[ts][k] * basis[k][j]
        Psi_t.append( np.power( np.absolute(subt),2) )

    z2 = tuple(Psi_t)
    pyplot.plot(x,z2)
    pyplot.title("Wavefunction at t = %.3f" %(ts*dt))
    pyplot.ylim( (-0.2, 1.2) )
    pyplot.savefig("%d.png" %ts)
    pyplot.cla()

# Find position expectation value over time
# You can speed this up by converting it into a matrix multiplication problem

Psi_t = (np.transpose(np.matrix(basis)) * np.transpose(np.matrix(c_n)))
Pr_density = np.multiply(Psi_t.conjugate(), Psi_t)

xt = []
for t in range(len(t_tup)):
    x1 = []
    for k in range(len(x)):
        x1.append(np.conj(Pr_density[(k,t)] * x[k]))
    xt.append(sum(x1) * step)


pyplot.plot(t_tup,tuple(xt))
pyplot.title("<x> (t) ")
pyplot.show()
print "Calculated average position in %.3f seconds." %(time.time() - t0)

fig = pyplot.figure()
ax = pyplot.axes(xlim=(-5, 5), ylim=(-0.2, 1.2))
line, = ax.plot([], [], lw=2)

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,

# animation function.  This is called sequentially
def animate(ts):
    y = []
    for k in range(len(x)):
        y.append(Pr_density[k,ts])
    line.set_data(x, y)
    return line,

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=1800, interval=10, blit=True)

anim.save('animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

print "Ran in %.2f seconds." %(time.time() - t0)

















