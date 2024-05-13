import matplotlib.pyplot as plt
import math
import numpy as np
import scipy.integrate as integrate
from scipy.integrate import quad
from scipy import special
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from sympy import *
from scipy.linalg import lu, inv
from scipy.special import binom
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import AutoMinorLocator

# Time interval for integration 
t_min = 1e-2
t_max = 0.2
t_span = (t_min, t_max)
timespan = np.linspace(t_min, t_max, 1000000)

# Physical constants
k = 200
Ωdm0 = 0.315
Ωγ0 = 9.28656e-5
ΩΛ0 = 1 - Ωdm0 - Ωγ0
H0 = 1/np.sqrt(3*ΩΛ0)
ainit = [H0 * np.sqrt(Ωγ0)*t_min]

xinit = np.array([1., 2., 1., 2., 1.])
yinit = [H0 * np.sqrt(Ωγ0)*t_min]
x5 = np.array([yinit[0],1, 2, 1, 2, 1])

# Define all functions and dummy variables
t = symbols("t")
s = Function("s")(t)
H = Function("H")(t)
Ωγ = Function("Ωγ")(t)
Ωdm = Function("Ωdm")(t)
ΩΛ = Function("ΩΛ")(t)

# Density function definitions
ΩΛ = ΩΛ0 * (s*H0/H)**2
Ωdm = Ωdm0 * s**(-3) * (s*H0/H)**2
Ωγ = Ωγ0 * s**(-4) * (s*H0/H)**2

######################################################
# DIFFERENTIAL EQUATION MATRIX 
# (Make sure to update equations for background if you change it!)

A = Matrix([[-2 *H* Ωγ, -k, -0.5*H*Ωdm, 0, -k**2/(3*H) - H],
                    [k/3, 0, 0, 0, k/3],
                    [-6 *H* Ωγ, 0, -1.5*H*Ωdm, -1j*k, -k**2/(H) - 3*H],
                    [0, 0, 0, -H/s, -1j*k],
                    [-2 *H* Ωγ, 0, -0.5*H*Ωdm, 0, -k**2/(3*H) - H]])

N = A.shape[0]

simplify_equation = {s.diff(t,1): s*H,
                     H.diff(t,1): (H**2)*(ΩΛ0 - Ωγ0*(s**-4) - 0.5*Ωdm0*(s**-3))*(s*H0/H)**2,
                     s.diff(t,2): s*(H**2)*(1 + (ΩΛ0 - Ωγ0*(s**-4) - 0.5*Ωdm0*(s**-3))*(s*H0/H)**2)}

B = [A,A.diff(t).xreplace(simplify_equation),
     A.diff(t).xreplace(simplify_equation).diff(t).xreplace(simplify_equation),
      A.diff(t).xreplace(simplify_equation).diff(t).xreplace(simplify_equation).diff(t).xreplace(simplify_equation),
      A.diff(t).xreplace(simplify_equation).diff(t).xreplace(simplify_equation).diff(t).xreplace(simplify_equation).diff(t).xreplace(simplify_equation)]

######################################################
# SCALE FACTOR BACKGROUND SOLVER 

# Friedmann equation ODE for scale factor
def scaleODE(t,x):
    dadt = H0*x**2*np.sqrt(Ωγ0*np.abs(x**-4) + Ωdm0*np.abs(x**-3) + ΩΛ0)
    return dadt

# Solve the system of ODEs using RKF45 method with solve_ivp
scalesol = solve_ivp(scaleODE, t_span, ainit, method='RK45', dense_output=True)
solution_at_t_eval = scalesol.sol(timespan)
solution_at_dt_eval = scaleODE(timespan,scalesol.sol(timespan))

print("Now solving the system for general time...")
######################################################

def findODEcoefficients(τ,initialflag):
    print('Time is ' + str(τ))
    # Define a substitution rule to swap the derivatives with dummy variables
    substitution_rule = {H: (scaleODE(τ,scalesol.sol(τ))[0]/scalesol.sol(τ)[0]),
                         s.diff(t,1): scaleODE(τ,scalesol.sol(τ))[0], s: scalesol.sol(τ)[0],
                         t: τ}
    if initialflag:
        x0 = [np.zeros(N, dtype=complex) for _ in range(N)]
        x0[0] = xinit
        
        for i in range(1,N):
            for j in range(i+1):
                x0[i] = x0[i] + np.transpose(np.array(binom(i-1,j) * B[j].xreplace(substitution_rule)) @ np.transpose(x0[i-j-1]))    
        return(np.array([j[0] for j in x0]).astype(complex))
    else:
        C = [np.array(B[i].xreplace(substitution_rule)).astype(complex) for i in range(N)]
        M = np.block([np.zeros((N*N,N), dtype=complex), -np.eye(N*N, dtype=complex)])
        for i in range(N):
            for j in range(i+1):
                M[i*N:(i+1)*N, j*N:(j+1)*N] = binom(i,j) * C[i-j] 
        i =  [j for j in range(N*(N+1)) if j%N != 0] + [j for j in range(N*(N+1)) if j%N == 0]
        M = M[:,i]
        _, _, u = lu(M)
        ans = u[-1, -N-1:]
        ans /= -ans[N]
        return(np.array([(ans[j]) for j in range(N)]))
        
    
######################################################
# METHOD A: Numerically decoupling the differential equations

print("Starting the solver...")

# The differential equation system as a function of time, needs to re-evaluate above at each time to get the new coefficients
def model(τ, y):
    dydt = np.zeros(N, dtype=complex)
    coeff = findODEcoefficients(τ,False)
    #print(coeff)
    dydt[0] = y[1]
    dydt[1] = y[2]
    dydt[2] = y[3]
    dydt[3] = y[4]
    dydt[4] = coeff[4]*y[4] + coeff[3]*y[3] + coeff[2]*y[2] + coeff[1]*y[1] + coeff[0]*y[0]
    return dydt

# Obtain the initial conditions, store in y0, and then perform odeint integration on decoupled equation
y0 = findODEcoefficients(t_min,True)

y = solve_ivp(model, t_span, y0, method='RK45', dense_output=True,rtol=1e-2,atol=1e-2)
nsolution_at_t_eval = y.sol(timespan)

#########################################################
# METHOD B: Verifying by solving the background

def system_of_ODEs(t, X):
    
    z,x = np.split(X,[1])
    a = z[0]
    dzdt = scaleODE(t,z)
    dadt = dzdt[0]
    H = dadt/a
    
    # Get the matrix A at the current time t
    ΩΛ = ΩΛ0 * (a*H0/H)**2
    Ωdm = Ωdm0 * ((a)**-3) * (a*H0/H)**2
    Ωγ =  Ωγ0 * ((a)**-4) * (a*H0/H)**2
    
    A = np.array([[-2 *H* Ωγ, -k, -0.5*H*Ωdm, 0, -k**2/(3*H) - H],
                    [k/3, 0, 0, 0, k/3],
                    [-6 *H* Ωγ, 0, -1.5*H*Ωdm, -1j*k, -k**2/(H) - 3*H],
                    [0, 0, 0, -H/a, -1j*k],
                    [-2 *H* Ωγ, 0, -0.5*H*Ωdm, 0, -k**2/(3*H) - H]])
    
    # Compute the derivative of the state vector
    dxdt = np.dot(A, x)
    
    return np.concatenate([dzdt,dxdt])

solution = solve_ivp(system_of_ODEs, t_span, x5, method='RK45', dense_output=True,rtol=1e-2,atol=1e-2)
solution_at_t_eval = solution.sol(timespan)

#########################################################
# Plot the final result
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 3))

fig.suptitle('Radiation dominated photon and CDM system: $\Theta_0$, $k$ = 200, $\eta_0 = 10^{-2}$')

ax1.plot(solution.t, solution.y[1], linestyle='None', linewidth=1, marker='.',color='turquoise', label="No decoupling", markersize=2)
ax1.plot(y.t, y.y[0], linestyle='None', linewidth=1, marker='x',label="LU decoupled", markersize=2)
ax1.legend(loc = "upper left") 
ax1.set_ylabel('$\Theta_0$')
plt.xlabel('$\eta$')

ax2.plot(timespan, np.abs(nsolution_at_t_eval[0] - solution_at_t_eval[1]),linestyle='--',linewidth=0.5,color='orange',label="Absolute error")
ax2.legend(loc = "upper left") 
ax2.set_ylabel('Error')

plt.tight_layout()
ax1.yaxis.set_major_locator(MaxNLocator(9))
plt.subplots_adjust(hspace=0)
plt.gca().xaxis.set_minor_locator(AutoMinorLocator(n=10)) 
ax2.yaxis.set_minor_locator(AutoMinorLocator(n=5))
plt.savefig("LUexample.pdf", bbox_inches='tight', pad_inches=0)
plt.show()   
plt.show()    