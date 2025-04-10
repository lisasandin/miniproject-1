###c
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


#nedbrytningshastigheten för repressorproteinet (γR) ska sättas till 0.05 h⁻¹ istället för 0.2 h⁻¹.
#For these parameter values, t < 0, so that the fixed point is stable



# reaction rates
alpha_a = 50
alphaprim_a = 500
alpha_r = 0.01
alphaprim_r = 50

beta_a = 50
beta_r = 5

gamma_a = 1
gamma_r = 1
gamma_c = 2

theta_a = 50
theta_r = 100

sigma_ma = 10
sigma_mr = 0.5
sigma_a = 1
#Ändrad
sigma_r = 0.05

# initial conditions
da = 1
dr = 1
da_prim = 0
dr_prim = 0
ma = 0
mr = 0
A = 0
R = 0
C = 0



def Predator_Prey(t, y):
    da, dr, da_prim, dr_prim, ma, A, mr, R, C = y
    yprime = np.zeros(9)
    yprime[0] = theta_a*da_prim - gamma_a*da*A
    yprime[1] = theta_r*dr_prim - gamma_r*dr*A
    yprime[2] = gamma_a*da*A - theta_a*da_prim
    yprime[3] = gamma_r*dr*A - theta_r*dr_prim
    yprime[4] = alphaprim_a*da_prim + alpha_a*da - sigma_ma*ma
    yprime[5] = beta_a*ma + theta_a*da_prim + theta_r*dr_prim - A*(gamma_a*da + gamma_r*dr + gamma_c*R + sigma_a)
    yprime[6] = alphaprim_r*dr_prim + alpha_r*dr - sigma_mr*mr
    yprime[7] = beta_r*mr - gamma_c*A*R + sigma_a*C - sigma_r*R
    yprime[8] = gamma_c*A*R - sigma_a*C
    return yprime


Initial = [da, dr, da_prim, dr_prim, ma, mr, A, R, C]
FinalTime = 400
teval = np.linspace(0, FinalTime, 1000)      # fine evaluation time samples

def RandExp(lam,N):
    U =  np.random.rand(N)
    X = -1/lam*np.log(1-U)
    return X

def RandDisct(x,p,N):
    cdf = np.cumsum(p)
    U = np.random.rand(N)
    idx = np.searchsorted(cdf, U)
    return x[idx]

def SSA(Initial, StateChangeMat, FinalTime):
   # Inputs:
   #  Initial: initial conditins of size (StateNo x 1)
   #  StateChangeMat: State-change matrix of size (ReactNo, StateNo)
   #  FinalTime: the maximum time we want the process be run

    [m,n] = StateChangeMat.shape
    ReactNum = np.array(range(m))
    AllTimes = {}   # define a dictionary for storing all time levels
    AllStates = {}  # define a dictionary for storing all states at all time levels
    AllStates[0] = Initial
    AllTimes[0] = [0]
    k = 0; t = 0; State = Initial
    while True:
        w = PropensityFunc(State, m)     # propensities
        a = np.sum(w)
        tau = RandExp(a,1)               # WHEN the next reaction happens
        t = t + tau     
        print("eh")                 # update time
        if t > FinalTime:
            break
        which = RandDisct(ReactNum,w/a,1)             # WHICH reaction occurs
        State = State + StateChangeMat[which.item(),] # Uppdate the state
        k += 1
        AllTimes[k] = t
        AllStates[k] = State
    return AllTimes, AllStates


Initial = [da, dr, da_prim, dr_prim, ma, mr, A, R, C]
FinalTime = 400
StateChangeMat = np.array([   #da, dr, da_prim, dr_prim, ma, mr, A, R, C
                          [0, 0, 0, 0, 0, 0, -1, -1, 1], 
                          [0, 0, 0, 0, 0, 0, -1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, -1],
                          [0, 0, 0, 0, 0, 0, 0, -1, 0],
                          [-1, 0, 1, 0, 0, 0, -1, 0, 0],
                          [0, -1, 0, 1, 0, 0, 1, 0, 0],
                          [1, 0, -1, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, -1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 1, 0, -1, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, -1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, 0]
                          ])



def PropensityFunc(State, ReactNo):
    # State: state variable [da, dr, da_prim, dr_prim, ma, mr, A, R, C]
    # ReactNo: number of reactions
    w = np.zeros(ReactNo)
    da, dr, da_prim, dr_prim, ma, mr, A, R, C = State
    w[0] = gamma_c*A*R
    w[1] = sigma_a*A
    w[2] = sigma_a*C
    w[3] = sigma_r*R
    w[4] = gamma_a*da*A
    w[5] = gamma_r*dr*A
    w[6] = theta_a*da_prim
    w[7] = alpha_a*da
    w[8] = alphaprim_a*da_prim
    w[9] = sigma_ma*ma
    w[10] = beta_a*ma
    w[11] = theta_r*dr_prim
    w[12] = alpha_r*dr
    w[13] = alphaprim_r*dr_prim
    w[14] = sigma_mr*mr 
    w[15] = beta_r*mr
    return w



# SSA Simulation of predator-prey model 

Times, States = SSA(Initial, StateChangeMat, FinalTime)

# retrieve state variables A and R from the output list

Times = list(Times.values())
A = [state[6] for state in States.values()]
R = [state[7] for state in States.values()]



def plot_graph():
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    sol = solve_ivp(Predator_Prey, [0, FinalTime], Initial, method = 'BDF', t_eval = teval)
    #sol = solve_ivp(Predator_Prey, [0, FinalTime], Initial, method = 'BDF', t_eval = teval)

    #deterministic model
    axs[0].plot(sol.t, sol.y[7], color='red')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Number of repressor molecules')
    axs[0].set_title('Fig5_a')
    axs[0].legend(loc='upper right')
    axs[0].grid()

    axs[1].plot(Times, R, color='red')
    axs[1].set_title('Fig5_b')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Number of repressor molecules')

    plt.tight_layout()
    plt.show()


plot_graph()
