#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, transpile, schedule, transpiler
from qiskit import IBMQ
from qiskit.tools.jupyter import *
from qiskit.tools import job_monitor
from qiskit.providers.ibmq.managed import IBMQJobManager

import numpy as np
import matplotlib.pyplot as plt


# In[2]:


provider = IBMQ.enable_account('account-id-here')
#provider = IBMQ.load_account()


# In[3]:


backend = provider.get_backend('ibmq_lima')
backend


# ## Get gates duration
# https://qiskit.org/documentation/stubs/qiskit.transpiler.InstructionDurations.get.html
# https://qiskit.org/documentation/tutorials/circuits_advanced/08_gathering_system_information.

# In[4]:


# Get duration of instructions

dt_in_s = backend.configuration().dt
Reset_duration = transpiler.InstructionDurations.from_backend(backend).get("reset",0)
I_duration     = transpiler.InstructionDurations.from_backend(backend).get("id",3)
Z_duration     = transpiler.InstructionDurations.from_backend(backend).get("rz",0)
SX_duration    = transpiler.InstructionDurations.from_backend(backend).get("sx",1)
X_duration     = transpiler.InstructionDurations.from_backend(backend).get("x",1)
Y_duration     = 3*Z_duration + 2*SX_duration
H_duration     = 2*Z_duration + SX_duration
Measurement_duration = transpiler.InstructionDurations.from_backend(backend).get("measure",1)
Measurement_duration3 = transpiler.InstructionDurations.from_backend(backend).get("measure",3)

CNOT_durations = [] # Will be in dt units
for pair in backend.configuration().coupling_map:
    CNOT_pair_duration = transpiler.InstructionDurations.from_backend(backend).get("cx",pair)
    CNOT_durations.append([str(pair),CNOT_pair_duration])
CNOT_durations = dict(CNOT_durations)

tau_cnot01 = CNOT_durations["[0, 1]"]
tau_cnot10 = CNOT_durations["[1, 0]"]
tau_cnot34 = CNOT_durations["[3, 4]"]
tau_cnot43 = CNOT_durations["[4, 3]"]
tau_cnot13 = CNOT_durations["[1, 3]"]


# ## Define the circuit creation functions

# In[5]:


def make_swap_transpiled_circuit(bell_state=[0,0]):
    
    q  = QuantumRegister(5, 'q')
    c  = ClassicalRegister(2, 'c')
    circuit = QuantumCircuit(q, c)

    #circuit.reset([0,1,3,4])
    
    # Data gate (Bell pair creation gate).
    if bell_state[0] == 1:
        circuit.x(1)
    if bell_state[1] == 1:
        circuit.x(3)
    circuit.h(1)
    circuit.cnot(1,3)
    
    circuit.barrier(q)
    
    # First SWAP gates.
    circuit.cnot(0,1)
    circuit.cnot(1,0)
    circuit.cnot(0,1)
    circuit.cnot(3,4)
    circuit.cnot(4,3)
    circuit.cnot(3,4)
    
    circuit.measure(0,0)
    circuit.measure(4,1)

    tcircuit = transpile(circuit, backend, scheduling_method="asap", optimization_level=0)
    
    return tcircuit


def make_swapWithZs_transpiled_circuit(bell_state=[0,0], wait=False):
    
    q  = QuantumRegister(5, 'q')
    c  = ClassicalRegister(2, 'c')
    circuit = QuantumCircuit(q, c)

    #circuit.reset([0,1,3,4])

    # Data gate (Bell pair creation gate).
    if bell_state[0] == 1:
        circuit.x(1)
    if bell_state[1] == 1:
        circuit.x(3)
    circuit.h(1)
    circuit.cnot(1,3)

    circuit.barrier(q)
    
    # First SWAP gates.
    circuit.z([0,1])
    circuit.cnot(0,1)
    circuit.z([0,1])
    circuit.cnot(1,0)
    circuit.z([0,1])
    circuit.cnot(0,1)
    
    circuit.z([3,4])
    circuit.cnot(3,4)
    circuit.z([3,4])
    circuit.cnot(4,3)
    circuit.z([3,4])
    circuit.cnot(3,4)
    
    if wait==True:
        circuit.z([0,1])
        circuit.delay(CNOT_durations["[1, 0]"], [0, 1], "dt")
        circuit.z([0,1])
        circuit.z([3,4])
        circuit.delay(CNOT_durations["[4, 3]"], [3, 4], "dt")
        circuit.z([3,4])
        
    circuit.measure(0,0)
    circuit.measure(4,1)

    tcircuit = transpile(circuit, backend, scheduling_method="asap", optimization_level=0)
    
    return tcircuit


def make_swapWithXs_transpiled_circuit(bell_state=[0,0], wait=False):
    
    q  = QuantumRegister(5, 'q')
    c  = ClassicalRegister(2, 'c')
    circuit = QuantumCircuit(q, c)

    #circuit.reset([0,1,3,4])

    # Data gate (Bell pair creation gate).
    if bell_state[0] == 1:
        circuit.x(1)
    if bell_state[1] == 1:
        circuit.x(3)
    circuit.h(1)
    circuit.cnot(1,3)

    circuit.barrier(q)
    
    # First SWAP gates.
    circuit.x([0,1])
    circuit.cnot(0,1)
    circuit.x([0,1])
    circuit.cnot(1,0)
    circuit.x([0,1])
    circuit.cnot(0,1)
    
    circuit.x([3,4])
    circuit.cnot(3,4)
    circuit.x([3,4])
    circuit.cnot(4,3)
    circuit.x([3,4])
    circuit.cnot(3,4)
    
    if wait==True:
        circuit.x([0,1])
        circuit.delay(CNOT_durations["[1, 0]"], [0, 1], "dt")
        circuit.x([0,1])
        circuit.x([3,4])
        circuit.delay(CNOT_durations["[4, 3]"], [3, 4], "dt")
        circuit.x([0,1])

    circuit.measure(0,0)
    circuit.measure(4,1)

    tcircuit = transpile(circuit, backend, scheduling_method="asap", optimization_level=0)
    
    return tcircuit


def make_2swap_transpiled_circuit(bell_state=[0,0]):
    
    q  = QuantumRegister(5, 'q')
    c  = ClassicalRegister(2, 'c')
    circuit = QuantumCircuit(q, c)

    #circuit.reset([0,1,3,4])

    # Data gate (Bell pair creation gate).
    if bell_state[0] == 1:
        circuit.x(1)
    if bell_state[1] == 1:
        circuit.x(3)
    circuit.h(1)
    circuit.cnot(1,3)

    circuit.barrier(q)
    
    # First SWAP gates.
    circuit.cnot(0,1)
    circuit.cnot(1,0)
    circuit.cnot(0,1)
    circuit.cnot(3,4)
    circuit.cnot(4,3)
    circuit.cnot(3,4)
    
    # Second SWAP gates
    circuit.cnot(0,1)
    circuit.cnot(1,0)
    circuit.cnot(0,1)
    circuit.cnot(3,4)
    circuit.cnot(4,3)
    circuit.cnot(3,4)

    circuit.measure(1,0)
    circuit.measure(3,1)

    tcircuit = transpile(circuit, backend, scheduling_method="asap", optimization_level=0)
    
    return tcircuit


def make_2swapWithZs_transpiled_circuit(bell_state=[0,0], wait1=False, wait2=False):
    
    q  = QuantumRegister(5, 'q')
    c  = ClassicalRegister(2, 'c')
    circuit = QuantumCircuit(q, c)

    #circuit.reset([0,1,3,4])

    # Data gate (Bell pair creation gate).
    if bell_state[0] == 1:
        circuit.x(1)
    if bell_state[1] == 1:
        circuit.x(3)
    circuit.h(1)
    circuit.cnot(1,3)

    circuit.barrier(q)
    
    # First SWAP gates.
    circuit.z([0,1])
    circuit.cnot(0,1)
    circuit.z([0,1])
    circuit.cnot(1,0)
    circuit.z([0,1])
    circuit.cnot(0,1)
    
    circuit.z([3,4])
    circuit.cnot(3,4)
    circuit.z([3,4])
    circuit.cnot(4,3)
    circuit.z([3,4])
    circuit.cnot(3,4)
    
    if wait1==True:
        circuit.z([0,1])
        circuit.delay(CNOT_durations["[1, 0]"], [0, 1], "dt")
        circuit.z([3,4])
        circuit.delay(CNOT_durations["[4, 3]"], [3, 4], "dt")
    
    # Second SWAP gates.
    circuit.z([0,1])
    circuit.cnot(0,1)
    circuit.z([0,1])
    circuit.cnot(1,0)
    circuit.z([0,1])
    circuit.cnot(0,1)
    
    circuit.z([3,4])
    circuit.cnot(3,4)
    circuit.z([3,4])
    circuit.cnot(4,3)
    circuit.z([3,4])
    circuit.cnot(3,4)

    if wait2==True:
        circuit.z([0,1])
        circuit.delay(CNOT_durations["[1, 0]"], [0, 1], "dt")
        circuit.z([3,4])
        circuit.delay(CNOT_durations["[4, 3]"], [3, 4], "dt")
    
    circuit.measure(1,0)
    circuit.measure(3,1)

    tcircuit = transpile(circuit, backend, scheduling_method="asap", optimization_level=0)
    
    return tcircuit


def make_2swapWithXs_transpiled_circuit(bell_state=[0,0], wait1=False, wait2=False):
    
    q  = QuantumRegister(5, 'q')
    c  = ClassicalRegister(2, 'c')
    circuit = QuantumCircuit(q, c)

    #circuit.reset([0,1,3,4])

    # Data gate (Bell pair creation gate).
    if bell_state[0] == 1:
        circuit.x(1)
    if bell_state[1] == 1:
        circuit.x(3)
    circuit.h(1)
    circuit.cnot(1,3)

    circuit.barrier(q)
    
    # First SWAP gates.
    circuit.x([0,1])
    circuit.cnot(0,1)
    circuit.x([0,1])
    circuit.cnot(1,0)
    circuit.x([0,1])
    circuit.cnot(0,1)
    
    circuit.x([3,4])
    circuit.cnot(3,4)
    circuit.x([3,4])
    circuit.cnot(4,3)
    circuit.x([3,4])
    circuit.cnot(3,4)
    
    if wait1==True:
        circuit.x([0,1])
        circuit.delay(CNOT_durations["[1, 0]"], [0, 1], "dt")
        circuit.x([3,4])
        circuit.delay(CNOT_durations["[4, 3]"], [3, 4], "dt")
    
    # Second SWAP gates.
    circuit.x([0,1])
    circuit.cnot(0,1)
    circuit.x([0,1])
    circuit.cnot(1,0)
    circuit.x([0,1])
    circuit.cnot(0,1)
    
    circuit.x([3,4])
    circuit.cnot(3,4)
    circuit.x([3,4])
    circuit.cnot(4,3)
    circuit.x([3,4])
    circuit.cnot(3,4)

    if wait2==True:
        circuit.x([0,1])
        circuit.delay(CNOT_durations["[1, 0]"], [0, 1], "dt")
        circuit.x([3,4])
        circuit.delay(CNOT_durations["[4, 3]"], [3, 4], "dt")
    
    circuit.measure(1,0)
    circuit.measure(3,1)

    tcircuit = transpile(circuit, backend, scheduling_method="asap", optimization_level=0)
    
    return tcircuit

def make_2swap_transpiled_circuit_full(bell_state=[0,0]):
    
    # With data decodification
    
    q  = QuantumRegister(5, 'q')
    c  = ClassicalRegister(2, 'c')
    circuit = QuantumCircuit(q, c)

    #circuit.reset([0,1,3,4])

    # Data gate (Bell pair creation gate).
    if bell_state[0] == 1:
        circuit.x(1)
    if bell_state[1] == 1:
        circuit.x(3)
    circuit.h(1)
    circuit.cnot(1,3)

    circuit.barrier(q)
    
    # First SWAP gates.
    circuit.cnot(0,1)
    circuit.cnot(1,0)
    circuit.cnot(0,1)
    circuit.cnot(3,4)
    circuit.cnot(4,3)
    circuit.cnot(3,4)
    
    # Second SWAP gates
    circuit.cnot(0,1)
    circuit.cnot(1,0)
    circuit.cnot(0,1)
    circuit.cnot(3,4)
    circuit.cnot(4,3)
    circuit.cnot(3,4)

    # Data gate (Bell pair creation gate).
    circuit.cnot(1,3)
    circuit.h(1)
    if bell_state[0] == 1:
        circuit.x(1)
    if bell_state[1] == 1:
        circuit.x(3)
        
    circuit.measure(1,0)
    circuit.measure(3,1)

    tcircuit = transpile(circuit, backend, scheduling_method="asap", optimization_level=0)
    
    return tcircuit


# **For checking that the circuit creation functions make the circuits correctly**

# In[33]:


make_2swapWithZs_transpiled_circuit(wait1=1, wait2=1).draw("mpl", fold=-1)


# ## Build the circuits

# In[6]:


repetitions = 50
bell_states = [[0,0], [0,1], [1,0], [1,1]]
transpiled_circuits = []
shots = 2**13 #8912

cases = ["1swap", "1swap-z", "1swap-z-delay", "1swap-x", "1swap-x-delay",
         "2swap", "2swap-z", "2swap-z-delay", "2swap-x", "2swap-x-delay", "2swap-full"]
# 11 cases

# 1 swap
for bell_state in bell_states:
    transpiled_circuits = transpiled_circuits + [make_swap_transpiled_circuit(bell_state)]*repetitions
    
# 1 swap - z 
for bell_state in bell_states:
    transpiled_circuits = transpiled_circuits + [make_swapWithZs_transpiled_circuit(bell_state)]*repetitions

# 1 swap - z - delay
for bell_state in bell_states:
    transpiled_circuits = transpiled_circuits + [make_swapWithZs_transpiled_circuit(bell_state, wait=True)]*repetitions    
    
# 1 swap - x 
for bell_state in bell_states:
    transpiled_circuits = transpiled_circuits + [make_swapWithXs_transpiled_circuit(bell_state)]*repetitions
    
# 1 swap - x - delay
for bell_state in bell_states:
    transpiled_circuits = transpiled_circuits + [make_swapWithXs_transpiled_circuit(bell_state, wait=True)]*repetitions

# 2 swap
for bell_state in bell_states:
    transpiled_circuits = transpiled_circuits + [make_2swap_transpiled_circuit(bell_state)]*repetitions

# 2 swap - z 
for bell_state in bell_states:
    transpiled_circuits = transpiled_circuits + [make_2swapWithZs_transpiled_circuit(bell_state)]*repetitions

# 2 swap - z - delay
for bell_state in bell_states:
    transpiled_circuits = transpiled_circuits + [make_2swapWithZs_transpiled_circuit(bell_state, wait1=True, wait2=True)]*repetitions

# 2 swap - x 
for bell_state in bell_states:
    transpiled_circuits = transpiled_circuits + [make_2swapWithXs_transpiled_circuit(bell_state)]*repetitions

# 2 swap - x - delay
for bell_state in bell_states:
    transpiled_circuits = transpiled_circuits + [make_2swapWithXs_transpiled_circuit(bell_state, wait1=True, wait2=True)]*repetitions

# 2 swap - data decodification
for bell_state in bell_states:
    transpiled_circuits = transpiled_circuits + [make_2swap_transpiled_circuit_full(bell_state)]*repetitions


# ## Send the job set to IBM

# In[76]:


job_manager = IBMQJobManager()
job_set = job_manager.run(transpiled_circuits, backend=backend, name="bellSwapTests_allCases", shots=shots)
#job_monitor(job_set)


# 12/08/2022
# 6fb31b4fa268469eb33fbeb001aa93c7-16602636328408778 # all 11 cases
# 84a98affb6444e6db3f58507de02994b-16603414103745694 # all 11 cases


# **For saving the job_set id for being able to retrieve it in the future.**

# In[ ]:


job_set_id = job_set.job_set_id()
print(job_set_id)


# **For checking the job status.**

# In[88]:


job_set.statuses()


# ## Define the fidelity function

# In[7]:


def fidelity(bell_state, counts, shots):
    
    counts = np.asarray(counts)
    
    if bell_state == "[0, 0]":
        f = (np.asarray(counts[0][0]) + np.asarray(counts[0][3]))/shots
    
    elif bell_state == "[0, 1]":
        f = (np.asarray(counts[1][1]) + np.asarray(counts[1][2]))/shots
    
    elif bell_state == "[1, 0]":
        f = (np.asarray(counts[2][0]) + np.asarray(counts[2][3]))/shots
        
    elif bell_state == "[1, 1]":
        f = (np.asarray(counts[3][1]) + np.asarray(counts[3][2]))/shots
    
    return f


def fidelity_full(bell_state, counts, shots):
    
    counts = np.asarray(counts)
    
    if bell_state == "[0, 0]":
        f = (np.asarray(counts[0][0]))/shots
    
    elif bell_state == "[0, 1]":
        f = (np.asarray(counts[1][0]))/shots
    
    elif bell_state == "[1, 0]":
        f = (np.asarray(counts[2][0]))/shots
        
    elif bell_state == "[1, 1]":
        f = (np.asarray(counts[3][0]))/shots
    
    return f


# ## Retrieve the job_set

# In[8]:


repetitions = 50
shots = 2**13
bell_states = [[0,0], [0,1], [1,0], [1,1]]
reshape_dims_circs = (11, 4, repetitions, 4)

job_manager = IBMQJobManager()
job_id = "job-set-id-here"
print("Getting job...")
job_set = job_manager.retrieve_job_set(job_id, provider)
print("Getting results...")
results_all = job_set.results()
print("Processing counts...")
all_counts_array = np.array([list(results_all.get_counts(i).values()) for i in range(11*4*repetitions)])
all_counts_array = [all_counts_array.reshape(reshape_dims_circs)[:5], # k=0 (1swap) 
                    all_counts_array.reshape(reshape_dims_circs)[5:]] # k=1 (2swap)
print("Done!")


# ## Plot the results

# In[9]:


fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(11,4), sharey=True)

data_labels = ["Without Z gates", "With Z gates"]
data_labels = ["No additional gates", "With Z gates", "With Z gates and wait", "With X gates", "With X gates and wait", "Bell undone"]
labels = ["$|\\beta_{00}\\rangle$", "$|\\beta_{01}\\rangle$", "$|\\beta_{10}\\rangle$", "$|\\beta_{11}\\rangle$"]
colors = ["#67a9cf", "#ef8a62", "#7fbf7b"]#["royalblue", "indianred"] #https://colorbrewer2.org/#type=diverging&scheme=RdBu&n=3
colors = ["C0", "C1", "C2", "C3", "C4", "C5"]
width = 0.29 # Width of the bars

x = np.arange(len(labels))*2 # For positioning the bars


for k in range(2):
    for i, counts_array in enumerate(all_counts_array[k]):

        xrange = x[-1] + 2*(width+0.4)
        
        if k==0:
            axs[k].set_title("1 Swap")
            axs[k].set_ylabel("Fidelity", fontsize=12)
        elif k==1:
            axs[k].set_title("2 Swaps")

        # Get the counts for the current initial state
        raw_countsBell00 = [counts_array[0][:,i] for i in range(4)]
        raw_countsBell01 = [counts_array[1][:,i] for i in range(4)]
        raw_countsBell10 = [counts_array[2][:,i] for i in range(4)]
        raw_countsBell11 = [counts_array[3][:,i] for i in range(4)]
        raw_bell_counts = [raw_countsBell00, raw_countsBell01, raw_countsBell10, raw_countsBell11]

        # Get the average of the repetitions
        countsBell00 = [np.round(np.average(counts_array[0][:,i])).astype(int) for i in range(4)]
        countsBell01 = [np.round(np.average(counts_array[1][:,i])).astype(int) for i in range(4)]
        countsBell10 = [np.round(np.average(counts_array[2][:,i])).astype(int) for i in range(4)]
        countsBell11 = [np.round(np.average(counts_array[3][:,i])).astype(int) for i in range(4)]
        bell_counts = [countsBell00, countsBell01, countsBell10, countsBell11]

        # Get the maximum count values of the repetitions
        max_countsBell00 = [np.max(counts_array[0][:,i]) for i in range(4)]
        max_countsBell01 = [np.max(counts_array[1][:,i]) for i in range(4)]
        max_countsBell10 = [np.max(counts_array[2][:,i]) for i in range(4)]
        max_countsBell11 = [np.max(counts_array[3][:,i]) for i in range(4)]
        max_bell_counts = [max_countsBell00, max_countsBell01, max_countsBell10, max_countsBell11]

        # Get the minimum count values of the repetitions
        min_countsBell00 = [np.min(counts_array[0][:,i]) for i in range(4)]
        min_countsBell01 = [np.min(counts_array[1][:,i]) for i in range(4)]
        min_countsBell10 = [np.min(counts_array[2][:,i]) for i in range(4)]
        min_countsBell11 = [np.min(counts_array[3][:,i]) for i in range(4)]
        min_bell_counts = [min_countsBell00, min_countsBell01, min_countsBell10, min_countsBell11]

        #print("Plotting job...")
        #ax.set_title(ax_titles[i])

        # Get the fidelities
        fidelities = []
        if k==1 and i==5:
            for j in range(4):
                fidelities.append(fidelity_full(str(bell_states[j]), bell_counts, shots))
        else:    
            for j in range(4):
                fidelities.append(fidelity(str(bell_states[j]), bell_counts, shots))
        
        # For placing the bars
        
        if i == 0: # 1 SWAP
            xi = x - 2*width
        elif i == 1:
            xi = x - width
        elif i == 2:
            xi = x
        elif i == 3:
            xi = x + width
        elif i == 4:
            xi = x + 2*width
        
        if k==1: # 2 SWAPs
            if i == 0:
                xi = x - 5*width/2
            elif i == 1:
                xi = x - 3*width/2
            elif i == 2:
                xi = x - width/2
            elif i == 3:
                xi = x + width/2
            elif i == 4:
                xi = x + 3*width/2
            elif i == 5:
                xi = x + 5*width/2

        barplot = axs[k].bar(xi, fidelities, color="white", alpha=1, zorder=5, width=width)
        barplot = axs[k].bar(xi, fidelities, color=colors[i], alpha=0.8, zorder=10, width=width,
                         error_kw={"color":"black", "capsize":5, "alpha":0.7}, label=data_labels[i])

        # Plot the black points, the fidelity of each repetition
        for j in range(4):
            if k==1 and i==5:
                fidelity_j = fidelity_full(str(bell_states[j]), raw_bell_counts, shots)
            else:
                fidelity_j = fidelity(str(bell_states[j]), raw_bell_counts, shots)
            xfid = np.ones(len(fidelity_j))*xi[j]
            axs[k].scatter(xfid, fidelity_j, color="black", s=8, marker="o", alpha=0.4, facecolor="k", edgecolors="k", zorder=26, linewidths=0)

        # https://moonbooks.org/Articles/How-to-add-text-on-a-bar-with-matplotlib-/
        
        for idx, rect in enumerate(barplot):
            height = rect.get_height()
            x_txt = rect.get_x() + rect.get_width()/1.8 
            y_txt = 0.685 
            txt = np.round(fidelities[idx], 3)
            axs[k].text(x_txt , y_txt, txt, ha='center', va='bottom', rotation=90, zorder=30, color="k", fontweight="medium")
        
    axs[k].set_xlim((-3.5*width,x[-1]+3.5*width))
    axs[k].set_ylim((0.68,0.9))
    axs[k].set_xticks(x, labels, fontsize=12)
    axs[k].grid(ls="--", alpha=0.4, zorder=0)
    for k, spine in axs[k].spines.items():  #ax.spines is a dictionary
        spine.set_zorder(50)

# Make the plot legend
legendEntries = data_labels
h1, l1 = axs[0].get_legend_handles_labels()
h2, l2 = axs[1].get_legend_handles_labels()
# Set figure legend entries, number of columns, location
fig.legend(h2, legendEntries, ncol=len(legendEntries), loc="lower center", bbox_to_anchor=(0.52, 0.02))

plt.tight_layout()

# Shrink the subplots to make room for the legend
box = axs[0].get_position()
axs[0].set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
box = axs[1].get_position()
axs[1].set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

fig.suptitle("Two qubit case: qubits 0, 1, 3 and 4 used", y=1.01)
plt.show()
#plt.savefig(r"lima_1and2swapFidelityTest_8192Shots_50Reps_12082022.pdf")


# In[ ]:




