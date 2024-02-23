########### IMPORTS #################
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


############ CONSTRUCTING MODEL ################
# parameters and variable definition
model = pyo.AbstractModel()
model.num_machines = pyo.Param(within=pyo.NonNegativeIntegers)
model.num_operations = pyo.Param(within=pyo.NonNegativeIntegers)
model.machine_types = pyo.RangeSet(1, model.num_machines)
model.operations = pyo.RangeSet(1, model.num_operations)
model.valid_machines = pyo.Param(model.operations, model.machine_types)
model.operation_time = pyo.Param(model.operations)
model.machines = pyo.Var(model.operations, domain=pyo.NonNegativeIntegers)
model.time_indices = pyo.Var(model.operations, domain=pyo.NonNegativeIntegers)
#model.assigned = pyo.Var(model.machines, model.operations, model.time_indices, domain=pyo.Binary)
big_m = 10000

# objective function and constraints
def objective(model):
    return sum(model.time_indices[o] + model.operation_time[o] for o in model.operations)  

def machine_const(model, operation):
    machine = model.machines[operation]
    return machine%2 == 1
    #return model.valid_machines[operation, model.machine_types[machine]] == 1
    """
    statement = False
    for m in model.machine_types:
        if (m*model.valid_machines[operation, m] >= machine):
            statement = True
    return statement
    """
    #return model.valid_machines[operation, 1] >= machine

def overlap_const(model, operation, operation_ref):
    #if (operation == operation_ref):
    #    return True
    start_time = model.time_indices[operation]
    start_time_ref = model.time_indices[operation_ref]
    end_time_ref = model.time_indices[operation_ref] + model.operation_time[operation_ref]
    machine = model.machines[operation]
    machine_ref = model.machines[operation_ref]
    start_of_interval_term = max(0, max(0, start_time_ref - start_time))
    end_of_interval_term = max(0, max(0, end_time_ref - start_time))
    machine_term = max(machine - machine_ref, machine_ref - machine)
    operation_term = 1 - min(1, max(operation-operation_ref, operation_ref-operation))
    return start_of_interval_term + end_of_interval_term + machine_term + operation_term > 0

model.objective = pyo.Objective(rule=objective)
model.machine_const = pyo.Constraint(model.operations, rule=machine_const)
#model.overlap_const = pyo.Constraint(model.operations, model.operations, rule=overlap_const)

############# SOLVE ############
instance = model.create_instance('Data/Pyomo/pyo_data.dat')
instance.pprint()
opt = SolverFactory('glpk')
result = opt.solve(instance)

# data management
def instance_2_numpy(instance_data: pyo.Var | pyo.Param | pyo.Set | pyo.RangeSet, 
                     shape_array: np.ndarray | list = [] ) -> any:
    """Converts parameters, variables or ints that starts with "instance." and has a lower dimension than 4.
    The return will be a numpy array/matrix but just the value in the case of a single value (dimension of 0).
    In the case of a single value the shape_array should be an empty array "[]".

    Args:
        instance_data (pyomo.Var, pyomo.Param or pyomo.set): This is your input data ex. "instance.num_machines" and should always start with "instance.".
        shape_array (Array or np.array): This is the dimensionality of your input data ex. "[3,2,4]". What you would expect from "np.shape".
    """
    df = pd.DataFrame.from_dict(instance_data.extract_values(), orient='index', columns=[str(model.assigned)])
    solution_flat_matrix = df.to_numpy()
    if len(shape_array) == 0:
        return(solution_flat_matrix[0,0])
    if len(shape_array) == 1:
        return(solution_flat_matrix[:,0])
    solution_matrix = np.empty(shape_array)
    if len(shape_array) == 2:
        for i in range(shape_array[0]):
            for j in range(shape_array[1]):
                solution_matrix[i,j] = solution_flat_matrix[shape_array[1]*i + j,0]
    if len(shape_array) == 3:
        for i in range(shape_array[0]):
            for j in range(shape_array[1]):
                for k in range(shape_array[2]):
                    solution_matrix[i,j,k] = solution_flat_matrix[shape_array[1]*shape_array[2]*i + shape_array[2]*j + k,0]
    return(solution_matrix)

"""
instance_num_machines = instance_2_numpy(instance.num_machines)
instance_num_operations = instance_2_numpy(instance.num_operations)
instance_num_time_indices = instance_2_numpy(instance.num_time_indices)
instance_operation_time = instance_2_numpy(instance.operation_time, [instance_num_operations])
solution_shape = [instance_num_machines, instance_num_operations, instance_num_time_indices]
instance_solution_matrix = instance_2_numpy(instance.assigned, solution_shape)
"""

############# PLOT ############
def plot_gantt(gantt_matrix, operations_times): #operation_time?
    fig, ax = plt.subplots()
    gantt_dims = gantt_matrix.shape
    for operation in range(gantt_dims[1]):
        gantt_of_operation = gantt_matrix[:,operation,:]
        machine, start_of_operation = np.where(gantt_of_operation == 1)
        plt.barh(y=machine, width=operations_times[operation], left= start_of_operation)#, color=team_colors[row['team']], alpha=0.4)
    plt.title('Project Management Schedule of Project X', fontsize=15)
    plt.gca().invert_yaxis()
    ax.xaxis.grid(True, alpha=0.5)
    #ax.legend(handles=patches, labels=team_colors.keys(), fontsize=11)
    plt.show()
# plot_gantt(instance_solution_matrix, instance_operation_time)