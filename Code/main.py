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
model.num_time_indices = pyo.Param(within=pyo.NonNegativeIntegers)
model.machines = pyo.RangeSet(1, model.num_machines)
model.operations = pyo.RangeSet(1, model.num_operations)
model.time_indices = pyo.RangeSet(1, model.num_time_indices)
model.operation_valid_machine_types = pyo.Param(model.operations, model.machines)
model.operation_time = pyo.Param(model.operations)
model.assigned = pyo.Var(model.machines, model.operations, model.time_indices, domain=pyo.Binary)
big_m = 10000

# objective function and constraints
def objective_rule(model):
    #return max(t * model.assigned[m, o, t] for m in model.machines for o in model.operations for t in model.time_indices)  
    return sum(t * model.assigned[m, o, t] 
               for m in model.machines
               for o in model.operations
               for t in model.time_indices)  

def no_duplicate_const(model, operation):
    return sum(model.assigned[m,operation,t]
               for m in model.machines 
               for t in model.time_indices) == 1

def machine_type_const(model, machine, operation):
    return sum(model.assigned[machine,operation,t] 
            for t in model.time_indices) <= model.operation_valid_machine_types[operation, machine]

def overlap_const(model, machine, operation, time_index):
    time_interval = (range(min(time_index,model.time_indices[-2]), 
                           min(model.operation_time[operation] + time_index, model.time_indices[-1])))
    return sum(model.assigned[machine, o, t] 
               for o in model.operations 
               for t in time_interval) <= 1+big_m*(1-model.assigned[machine, operation, time_index])

model.objective = pyo.Objective(rule=objective_rule)
model.no_duplicate = pyo.Constraint(model.operations, rule=no_duplicate_const)
model.machine_type_const = pyo.Constraint(model.machines, model.operations, rule=machine_type_const)
model.overlap_const = pyo.Constraint(model.machines, model.operations, model.time_indices, rule=overlap_const)

############# SOLVE ############
instance = model.create_instance('Data/Pyomo/pyo_data.dat')
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

instance_num_machines = instance_2_numpy(instance.num_machines)
instance_num_operations = instance_2_numpy(instance.num_operations)
instance_num_time_indices = instance_2_numpy(instance.num_time_indices)
instance_operation_time = instance_2_numpy(instance.operation_time, [instance_num_operations])
solution_shape = [instance_num_machines, instance_num_operations, instance_num_time_indices]
instance_solution_matrix = instance_2_numpy(instance.assigned, solution_shape)

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
plot_gantt(instance_solution_matrix, instance_operation_time)