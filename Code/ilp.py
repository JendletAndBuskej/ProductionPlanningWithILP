########### IMPORTS #################
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_and_run_ilp(ilp_data : dict | str):
    ############ CONSTRUCTING MODEL ################
    # parameters and variable definition
    model = pyo.AbstractModel()
    model.num_machines = pyo.Param(within=pyo.NonNegativeIntegers)
    model.num_operations = pyo.Param(within=pyo.NonNegativeIntegers)
    model.num_locked_operations = pyo.Param(within=pyo.NonNegativeIntegers)
    model.num_time_indices = pyo.Param(within=pyo.NonNegativeIntegers)
    model.machines = pyo.RangeSet(1, model.num_machines)
    model.oper = pyo.RangeSet(1, model.num_operations)
    model.locked_oper = pyo.RangeSet(1, model.num_locked_operations)
    model.time_indices = pyo.RangeSet(1, model.num_time_indices)
    model.valid_machines = pyo.Param(model.oper, model.machines)
    model.exec_time = pyo.Param(model.oper)
    model.locked_oper_machine = pyo.Param(model.locked_oper)
    model.locked_oper_start_time = pyo.Param(model.locked_oper)
    model.locked_oper_exec_time = pyo.Param(model.locked_oper)
    model.assigned = pyo.Var(model.machines, model.oper, model.time_indices, domain=pyo.Binary)
    big_m = 10000


    # objective function and constraints
    def objective(model):
        return sum(t * model.assigned[m, o, t] 
                   for m in model.machines
                   for o in model.oper
                   for t in model.time_indices)  

    def duplicate_const(model, operation):
        return sum(model.assigned[m,operation,t]
                   for m in model.machines 
                   for t in model.time_indices) == 1

    def machine_const(model, machine, operation):
        valid_machine = model.valid_machines[operation, machine]
        return sum(model.assigned[machine,operation,t] 
                for t in model.time_indices) <= valid_machine

    def overlap_const(model, machine, operation, time_index):
        start_interval = min(time_index, model.time_indices[-2])
        end_interval = min(model.exec_time[operation] + time_index, model.time_indices[-1])
        time_interval = range(start_interval, end_interval) 
        assigned = model.assigned[machine, operation, time_index]      
        return sum(model.assigned[machine, o, t] 
                   for o in model.oper 
                   for t in time_interval) <= 1 + big_m*(1-assigned)

    def locked_scheme_const(model, locked_oper):
        machine = model.locked_oper_machine[locked_oper]
        last_time_index = model.time_indices[-1]
        second_to_last_time_index = model.time_indices[-2]
        start_interval = min(second_to_last_time_index, model.locked_oper_start_time[locked_oper])
        oper_end_time = model.locked_oper_start_time[locked_oper] + model.locked_oper_exec_time[locked_oper]
        end_interval = min(last_time_index, oper_end_time)
        time_interval = range(start_interval, end_interval)
        return sum (model.assigned[machine, o, t]
                    for o in model.oper
                    for t in time_interval) <= 0

    model.objective = pyo.Objective(rule=objective)
    model.no_duplicate = pyo.Constraint(model.oper, rule=duplicate_const)
    model.machine_const = pyo.Constraint(model.machines, model.oper, rule=machine_const)
    model.overlap_const = pyo.Constraint(model.machines, model.oper, model.time_indices, rule=overlap_const)
    model.locked_scheme_const = pyo.Constraint(model.locked_oper, rule=locked_scheme_const)
    
    ############# SOLVE ############
    instance = model.create_instance(ilp_data)
    opt = SolverFactory("glpk")
    result = opt.solve(instance)
    return instance


if (__name__ == "__main__"):
    # https://readthedocs.org/projects/pyomo/downloads/pdf/stable/   
    # SIDA 69/743 för att se hur den skrivs 
    test_dict = {
        None: {
            "num_machines" : {
                None: 2
            },
            "num_operations" : {
                None: 2
            },
            "num_locked_operations" : {
                None: 1
            },
            "num_time_indices" : {
                None: 4
            },
            "valid_machines" : {
                (1,1): 1,
                (1,2): 0,
                (2,1): 1,
                (2,2): 1
            },
            "exec_time" : {
                1: 1,
                2: 2
            },
            "locked_oper_machine" : {
                1: 1
            },
            "locked_oper_start_time" : {
                1: 1
            },
            "locked_oper_exec_time" : {
                1: 2
            }
        }
    }
    
    instance = create_and_run_ilp(test_dict)
    #instance = create_and_run_ilp("Data/Pyomo/pyo_data.dat")
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
        df = pd.DataFrame.from_dict(instance_data.extract_values(), orient='index')
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
    instance_operation_time = instance_2_numpy(instance.exec_time, [instance_num_operations])
    solution_shape = [instance_num_machines, instance_num_operations, instance_num_time_indices]
    instance_solution_matrix = instance_2_numpy(instance.assigned, solution_shape)

    ########### PLOT ############
    def plot_gantt(gantt_matrix, operations_times): #exec_time?
        fig, ax = plt.subplots()
        gantt_dims = gantt_matrix.shape
        for operation in range(gantt_dims[1]):
            gantt_of_operation = gantt_matrix[:,operation,:]
            machine, start_of_operation = np.where(gantt_of_operation == 1)
            plt.barh(y=machine, width=operations_times[operation], left= start_of_operation)#, color=team_colors[row['team']], alpha=0.4)
        plt.title('Project Management Schedule of Project X', fontsize=15)
        plt.gca().invert_yaxis()
        ax.xaxis.grid(True, alpha=0.5)
        # ax.legend(handles=patches, labels=team_colors.keys(), fontsize=11)
        plt.show()
    plot_gantt(instance_solution_matrix, instance_operation_time)