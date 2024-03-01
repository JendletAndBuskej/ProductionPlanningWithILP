########### IMPORTS #################
from platform import machine
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
    model.num_opers = pyo.Param(within=pyo.NonNegativeIntegers)
    model.num_locked_opers = pyo.Param(within=pyo.NonNegativeIntegers)
    model.num_time_indices = pyo.Param(within=pyo.NonNegativeIntegers)
    model.machines = pyo.RangeSet(1, model.num_machines)
    model.opers = pyo.RangeSet(1, model.num_opers)
    model.locked_opers = pyo.RangeSet(1, model.num_locked_opers)
    model.all_opers = pyo.RangeSet(1,model.num_opers+model.num_locked_opers)
    model.time_indices = pyo.RangeSet(1, model.num_time_indices)
    model.valid_machines = pyo.Param(model.opers, model.machines)
    model.precedence = pyo.Param(model.all_opers, model.all_opers)
    model.exec_time = pyo.Param(model.opers)
    model.locked_exec_time = pyo.Param(model.locked_opers)
    model.locked_schedule = pyo.Param(model.machines, model.locked_opers, model.time_indices)
    model.assigned = pyo.Var(model.machines, model.opers, model.time_indices, domain=pyo.Binary)
    big_m = 10000


    # objective function and constraints
    def objective(model):
        return sum(t * model.assigned[m, o, t] 
                   for m in model.machines
                   for o in model.opers
                   for t in model.time_indices)  

    def duplicate_const(model, oper):
        return sum(model.assigned[m,oper,t]
                   for m in model.machines 
                   for t in model.time_indices) == 1

    def machine_const(model, machine, oper):
        valid_machine = model.valid_machines[oper, machine]
        return sum(model.assigned[machine,oper,t] 
                for t in model.time_indices) <= valid_machine

    def overlap_const(model, machine, oper, time_index):
        start_interval = min(time_index, model.time_indices[-2])
        end_interval = min(model.exec_time[oper] + time_index, model.time_indices[-1])
        time_interval = range(start_interval, end_interval) 
        locked_schedule = model.assigned[machine, oper, time_index]      
        return sum(model.assigned[machine, o, t] 
                   for o in model.opers 
                   for t in time_interval) <= 1 + big_m*(1-locked_schedule)

    def locked_overlap_const(model, machine, locked_oper, time_index):
        if (time_index == model.time_indices[-1]):
            return(pyo.Constraint.Skip)
        start_interval = min(time_index, model.time_indices[-2])
        end_interval = min(model.locked_exec_time[locked_oper] + time_index, model.time_indices[-1])
        time_interval = range(start_interval, end_interval) 
        locked = model.locked_schedule[machine, locked_oper, time_index]      
        return sum(model.assigned[machine, o, t] 
                   for o in model.opers 
                   for t in time_interval) <= big_m*(1-locked)
    
    def precedence_const(model, oper, other_oper):
        if (oper == other_oper):
            return(pyo.Constraint.Skip)
        start_time_oper = sum(t*model.assigned[m, other_oper, t] 
                             for m in model.machines 
                             for t in model.time_indices)
        end_time_sum = sum((t + model.exec_time[oper])*model.assigned[m, oper, t]
                             for m in model.machines 
                             for t in model.time_indices)
        end_time_other_oper = end_time_sum*model.precedence[oper,other_oper]
        return(start_time_oper >= end_time_other_oper)

    model.objective = pyo.Objective(rule=objective)
    model.no_duplicate = pyo.Constraint(model.opers, rule=duplicate_const)
    model.machine_const = pyo.Constraint(model.machines, model.opers, rule=machine_const)
    model.overlap_const = pyo.Constraint(model.machines, model.opers, model.time_indices, rule=overlap_const)
    model.locked_overlap_const = pyo.Constraint(model.machines, model.locked_opers, model.time_indices, rule=locked_overlap_const)
    #model.precedence_const = pyo.Constraint(model.opers, model.opers, rule=precedence_const)
    
    ############# SOLVE ############
    instance = model.create_instance(ilp_data)
    opt = SolverFactory("glpk")
    result = opt.solve(instance)
    return(instance)


if (__name__ == "__main__"):
    # https://readthedocs.org/projects/pyomo/downloads/pdf/stable/   
    # SIDA 69/743 för att se hur den skrivs
    test_dict = {
        None: {
            "num_machines" : {
                None: 2
            },
            "num_opers" : {
                None: 3
            },
            "num_locked_opers" : {
                None: 1
            },
            "num_time_indices" : {
                None: 5
            },
            "valid_machines" : {
                (1,1): 1,
                (1,2): 1,
                (2,1): 1,
                (2,2): 1,
                (3,1): 1,
                (3,2): 1,
            },
            "exec_time" : {
                1: 1,
                2: 2,
                3: 1,
            },
            "locked_schedule" : {
                (1,1,1): 1,
                (1,1,2): 0,
                (1,1,3): 0,
                (1,1,4): 0,
                (1,1,5): 0,
                (2,1,1): 0,
                (2,1,2): 0,
                (2,1,3): 0,
                (2,1,4): 0,
                (2,1,5): 0
            },
            "locked_exec_time" : {
                1: 2
            },
            "precedence" : {
                (1,1): 1,
                (1,2): 1,
                (1,3): 1,
                (2,1): 0,
                (2,2): 1,
                (2,3): 0,
                (3,1): 0,
                (3,2): 1,
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
        return (solution_matrix)

    instance_num_machines = instance_2_numpy(instance.num_machines)
    instance_num_opers = instance_2_numpy(instance.num_opers)
    instance_num_time_indices = instance_2_numpy(instance.num_time_indices)
    instance_operation_time = instance_2_numpy(instance.exec_time, [instance_num_opers])
    solution_shape = [instance_num_machines, instance_num_opers, instance_num_time_indices]
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