########### IMPORTS #################
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
### IMPORTS_FOR_TEST ###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def create_ilp(weight_json: dict | str = {}):
    """This function constructs an abstract ILP model of our problem.
    This constructions is done with a set of constrains and objective
    function parts that can be weighted in an json to get an desired result.

    Args:
    weight_json (dict | str): If specified this will change the weights on the objective
    functions.

    Returns:
        pyo.AbstractModel: an abstract ILP model.
    """
    if (len(weight_json) == 0):
        weight_json = {
            "make_span": 1
        }
    model = pyo.AbstractModel()
    # CONSTANTS
    BIG_M = 10000
    model.num_machines = pyo.Param(within=pyo.NonNegativeIntegers)
    model.num_opers = pyo.Param(within=pyo.NonNegativeIntegers)
    model.num_locked_opers = pyo.Param(within=pyo.NonNegativeIntegers)
    model.num_time_indices = pyo.Param(within=pyo.NonNegativeIntegers)
    # RANGES
    model.machines = pyo.RangeSet(1, model.num_machines)
    model.opers = pyo.RangeSet(1, model.num_opers)
    model.locked_opers = pyo.RangeSet(1, model.num_locked_opers)
    model.time_indices = pyo.RangeSet(1, model.num_time_indices)
    # PARAMETER
    model.valid_machines = pyo.Param(model.opers, 
                                     model.machines)
    model.precedence = pyo.Param(model.opers, 
                                 model.opers)
    model.locked_prece_before = pyo.Param(model.locked_opers, 
                                          model.opers)
    model.locked_prece_after = pyo.Param(model.opers, 
                                         model.locked_opers)
    model.exec_time = pyo.Param(model.opers)
    model.locked_exec_time = pyo.Param(model.locked_opers)
    model.locked_schedule = pyo.Param(model.machines, 
                                      model.locked_opers, 
                                      model.time_indices)
    # VARIABLE
    model.assigned = pyo.Var(model.machines, 
                             model.opers, 
                             model.time_indices, 
                             domain=pyo.Binary)

    #################### CONSTRAINTS ########################
    def duplicate_const(model, oper):
        return sum(model.assigned[m,oper,t]
                   for m in model.machines 
                   for t in model.time_indices) == 1

    def machine_const(model, machine, oper):
        valid_machine = model.valid_machines[oper, machine]
        if (valid_machine == 1):
            return (pyo.Constraint.Skip)
        return sum(model.assigned[machine,oper,t] 
                   for t in model.time_indices) <= valid_machine

    def overlap_const(model, machine, oper, time_index):
        if (time_index == model.time_indices.at(-1)):
            return (pyo.Constraint.Skip)
        start_interval = min(time_index, model.time_indices.at(-2))
        end_interval = min(model.exec_time[oper] + time_index, model.time_indices.at(-1))
        time_interval = range(start_interval, end_interval) 
        locked_schedule = model.assigned[machine, oper, time_index]      
        return sum(model.assigned[machine, o, t] 
                   for o in model.opers 
                   for t in time_interval) <= 1 + BIG_M*(1-locked_schedule)

    def locked_overlap_const(model, machine, locked_oper, time_index):
        if (time_index == model.time_indices.at(-1)):
            return (pyo.Constraint.Skip)
        start_interval = min(time_index, model.time_indices.at(-2))
        end_interval = min(model.locked_exec_time[locked_oper] 
                           + time_index, model.time_indices.at(-1))
        time_interval = range(start_interval, end_interval) 
        locked = model.locked_schedule[machine, locked_oper, time_index]
        return sum(model.assigned[machine, o, t] 
                   for o in model.opers 
                   for t in time_interval) <= BIG_M*(1-locked)
    
    def precedence_const(model, oper, other_oper):
        precedence = model.precedence[oper,other_oper]
        if (precedence == 0 or oper == other_oper):
            return (pyo.Constraint.Skip)
        start_time_oper = sum(t*model.assigned[m, other_oper, t] 
                              for m in model.machines 
                              for t in model.time_indices)
        end_time_sum = sum((t + model.exec_time[oper])*model.assigned[m, oper, t] 
                           for m in model.machines 
                           for t in model.time_indices)
        end_time_other_oper = end_time_sum*precedence
        return (start_time_oper >= end_time_other_oper)
    
    def locked_prece_after_const(model, oper, locked_oper):
        precedence = model.locked_prece_after[oper, locked_oper]
        if (precedence == 0):
            return (pyo.Constraint.Skip)
        start_time_locked = sum(t*model.locked_schedule[m, locked_oper, t] 
                                for m in model.machines 
                                for t in model.time_indices)
        end_time_sum = sum((t + model.exec_time[oper])*model.assigned[m, oper, t] 
                           for m in model.machines 
                           for t in model.time_indices)
        end_time_other_oper = end_time_sum*precedence
        return (start_time_locked >= end_time_other_oper)
    
    def locked_prece_before_const(model, locked_oper, oper):
        precedence = model.locked_prece_before[locked_oper, oper]
        if (precedence == 0):
            return (pyo.Constraint.Skip)
        start_time_oper = sum(t*model.assigned[m, oper, t] 
                              for m in model.machines 
                              for t in model.time_indices)
        exec_time = model.locked_exec_time[locked_oper]
        end_time_sum = sum((t + exec_time)*model.locked_schedule[m, locked_oper, t]
                           for m in model.machines 
                           for t in model.time_indices)
        end_time_locked_oper = end_time_sum*precedence
        return (start_time_oper >= end_time_locked_oper)

    ################ OBJECTIVE_FUNCTION ######################
    def objective(model):
        def make_span_behavior():
            return sum(t * model.assigned[m, o, t] 
                       for m in model.machines 
                       for o in model.opers 
                       for t in model.time_indices)
        
        make_span = weight_json["make_span"]*make_span_behavior()
        return (make_span)

    ############## SET_MODEL ###############
    model.objective = pyo.Objective(rule=objective)
    model.no_duplicate = pyo.Constraint(model.opers, 
                                        rule=duplicate_const)
    model.machine_const = pyo.Constraint(model.machines, 
                                         model.opers, 
                                         rule=machine_const)
    model.overlap_const = pyo.Constraint(model.machines, 
                                         model.opers, 
                                         model.time_indices, 
                                         rule=overlap_const)
    model.locked_overlap_const = pyo.Constraint(model.machines, 
                                                model.locked_opers, 
                                                model.time_indices, 
                                                rule=locked_overlap_const)
    model.precedence_const = pyo.Constraint(model.opers, 
                                            model.opers, 
                                            rule=precedence_const)
    model.locked_prece_before_const = pyo.Constraint(model.locked_opers, 
                                                     model.opers, 
                                                     rule=locked_prece_before_const)
    model.locked_prece_after_const = pyo.Constraint(model.opers, 
                                                    model.locked_opers, 
                                                    rule=locked_prece_after_const)
    return (model)

def run_ilp(model, ilp_data : dict | str):
    """This function runs an abstract model with given instance data
    and returns the solved instance if the model.

    Args:
        model (pyo.AbstractModel): An abstract model.
        ilp_data (dict | str): The instance data.
    """
    instance = model.create_instance(ilp_data)
    opt = SolverFactory("glpk")
    opt.solve(instance)
    return (instance)


#####################################
################ TEST #####################
#####################################
if (__name__ == "__main__"):
    # https://readthedocs.org/projects/pyomo/downloads/pdf/stable/   
    # SIDA 69/743 för att se hur den skrivs
    # test_dict = {
    #     None: {
    #         "num_machines" : {
    #             None: 2
    #         },
    #         "num_opers" : {
    #             None: 2
    #         },
    #         "num_locked_opers" : {
    #             None: 2
    #         },
    #         "num_time_indices" : {
    #             None: 5
    #         },
    #         "valid_machines" : {
    #             (1,1): 1,
    #             (1,2): 1,
    #             (2,1): 1,
    #             (2,2): 1,
    #         },
    #         "exec_time" : {
    #             1: 2,
    #             2: 2,
    #         },
    #         "locked_schedule" : {
    #             (1,1,1): 1,
    #             (1,1,2): 0,
    #             (1,1,3): 0,
    #             (1,1,4): 0,
    #             (1,1,5): 0,
    #             (2,1,1): 0,
    #             (2,1,2): 0,
    #             (2,1,3): 0,
    #             (2,1,4): 0,
    #             (2,1,5): 0,
    #             (1,2,1): 0,
    #             (1,2,2): 0,
    #             (1,2,3): 0,
    #             (1,2,4): 0,
    #             (1,2,5): 0,
    #             (2,2,1): 1,
    #             (2,2,2): 0,
    #             (2,2,3): 0,
    #             (2,2,4): 0,
    #             (2,2,5): 0
    #         },
    #         "locked_exec_time" : {
    #             1: 1,
    #             2: 1,
    #         },
    #         "precedence" : {
    #             (1,1): 0,
    #             (1,2): 0,
    #             (2,1): 0,
    #             (2,2): 0,
    #         },
    #         "locked_prece_before" : {
    #             (1,1): 1,
    #             (1,2): 0,
    #             (2,1): 0,
    #             (2,2): 1,
    #         },
    #         "locked_prece_after" : {
    #             (1,1): 0,
    #             (1,2): 0,
    #             (2,1): 0,
    #             (2,2): 0,
    #         }
    #     }
    # }
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
                (1,1,1): 0,
                (1,1,2): 0,
                (1,1,3): 1,
                (1,1,4): 0,
                (1,1,5): 0,
                (2,1,1): 0,
                (2,1,2): 0,
                (2,1,3): 0,
                (2,1,4): 0,
                (2,1,5): 0
            },
            "locked_exec_time" : {
                1: 1
            },
            "precedence" : {
                (1,1): 0,
                (1,2): 0,
                (1,3): 0,
                (2,1): 0,
                (2,2): 0,
                (2,3): 0,
                (3,1): 0,
                (3,2): 0,
                (3,3): 0,
            },
            "locked_prece_before" : {
                (1,1): 0,
                (1,2): 1,
                (1,3): 0,
            },
            "locked_prece_after" : {
                (1,1): 0,
                (2,1): 0,
                (3,1): 0,
            }
        }
    }
    
    ilp_model = create_ilp()
    instance = run_ilp(ilp_model, test_dict)
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
            plt.barh(y=machine, width=operations_times[operation], left= start_of_operation + 1)#, color=team_colors[row['team']], alpha=0.4)
        plt.title('Project Management Schedule of Project X', fontsize=15)
        plt.gca().invert_yaxis()
        ax.xaxis.grid(True, alpha=0.5)
        # ax.legend(handles=patches, labels=team_colors.keys(), fontsize=11)
        plt.show()
    plot_gantt(instance_solution_matrix, instance_operation_time)