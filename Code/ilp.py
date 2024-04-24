########### IMPORTS #################
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, TerminationCondition
### IMPORTS_FOR_TEST ###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def instanciate_ilp_model(weight_json: dict | str = {}):
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
            "max_amount_operators": 10,
            "make_span": 1,
            "lead_time": 1,
            "operators": 1,
            "earliness": 1,
            "tardiness": 2,
        }
    model = pyo.AbstractModel()
    # CONSTANTS
    BIG_M = 10000
    model.num_machines = pyo.Param(within=pyo.NonNegativeIntegers)
    model.num_opers = pyo.Param(within=pyo.NonNegativeIntegers)
    model.num_locked_opers = pyo.Param(within=pyo.NonNegativeIntegers)
    model.num_time_indices = pyo.Param(within=pyo.NonNegativeIntegers)
    model.num_orders = pyo.Param(within=pyo.NonNegativeIntegers)
    # RANGES
    model.machines = pyo.RangeSet(1, model.num_machines)
    model.opers = pyo.RangeSet(1, model.num_opers)
    model.orders = pyo.RangeSet(1, model.num_orders)
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
    model.is_init_order_in = pyo.Param(model.orders)
    model.is_final_order_in = pyo.Param(model.orders)
    model.is_oper_in_order = pyo.Param(model.opers, 
                                       model.orders)
    model.is_locked_in_order = pyo.Param(model.locked_opers, 
                                         model.orders)
    model.amount_operators = pyo.Param(model.opers)
    model.locked_amount_operators = pyo.Param(model.locked_opers)
    model.order_due_dates = pyo.Param(model.orders)
    model.orders_last_oper = pyo.Param(model.orders)
    # INITIAL VALUE OF VARIABLE
    model.previous_schedule = pyo.Param(model.machines, 
                                        model.opers, 
                                        model.time_indices,
                                        domain=pyo.Binary)
    # VARIABLES
    model.assigned = pyo.Var(model.machines, 
                             model.opers, 
                             model.time_indices, 
                            #  domain=pyo.Binary) 
                             domain=pyo.Binary, 
                             initialize=model.previous_schedule)
    model.orders_start_time = pyo.Var(model.orders, 
                                      domain=pyo.NonNegativeIntegers)
    model.orders_finished_time = pyo.Var(model.orders, 
                                         domain=pyo.NonNegativeIntegers)
    model.is_order_in_time = pyo.Var(model.orders, 
                                     domain=pyo.Binary)
    model.max_operators = pyo.Var(domain=pyo.NonNegativeIntegers)
    # model.earliness = pyo.Var(model.orders, 
    #                                  domain=pyo.NonNegativeIntegers)
    # model.tardiness = pyo.Var(model.orders, 
                                    #  domain=pyo.NonNegativeIntegers)

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

    def locked_overlap_after_const(model, machine, locked_oper, time_index):
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
    
    def locked_overlap_before_const(model, machine, oper, time_index):
        if (time_index == model.time_indices.at(-1)):
            return (pyo.Constraint.Skip)
        start_interval = min(time_index, model.time_indices.at(-2))
        end_interval = min(model.exec_time[oper] 
                            + time_index, model.time_indices.at(-1))
        time_interval = range(start_interval, end_interval) 
        unlocked = model.assigned[machine, oper, time_index]
        return sum(model.locked_schedule[machine, o, t] 
                    for o in model.locked_opers 
                    for t in time_interval) <= BIG_M*(1-unlocked)
        
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
    
    def earliest_order_oper_const(model, order, machine, oper, time_index):
        return (model.orders_start_time[order] <= BIG_M - (model.is_oper_in_order[oper, order]*(BIG_M - time_index*model.assigned[machine,oper,time_index])))
    
    def earliest_order_locked_const(model, order, machine, locked_oper, time_index):
        return (model.orders_start_time[order] <= BIG_M - (model.is_locked_in_order[locked_oper, order]*(BIG_M - time_index*model.locked_schedule[machine,locked_oper,time_index])))
    
    def last_order_oper_const(model, order, machine, oper, time_index):
        return (model.orders_finished_time[order] >= (model.is_oper_in_order[oper, order] * (time_index * model.assigned[machine,oper,time_index] + model.exec_time[oper])))
    
    def last_order_locked_const(model, order, machine, locked_oper, time_index):
        return (model.orders_finished_time[order] >= (model.is_locked_in_order[locked_oper, order] * (time_index*model.locked_schedule[machine,locked_oper,time_index] + model.locked_exec_time[locked_oper])))

    def max_operators_const(model, time_index):
        def get_time_interval(oper, time_index):
            start_interval = max(time_index - model.exec_time[oper], model.time_indices.at(1))
            end_interval = min(time_index, model.time_indices.at(-1))
            time_interval = range(start_interval, end_interval) 
            return (time_interval)
        def get_locked_time_interval(oper, time_index):
            start_interval = max(time_index - model.locked_exec_time[oper], model.time_indices.at(1))
            end_interval = min(time_index, model.time_indices.at(-1))
            time_interval = range(start_interval, end_interval) 
            return (time_interval)
    
        amount = sum(model.assigned[m, o, t]*model.amount_operators[o] 
                     for m in model.machines 
                     for o in model.opers
                     for t in get_time_interval(o,time_index))
        locked_amount = sum(model.locked_schedule[m, locked, t]*model.locked_amount_operators[locked] 
                            for m in model.machines 
                            for locked in model.locked_opers
                            for t in get_locked_time_interval(locked,time_index))
        return (model.max_operators >= amount + locked_amount - weight_json["max_amount_operators"])
            
    def order_in_time_const(model, order):
        # return (model.is_order_in_time[order] == min(1,max(0,model.orders_finished_time[order] - model.order_due_dates[order])))
        return (model.is_order_in_time[order] <= model.orders_finished_time[order] - model.order_due_dates[order])
    
    ################ OBJECTIVE_FUNCTION ######################
    def objective(model):
        def make_span_behaviour():
            return sum(t * model.assigned[m, o, t]  
                       for m in model.machines 
                       for o in model.opers 
                       for t in model.time_indices)
            
        def lead_time_behaviour():
            def calculate_max_time(order):
                return (model.is_final_order_in[order]*model.orders_finished_time[order])
            def calculate_min_time(order):
                return (-model.is_init_order_in[order]*model.orders_start_time[order])
            return (sum(calculate_max_time(order) - calculate_min_time(order) 
                        for order in model.orders))
            
        def operators_behaviour():
            return (model.max_operators)
    
        def earliness_behaviour():
            return (sum(model.is_order_in_time[order]*model.is_final_order_in[order]*(model.orders_finished_time[order] - model.order_due_dates[order])
                        for order in model.orders))
            
        def tardiness_behaviour():
            return (sum((1 - model.is_order_in_time[order])*model.is_final_order_in[order]*(model.order_due_dates[order] - model.orders_finished_time[order])
                        for order in model.orders))
            
        objective_fun = (weight_json["make_span"]*make_span_behaviour() 
                         + weight_json["lead_time"]*lead_time_behaviour()
                         + weight_json["operators"]*operators_behaviour()
                        #  + weight_json["earliness"]*earliness_behaviour()
                        #  + weight_json["tardiness"]*tardiness_behaviour()
                         )
        return (objective_fun)

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
    model.locked_overlap_after_const = pyo.Constraint(model.machines, 
                                                      model.locked_opers, 
                                                      model.time_indices, 
                                                      rule=locked_overlap_after_const)
    model.locked_overlap_before_const = pyo.Constraint(model.machines, 
                                                       model.opers, 
                                                       model.time_indices, 
                                                       rule=locked_overlap_before_const)
    model.precedence_const = pyo.Constraint(model.opers, 
                                            model.opers, 
                                            rule=precedence_const)
    model.locked_prece_before_const = pyo.Constraint(model.locked_opers, 
                                                     model.opers, 
                                                     rule=locked_prece_before_const)
    model.locked_prece_after_const = pyo.Constraint(model.opers, 
                                                    model.locked_opers, 
                                                    rule=locked_prece_after_const)
    model.earliest_order_oper_const = pyo.Constraint(model.orders,
                                                     model.machines,
                                                     model.opers,
                                                     model.time_indices,
                                                     rule=earliest_order_oper_const)
    model.earliest_order_locked_const = pyo.Constraint(model.orders,
                                                       model.machines,
                                                       model.locked_opers,
                                                       model.time_indices,
                                                       rule=earliest_order_locked_const)
    model.last_order_oper_const = pyo.Constraint(model.orders,
                                                 model.machines,
                                                 model.opers,
                                                 model.time_indices,
                                                 rule=last_order_oper_const)
    model.last_order_locked_const = pyo.Constraint(model.orders,
                                                   model.machines,
                                                   model.locked_opers,
                                                   model.time_indices,
                                                   rule=last_order_locked_const)
    model.order_in_time_const = pyo.Constraint(model.orders,
                                               rule=order_in_time_const)
    model.max_operators_const = pyo.Constraint(model.time_indices,
                                               rule=max_operators_const)
    return (model)

def run_ilp(model, ilp_data : dict | str, timelimit: int | None = None) -> None:
    """This function runs an abstract model with given instance data
    and returns the solved instance if the model.

    Args:
        model (pyo.AbstractModel): An abstract model.
        ilp_data (dict | str): The instance data.
        timelimit (int | None, optional): The time limit of the solution. Defaults to no time limit.
    """
    instance = model.create_instance(ilp_data)
    opt = SolverFactory("glpk")
    if (timelimit is None):
        opt.solve(instance)
        return (instance)
    results = opt.solve(instance)#, timelimit=timelimit)
    opt.options["TimeLimit"] = timelimit
    if (results.solver.termination_condition == TerminationCondition.maxTimeLimit):
        print("Maximum time limit reached")
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
    
    ilp_model = instanciate_ilp_model()
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