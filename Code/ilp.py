########### IMPORTS #################
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, TerminationCondition

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
    # if (len(weight_json) == 0):
    #     weight_json = {
    #         "max_amount_operators": 4,
    #         "make_span": 1,
    #         "lead_time": 1,
    #         "operators": 1,
    #         "earliness": 0,
    #         "tardiness": 0,
    #     }
    model = pyo.AbstractModel()
    # CONSTANTS
    BIG_M = 10000
    model.num_machines = pyo.Param(within=pyo.NonNegativeIntegers)
    model.num_opers = pyo.Param(within=pyo.NonNegativeIntegers)
    model.num_locked_opers = pyo.Param(within=pyo.NonNegativeIntegers)
    model.num_time_indices = pyo.Param(within=pyo.NonNegativeIntegers)
    model.num_orders = pyo.Param(within=pyo.NonNegativeIntegers)
    model.time_index_to_real = pyo.Param(within=pyo.NonNegativeIntegers)
        #balance_json
    model.balance_make_span = pyo.Param(within=pyo.NonNegativeReals)
    model.balance_make_real = pyo.Param(within=pyo.NonNegativeReals)
    model.balance_lead_time = pyo.Param(within=pyo.NonNegativeReals)
    model.balance_lead_fake = pyo.Param(within=pyo.NonNegativeReals)
    model.balance_operator = pyo.Param(within=pyo.NonNegativeReals)
    model.balance_operator_fake = pyo.Param(within=pyo.NonNegativeReals)
    model.balance_earliness = pyo.Param(within=pyo.NonNegativeReals)
    model.balance_tardiness = pyo.Param(within=pyo.NonNegativeReals)
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
    model.is_final_oper_in = pyo.Param(domain=pyo.Binary)
    model.is_oper_in_order = pyo.Param(model.opers, 
                                       model.orders)
    model.is_locked_in_order = pyo.Param(model.locked_opers, 
                                         model.orders)
    model.amount_operators = pyo.Param(model.opers)
    model.locked_amount_operators = pyo.Param(model.locked_opers)
    model.order_due_dates = pyo.Param(model.orders)
    model.orders_last_oper = pyo.Param(model.orders)
    model.last_oper_indices = pyo.Param(model.orders, 
                                        domain=pyo.NonNegativeIntegers)
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
                                      bounds=(0,model.num_time_indices),
                                      domain=pyo.NonNegativeIntegers)
    # model.orders_finished_time = pyo.Var(model.orders, 
    #                                      bounds=(0,model.num_time_indices),
    #                                      domain=pyo.NonNegativeIntegers)
    # model.is_order_in_time = pyo.Var(model.orders, 
    #                                  domain=pyo.Binary)
    model.max_operators = pyo.Var(domain=pyo.NonNegativeIntegers)
    model.operators_per_time = pyo.Var(model.time_indices, 
                                           domain=pyo.NonNegativeIntegers)
    model.earliness = pyo.Var(model.orders, 
                              bounds=(0,model.num_time_indices),
                              domain=pyo.NonNegativeIntegers)
    # model.max_time = pyo.Var(domain=pyo.NonNegativeIntegers)
   
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

    def spill_over_const(model, oper):
        return (sum(model.assigned[m, oper, t] * (t + model.exec_time[oper])
                    for m in model.machines
                    for t in model.time_indices) <= model.time_indices.at(-1))
    
    def overlap_const(model, machine, oper, time_index):
        start_interval = min(time_index, model.time_indices.at(-2))
        end_interval = min(model.exec_time[oper] + time_index, model.time_indices.at(-1))
        time_interval = range(start_interval, end_interval) 
        locked_schedule = model.assigned[machine, oper, time_index]      
        return sum(model.assigned[machine, o, t] 
                   for o in model.opers 
                   for t in time_interval) <= 1 + BIG_M*(1-locked_schedule)
        
    def locked_overlap_after_const(model, machine, locked_oper, time_index):
        start_interval = min(time_index, model.time_indices.at(-2))
        end_interval = min(model.locked_exec_time[locked_oper] 
                           + time_index, model.time_indices.at(-1))
        time_interval = range(start_interval, end_interval) 
        locked = model.locked_schedule[machine, locked_oper, time_index]
        return sum(model.assigned[machine, o, t] 
                   for o in model.opers 
                   for t in time_interval) <= BIG_M*(1-locked)
    
    def locked_overlap_before_const(model, machine, oper, time_index):
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
    
    def earliest_order_oper_const(model, order, oper):
        return (model.orders_start_time[order] <= BIG_M + sum(model.assigned[m,oper,t]*model.is_oper_in_order[oper, order]*(t - BIG_M)
                                                              for m in model.machines
                                                              for t in model.time_indices))
                
                # (model.is_oper_in_order[oper, order]*(BIG_M - time_index*model.assigned[machine,oper,time_index])))
    
    def earliest_order_locked_const(model, order, locked_oper):
        return (model.orders_start_time[order] <= BIG_M + sum(model.locked_schedule[m,locked_oper,t]*model.is_locked_in_order[locked_oper, order]*(t - BIG_M)
                                                              for m in model.machines
                                                              for t in model.time_indices))
                
                # (model.is_locked_in_order[locked_oper, order]*(BIG_M - time_index*model.locked_schedule[machine,locked_oper,time_index])))
    
    #def last_order_oper_const(model, order, machine, oper, time_index):
        #return (model.orders_finished_time[order] >= (model.is_oper_in_order[oper, order] * (time_index * model.assigned[machine,oper,time_index] + model.exec_time[oper])))
    # def last_order_oper_const(model, order, oper):
    #     return (model.orders_finished_time[order] >= sum(model.assigned[m,oper,t]*model.is_oper_in_order[oper, order]*(t + model.exec_time[oper])
    #                 for m in model.machines
    #                 for t in model.time_indices))
    
    # def last_order_locked_const(model, order, locked_oper):
    #     return (model.orders_finished_time[order] >= sum(model.locked_schedule[m,locked_oper,t]*model.is_locked_in_order[locked_oper, order]*(t + model.locked_exec_time[locked_oper])
    #                 for m in model.machines
    #                 for t in model.time_indices))

    # def last_order_locked_const(model, order, machine, locked_oper, time_index):
        # return (model.orders_finished_time[order] >= (model.is_locked_in_order[locked_oper, order] * (time_index*model.locked_schedule[machine,locked_oper,time_index] + model.locked_exec_time[locked_oper])))

    def max_operators_const(model, time_index):
        def get_time_interval(oper, time_index):
            start_interval = max(time_index - model.exec_time[oper], model.time_indices.at(1))
            end_interval = time_index
            time_interval = range(start_interval, end_interval) 
            return (time_interval)
        def get_locked_time_interval(oper, time_index):
            start_interval = max(time_index - model.locked_exec_time[oper], model.time_indices.at(1))
            end_interval = time_index
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
    
    def operators_const(model, time_index):
        def get_time_interval(oper, time_index):
            start_interval = max(time_index - model.exec_time[oper], model.time_indices.at(1))
            end_interval = time_index
            time_interval = range(start_interval + 1, end_interval + 1) 
            return (time_interval)
        def get_locked_time_interval(oper, time_index):
            start_interval = max(time_index - model.locked_exec_time[oper], model.time_indices.at(1))
            end_interval = time_index
            time_interval = range(start_interval + 1, end_interval + 1) 
            return (time_interval)

        amount = sum(model.assigned[m, o, t]*model.amount_operators[o] 
                        for m in model.machines 
                        for o in model.opers
                        for t in get_time_interval(o,time_index))
        locked_amount = sum(model.locked_schedule[m, locked, t]*model.locked_amount_operators[locked] 
                            for m in model.machines 
                            for locked in model.locked_opers
                            for t in get_locked_time_interval(locked,time_index))
        return (model.operators_per_time[time_index] >= amount + locked_amount - weight_json["max_amount_operators"])

    def earliness_help_const(model, order):
        if (model.last_oper_indices[order] == 0):
            return (pyo.Constraint.Skip)
        oper = model.last_oper_indices[order]
        return (model.earliness[order] >= model.order_due_dates[order] - sum(model.assigned[m,oper,t]*(t + model.exec_time[oper])
                                                                             for m in model.machines
                                                                             for t in model.time_indices))
    
    # def earliness_locked_help_const(model, locked_oper, order):
    #     return (model.earliness[order] >= sum(model.is_locked_in_order[locked_oper,order]*model.locked_schedule[m,locked_oper,t]
    #                                           *(model.order_due_dates[order] - t - model.locked_exec_time[locked_oper])
    #                                           for m in model.machines
    #                                           for t in model.time_indices))
            
    # def last_time_const(model, time_index):
    #     def get_activity(m, o, t):
    #         return(model.assigned[m,o,t] + model.locked_schedule[m,o,t])
    #     return (model.last_time >= sum(time_index*get_activity(m, o, time_index)
    #                                    for m in model.machines
    #                                    for o in model.opers))

    
    ################ OBJECTIVE_FUNCTION ######################
    def objective(model):
        balance_json = {
            "make_span": model.balance_make_span,
            "make_span_real": model.balance_make_real,
            "lead_time": model.balance_lead_time,
            "lead_time_fake": model.balance_lead_fake,
            "operators": model.balance_operator,
            "fake_operators": model.balance_operator_fake,
            "earliness": model.balance_earliness,
            "tardiness": model.balance_tardiness,
        }

        def make_span_behaviour():
            # return (balance_json["make_span"]*sum(t * model.assigned[m, o, t]  
            #            for m in model.machines 
            #            for o in model.opers 
            #            for t in model.time_indices))
            return (sum(t * model.assigned[m, o, t]  
                       for m in model.machines 
                       for o in model.opers 
                       for t in model.time_indices))
        # def make_span_real_behaviour():
        #     #needs to set last_time_const(time_indeces)
        #     return (balance_json["make_span_real"]*model.is_final_oper_in*model.last_time)
            
        def lead_time_behaviour():
            def calculate_max_time(order):
                oper = model.last_oper_indices[order]
                if (oper == 0):
                    return (0)
                return (model.is_final_order_in[order]*sum(model.assigned[m,oper,t]*(t + model.exec_time[oper])
                                                           for m in model.machines
                                                           for t in model.time_indices))
            def calculate_min_time(order):
                return (model.is_init_order_in[order]*model.orders_start_time[order])
            return (balance_json["lead_time"]*sum(calculate_max_time(order) - calculate_min_time(order) 
                        for order in model.orders))
            
        def operators_behaviour():
            return (balance_json["operators"]*model.max_operators)
        
        def fake_operators_behaviour():
            return (balance_json["fake_operators"]*sum(model.operators_per_time[t]
                                                       for t in model.time_indices))
    
        def earliness_behaviour():
            return (balance_json["earliness"]*sum(model.is_final_order_in[order]*model.earliness[order]
                        for order in model.orders))
            
        def tardiness_behaviour():
            def last_oper_end_time(order):
                oper = model.last_oper_indices[order]
                if (oper == 0):
                    return (0)
                last_time = sum(model.assigned[m,oper,t]*(t + model.exec_time[oper])
                                for m in model.machines
                                for t in model.time_indices)
                return(last_time)
            return (balance_json["tardiness"]*sum(model.is_final_order_in[order]*(last_oper_end_time(order) - model.order_due_dates[order])
                        for order in model.orders))
            
        objective_fun = (0
                         + weight_json["make_span"]*make_span_behaviour() 
                         + weight_json["lead_time"]*lead_time_behaviour()
                         + weight_json["operators"]*operators_behaviour()
                         + weight_json["fake_operators"]*fake_operators_behaviour()
                         + (weight_json["earliness"] + weight_json["tardiness"])*earliness_behaviour()
                         + weight_json["tardiness"]*tardiness_behaviour()
                         )
        return (objective_fun)

    ############## SET_MODEL ###############
    model.objective = pyo.Objective(rule=objective,
                                    sense=pyo.minimize)
    model.no_duplicate = pyo.Constraint(model.opers, 
                                        rule=duplicate_const)
    model.machine_const = pyo.Constraint(model.machines, 
                                         model.opers, 
                                         rule=machine_const)
    model.spill_over_const = pyo.Constraint(model.opers,
                                            rule=spill_over_const) 
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
                                                    #  model.machines,
                                                     model.opers,
                                                    #  model.time_indices,
                                                     rule=earliest_order_oper_const)
    model.earliest_order_locked_const = pyo.Constraint(model.orders,
                                                    #    model.machines,
                                                       model.locked_opers,
                                                    #    model.time_indices,
                                                       rule=earliest_order_locked_const)
    # model.last_order_oper_const = pyo.Constraint(model.orders,
    #                                             #  model.machines,
    #                                              model.opers,
    #                                             #  model.time_indices,
    #                                              rule=last_order_oper_const)
    # model.last_order_locked_const = pyo.Constraint(model.orders,
    #                                             #    model.machines,
    #                                                model.locked_opers,
    #                                             #    model.time_indices,
    #                                                rule=last_order_locked_const)
    model.earliness_help_const = pyo.Constraint(
                                                # model.machines, 
                                                # model.opers, 
                                                # model.time_indices, 
                                                model.orders,
                                                rule=earliness_help_const)
    # model.earliness_locked_help_const = pyo.Constraint(
    #                                                    # model.machines, 
    #                                                    model.locked_opers, 
    #                                                    # model.time_indices, 
    #                                                    model.orders,
    #                                                    rule=earliness_locked_help_const)
    # model.earliness_pos_const = pyo.Constraint(model.orders,
    #                                            rule=earliness_pos_const)
    model.max_operators_const = pyo.Constraint(model.time_indices,
                                               rule=max_operators_const)
    model.operators_const = pyo.Constraint(model.time_indices, 
                                           rule=operators_const)
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
    solver = SolverFactory("glpk")
    if (timelimit is None):
        solver.solve(instance, tee=False)
        return (instance)
    solver.options["tmlim"] = timelimit
    results = solver.solve(instance, tee=False)
    # print(pyo.value(instance.obj))
    if (results.solver.termination_condition == TerminationCondition.maxTimeLimit):
        print("Maximum time limit reached")
    return (instance)
