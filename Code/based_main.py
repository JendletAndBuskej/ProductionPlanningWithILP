######### IMPORTS ##############
import numpy as np
import matplotlib.pyplot as plt
import os, json, math, random
from classes import Operation, Order
from environment import Environment
from Data_Converter.batch_data import BatchData
from ilp import instanciate_ilp_model, run_ilp
import pyomo as pyo
from pyomo.opt import SolverFactory
import pandas as pd
import time

########## PARAMETERS #############
size_steps = 1
t_max = None
iter_per_size =  np.ones([size_steps])
iter_per_size[0] = 1
num_orders = 10
num_machines = 35
num_machine_unlock_amount = 1
num_order_unlock_amount = 1
weight_json = {
            "max_amount_operators": 4,
            "make_span": 0,
            "lead_time": 0, 
            "operators": 1,
            "fake_operators": 1,
            "earliness": 0,
            "tardiness": 0,
        }

####### INITIALIZE ########
schedule_length = 0
data_bank = BatchData(num_orders)
input_json = data_bank.get_batch()
input_json = data_bank.generate_due_dates("",[1,1])
#data_bank.save_as_json(input_json, "/Parsed_Json/batched.json")
with open("/home/buske/Progress/ProductionPlanningWithILP/Data/Parsed_Json/batched_same_operator.json") as f:
    input_json = json.load(f)
env = Environment(input_json, weight_json)
time_length = math.ceil(len(env.time_line))
total_loops = int(2*sum(iter_per_size))
loops_count = 0
old_run_ob_val = 10000000000000000

init_obj_value = env.get_objective_value(weight_json)
env.plot()


###### HELP_FUNCTIONS #######
def CalculateComplexity(nr_time_steps, nr_unlock_oper):
    """Calculates if the complexity of the planned unlocked ILP is smaller or greater than a good reference.
    This returns true if the reference has a lower complexity and false otherwise. nr_time_steps_good_test and 
    nr_oper_good_test is used to calculate the good reference value.

    Args:
        nr_time_steps (int): this is the planed amount of unlocked time steps.
        nr_unlock_oper (int): this is the planed amount of unlocked operations.
    """
    nr_time_steps_good_test = 28
    nr_oper_good_test = 7
    max_permutations = math.factorial(nr_oper_good_test)*math.comb(num_machines*nr_time_steps_good_test, nr_oper_good_test)

    possibilities = (math.factorial(nr_unlock_oper) + nr_unlock_oper)*math.comb(num_machines*nr_time_steps,nr_unlock_oper)
    if (possibilities > max_permutations):
        return(True)
    return(False)

def run_t_interval(type, env, type_list, t_interval):
    obj_value = env.get_objective_value(weight_json)
    env.plot(real_size = True, save_plot = True)

    print(obj_value[0])
    sub_run_counter = 0
    t_interval2 = [t_interval[0], t_interval[1]]
    if (type == "machine"):
        nr_types = num_machines
    elif (type == "order"):
        nr_types = num_orders
    else:
        print("not a valid type")
        return()
    while (t_interval2[0] != t_interval2[1]):
        nr_time_steps = int(t_interval2[1] - t_interval2[0])
        unlock, lock = env.unlock(0, t_interval2, type, type_list)
        if not (CalculateComplexity(nr_time_steps, len(unlock))):
            if (sub_run_counter > 0):
                print("       sub run: "  + str(sub_run_counter + 1))
            if (len(unlock) == 0):
                return()
            dict = env.to_ilp(unlock, lock, t_interval2)
            csv_save(True, unlock, lock, env)
            instance = env.run_ilp_instance(dict, timelimit=t_max)
            env.update_from_ilp_solution(instance, t_interval2)
            csv_save(False, unlock, lock, env)
            return ()
        if (sub_run_counter == 0):
            print("       split time axis: ")
        time_cut = int(t_interval2[0])
        step_length = int(nr_time_steps)
        for i in range(nr_time_steps):
            step_length = int(np.floor(step_length/2))
            if (step_length <= 0):
                while(CalculateComplexity(nr_time_steps, len(unlock))):
                    time_cut += 1
                    nr_time_steps = int(t_interval2[1] - time_cut)
                    unlock, lock = env.unlock(0, [time_cut, t_interval2[1]], type, type_list)
                sub_run_counter += 1
                print("       sub run: " + str(sub_run_counter))
                print("time_interval: " + str(t_interval2[1]) + "," + str(time_cut))
                break
            nr_time_steps = int(t_interval2[1] - time_cut)
            unlock, lock = env.unlock(0, [time_cut, t_interval2[1]], type, type_list)
            if not (CalculateComplexity(nr_time_steps, len(unlock))):
                time_cut -= step_length
            else:
                time_cut += step_length
            time_cut = int(time_cut)
        unlock, lock = env.unlock(0 , [time_cut, t_interval2[1]], type, type_list)
        if (len(unlock) == 0):
            unlock, lock = env.unlock(0, t_interval2, type, type_list)
            RunSemiBatch(env, unlock, lock, t_interval2)
            print("deadlock saved")
            return(0)
        if (nr_time_steps <= 1):
            print("trying to unlock to many operations at one time, skipping this iteration and continues")
            return()
        dict = env.to_ilp(unlock, lock, [time_cut, t_interval2[1]])
        instance = env.run_ilp_instance(dict, timelimit=t_max)
        env.update_from_ilp_solution(instance, [time_cut, t_interval2[1]])
        t_interval2[1] = int(np.floor((t_interval2[1] - time_cut)/2) + time_cut)

def RunSemiBatch(env, unlock, lock, t_interval):
    """this calculates if the complexity of the planned ILP run is to big and if so it will run multiple smaller
    ILP solves to get a resonable run time.

    Args:
        env (_type_): _description_
        unlock (_type_): _description_
        lock (_type_): _description_
        t_interval (_type_): _description_
    """

    unlock = np.array(unlock)
    nr_time_steps = t_interval[1] - t_interval[0]
    nr_unlock = len(unlock)
    too_big = CalculateComplexity(nr_time_steps, nr_unlock)
    if not (too_big):
        dict = env.to_ilp(unlock, lock, t_interval)
        instance = env.run_ilp_instance(dict, timelimit=t_max)
        env.update_from_ilp_solution(instance, t_interval)
        return()
    preferred_unlock_amount = nr_unlock
    while (CalculateComplexity(nr_time_steps, preferred_unlock_amount)):
        preferred_unlock_amount -= 1
    done_matrix = np.zeros([nr_unlock,nr_unlock]).astype(int)
    for diagonal in range(nr_unlock):
        done_matrix[diagonal,diagonal] = 1
    while ((done_matrix == 0).any()):
        chosen_idx = np.array([]).astype(int)
        for unlock_idx in range(nr_unlock):
            if (len(chosen_idx) >= preferred_unlock_amount):
                break
            not_done_idx = np.where(done_matrix[unlock_idx,:] == 0)[0]
            if (len(not_done_idx) == 0):
                continue
            chosen_idx = np.append(chosen_idx, [unlock_idx])
            chosen_idx = np.unique(chosen_idx)
            if (len(chosen_idx) + len(not_done_idx) <= preferred_unlock_amount):
                chosen_idx = np.append(chosen_idx, not_done_idx)
                chosen_idx = np.unique(chosen_idx)
            elif (len(chosen_idx) < preferred_unlock_amount):
                random_idx = np.array(random.sample(not_done_idx.tolist(), 
                                                    preferred_unlock_amount - len(chosen_idx)))
                chosen_idx = np.append(chosen_idx, random_idx)
                chosen_idx = np.unique(chosen_idx)
                break
        for i in chosen_idx:
            for j in chosen_idx:
                done_matrix[i,j] = 1
        sub_batch = unlock[chosen_idx]
        locked_part = np.setdiff1d(unlock, sub_batch)
        if (len(lock) > 0):
            locked_part = np.append(locked_part, lock)
        dict = env.to_ilp(sub_batch, locked_part, t_interval)
        instance = env.run_ilp_instance(dict, timelimit=t_max)
        env.update_from_ilp_solution(instance, t_interval)

def RunGroup(type, nr_unlock, env, t_interval, is_time_based=True):
    """This runs all orders or machines in random order and can
    be grouped together if nr_unlock is not 1. it will also cut the 
    time at the end.

    Args:
        type (string): machine or order
        nr_unlock (int): how many orders or machines should be unlocked together
        env (environment): environment
        t_interval (list[int,int]): time interval that you want to run.
    """
    if (type == "machine"):
        nr_types = num_machines
    elif (type == "order"):
        nr_types = num_orders
    else:
        print("not a valid type")
        return()
    #shuffle order
    shuffled_type = list(range(nr_types))
    random.shuffle(shuffled_type)
    nr_runs = math.ceil(nr_types/nr_unlock)
    runs_list = []
    for i_run in range(nr_runs):
        run = np.array([]).astype(int)
        for j_type in range(nr_unlock):
            idx = i_run*nr_unlock + j_type
            if (idx == nr_types):
                break
            run = np.append(run, shuffled_type[idx])
        runs_list.append(run)
    # runs
    for iRun, run in enumerate(runs_list):
        run_start_time = time.time()
        if (is_time_based):
            run_t_interval(type, env, run, t_interval)
        else:
            unlock, lock = env.unlock(nr_unlock, t_interval, type, run)
            if(len(unlock) == 0):
                continue
            RunSemiBatch(env, unlock, lock, t_interval)
        print("     run: " + str(iRun + 1) + "/" + str(len(runs_list)) 
                  + "    " + time_to_string(run_start_time))
    #env.remove_excess_time()

def print_progress(loops_count, total_loops):
    loops_count += 1
    print("############ Main Loop: " + str(loops_count) 
        + "/" + str(total_loops) + " ############")
    return (loops_count)

def time_to_string(start_time):
    time_value = time.time() - start_time
    time_value = np.ceil(10*time_value)/10
    string = str(time_value) + "s"
    return (string)

def csv_save(is_data_new, unlock, lock, env):
    schedule = env.schedule
    np_save = np.zeros([4,len(schedule[1,:])])
    for i in range(len(schedule[1,:])):
        np_save[0,i] = schedule[0,i]
        np_save[1,i] = i
        np_save[2,i] = schedule[2,i]
    for i in unlock:
        np_save[3,i] = 1
    for i in lock:
        np_save[3,i] = -1
    df = pd.DataFrame(np_save)
    if (is_data_new):
        df.to_csv("/home/buske/Progress/ProductionPlanningWithILP/Data/new.csv")
    else:
        df.to_csv("/home/buske/Progress/ProductionPlanningWithILP/Data/old.csv")

    

######## MAIN #########
# for i_size in range(size_steps):
#     if (i_size != 0):
#         env.divide_timeline(1)
#     time_length = math.ceil(len(env.time_line)/(i_size + 1))
#     nr_time_axises = 2*(i_size + 1)
#     if (i_size ==  0):
#         nr_time_axises = 1
#     for j_axis in range(nr_time_axises):
#         #some while catch done loop
#         t_interval = [time_length*j_axis, 
#                       time_length*(j_axis + 1) + 1]
#         if ((time_length + 1)*j_axis > time_length):
#             break
#         for iter in range(iter_per_size):
#             print("Main Loop: " + str(i_size + 1) + "/" + str(total_iterations))
#             RunGroup("order", 1, env, t_interval)
#             RunGroup("machine", 1, env, t_interval)
main_start_time = time.time()
for iSize in range(size_steps):
    if (iSize != 0):
        env.divide_timeline(1)
    nr_time_axises = len(env.time_line)
    t_interval = [0, nr_time_axises]
    
    for jIter in range(int(iter_per_size[iSize])):
        loops_count = print_progress(loops_count, total_loops)
        RunGroup("order", 1, env, t_interval)
        loops_count = print_progress(loops_count, total_loops)
        RunGroup("order", int(1), env, t_interval)
print("Total Run Time: " + time_to_string(main_start_time))


############ Plot ############
final_obj_value = env.get_objective_value(weight_json)
print("initial objective value:")
print(init_obj_value[0])
print("final objective value:")
print(final_obj_value[0])
env.plot(True)