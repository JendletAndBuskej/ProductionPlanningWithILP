######### IMPORTS ##############
import numpy as np
import matplotlib.pyplot as plt
import os, json, math, random
from classes import Operation, Order
from environment import Environment
from Data_Converter.batch_data import BatchData
from ilp import create_ilp, run_ilp
import pyomo as pyo
from pyomo.opt import SolverFactory
import pandas as pd

########## PARAMETERS #############
iter_per_size = 2
size_steps = 2
num_orders = 10
num_machines = 35
max_permutations = math.comb(num_machines*28,10)
num_machine_unlock_amount = 1
num_order_unlock_amount = 1

schedule_length = 0

####### INITIALIZE ########
data_bank = BatchData(num_orders)
input_json = data_bank.get_batch()
env = Environment(input_json)


###### HELP_FUNCTIONS #######
def RunSemiBatch(env, unlock, lock, t_interval):
    """this calculates if the complexity of the planned ILP run is to big and if so it will run multiple smaller
    ILP solves to get a resonable run time.

    Args:
        env (_type_): _description_
        unlock (_type_): _description_
        lock (_type_): _description_
        t_interval (_type_): _description_
    """
    def CalculateComplexity(nr_time_steps, nr_unlock):
        """Calculates if the complexity of the planned unlocked ILP is smaller or greater than a good reference.
        This returns true if the reference has a lower complexity and false otherwise. nr_time_steps_good_test and 
        nr_oper_good_test is used to calculate the good reference value.

        Args:
            nr_time_steps (int): this is the planed amount of unlocked time steps.
            nr_unlock (int): this is the planed amount of unlocked operations.
        """
        nr_time_steps_good_test = 28
        nr_oper_good_test = 9
        max_permutations = nr_oper_good_test*math.comb(num_machines*nr_time_steps_good_test, nr_oper_good_test)

        possibilities = nr_oper_good_test*math.comb(num_machines*nr_time_steps,nr_unlock)
        if (possibilities > max_permutations):
            return(True)
        return(False)

    unlock = np.array(unlock)
    nr_time_steps = t_interval[1] - t_interval[0]
    nr_unlock = len(unlock)
    too_big = CalculateComplexity(nr_time_steps, nr_unlock)
    if not (too_big):
        dict = env.to_ilp(unlock, lock, t_interval)
        instance = env.run_ilp_instance(dict)
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
        locked_part = np.append(locked_part, lock)
        dict = env.to_ilp(sub_batch, locked_part, t_interval)
        instance = env.run_ilp_instance(dict)
        env.update_from_ilp_solution(instance, t_interval)

def Compression():
    """This will compress the schedule to make the upcoming ILP runs faster
    by removing some of the empty space at the end of the schedule.
    """
    pass

def RunGroup(type, nr_unlock, env, t_interval):
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
    for i_run in runs_list:
        print("run: " + str(i_run))
        unlock, lock = env.unlock(nr_unlock, t_interval, type, i_run)
        if(len(unlock) == 0):
            continue
        RunSemiBatch(env, unlock, lock, t_interval)
    env.remove_excess_time()


######## MAIN #########
for i_size in range(size_steps):
    if (i_size != 0):
        env.divide_timeline(1)
    time_length = math.ceil(len(env.time_line)/(i_size + 1))
    nr_time_axises = 2*(i_size + 1)
    if (i_size ==  0):
        nr_time_axises = 1
    for j_axis in range(nr_time_axises):
        #some while catch done loop
        t_interval = [time_length*j_axis, 
                      time_length*(j_axis + 1)]
        print(t_interval)
        if ((time_length + 1)*j_axis > time_length):
            break
        RunGroup("order", 1, env, t_interval)
        RunGroup("machine", 1, env, t_interval)

env.plot(True)