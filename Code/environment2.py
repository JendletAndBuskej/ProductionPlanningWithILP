############## IMPORT ################
from functools import partial
from math import ceil
from re import T
import numpy as np
import matplotlib.pyplot as plt
import os, json
from classes import *
import pyomo as pyo
import pandas as pd

############# ENVIRONMENT_CLASS_##############
class Environment:
    def __init__(self, data_json: str) -> None:
        with open(data_json, "r") as f:
            self.orders_json = json.load(f)
        for data in self.orders_json:
            self.orders_json[data]
        # class-variables that is zero should be set from json here and probably need another help function to do so
        self.machines = 0
        self.operations = 0
        self.orders = 0    # is it easy to make them ordered, from time low to high
        self.time_index_value = 0
        self.longest_exec = 0

        self.oper_exec_num_unit = {}
        for i in range(len(self.operations)):
            self.oper_exec_num_unit[None] = {"exec_time": {i:1}}
        self.schedule_matrix = self.initial_schedule(self.orders, len(self.operations), 
                                                len(self.machines))    #order id = index in matrix
        self.last_sent_indices = []  #list of machines, list of operations, time_interval 

    def unlock_order(self, amount, t_interval , order_id=[0]):
        """this will return the input dict that ILP takes as input.
        it will unlock the amount of of operations given

        Args:
            amount (_type_): _description_
            t_interval (_type_): _description_
            order_idx (list, optional): _description_. Defaults to [0].
        """
        #unlocked operations in interval, just for you Theo my friend, know these comments will be removed
        sorted_id = id_handler(amount, order_id, len(self.operations) - 1)
        part_schedule = self.schedule_matrix[:, sorted_id, t_interval[0]:t_interval[1]]
        #locked operations in interval
        locked_operations = self.schedule_matrix[:, :, t_interval[0]:t_interval[1]]
        locked_operations = np.delete(locked_operations, sorted_id, 1)
        self.last_sent_indices = [range(self.schedule_matrix.shape[0]), sorted_id, t_interval]
        return(to_ilp(part_schedule, t_interval, locked_operations))

    def unlock_machine(self, amount, t_interval, machine_id=[0]):
        sorted_id = id_handler(amount, machine_id, len(self.machines) - 1)
        part_schedule = self.schedule_matrix[sorted_id, :, t_interval[0]:t_interval[1]]
        return(to_ilp(part_schedule, t_interval))

    def to_ilp(self, part_schedule, t_interval, locked_operations=np.zeros[1,1,1]): #this preset is because machines don't need it
        """_summary_

        Args:
            part_schedule (_type_): _description_
            t_interval (_type_): _description_
            locked_operations (_type_): this is a 3dim numpy matrix as part schedule
        """
        pass

    def update_from_ilp_instance(self, ilp_output):
        """_summary_

        Args:
            ilp_output (_type_): _description_
        """
        def instance_2_numpy(instance_data: pyo.Var | pyo.Param | pyo.Set | pyo.RangeSet, 
                             shape_array: np.ndarray | list = [] ) -> any:
        pass
    pass

    def double_steps(self):
        """this doubles the amount of steps in the schedule and changes the execution time if
        operations have longer execution times.
        """
        for t in range(self.schedule_matrix.shape[2])[::-1]:
            zero_matrix = np.zeros(self.schedule_matrix.shape[:,:,1])
            np.insert(self.schedule_matrix, t, zero_matrix, axis=2)
        for i in range(self.operations):
            for j in range(1000):
                if ((j+1)*self.time_index_value < self.operations[i].get_execution_time()):
                    self.oper_exec_num_unit["None"]["exec_time"][i] = j
                    break


    def plot(t_interval):
        pass

    
    ############ HELP_FUNCTIONS ###############
    def initial_schedule(self, orders, num_operations, num_machines):
        """this will return a semi-bad schedule that dosn't break
        any constraints. It assumes that all operations is of the
        same length and that order.get_operations() is sorted so that 
        placing the operations in that order wont break precedence constraint.
        """
        def find_space(operation, min_time, schedule, schedule2d):
            if (min_time < schedule.shape[2]):
                for time in range(min_time, schedule.shape[2]):
                    for machine in operation.get_valid_machines():
                        if (schedule_2d[machine, time] == 0):
                            min_time = time + 1
                            schedule_2d[machine, time] = 1
                            schedule[machine, operation, time] = 1
                            return(min_time, schedule, schedule2d)
            min_time = schedule.shape[2]
            np.append(schedule2d, np.zeros(schedule2d.shape[0]), axis=1)
            np.append(schedule, np.zeros(schedule.shape[0:1]), axis=2)
            return(find_space(operation, min_time, schedule, schedule2d))

        schedule = np.zeros([num_machines, num_operations, 1])
        schedule_2d = np.zeros([num_machines, 1])
        for order in orders:
            min_time = 0
            for operation in order.get_operations():
                schedule, min_time, schedule_2d = find_space(operation, min_time,
                                                            schedule, schedule_2d)
        return(schedule)

    def time_cut(self, t_interval):
        
        def look_plane(plane, t_between, oper_list):
            for m in range(plane.shape[0]):
                for o in range(plane.shape[1]):
                    unit_check = self.oper_exec_num_unit[o] > t_between 
                    if (plane[m,o] == 1 and unit_check):
                        oper_list.append(o)
                        break
            return(oper_list)
                            
        schedules_left = []
        fake_cut = 0
        if (t_interval[0] - int(self.longest_exec/self.time_index_value) > 0):
            fake_cut = int(t_interval[0] - self.longest_exec)
        for t in range(fake_cut, t_interval[0]):
            plane = self.schedule_matrix[:,:,t]
            schedules_left = look_plane(plane, t_interval[0]-t,schedules_left)
        schedules_right = []
        fake_cut = int(t_interval[1] - self.longest_exec)
        for t in range(fake_cut, t_interval[1]):
            plane = self.schedule_matrix[:,:,t]
            schedules_right = look_plane(plane, t_interval[1]-t,schedules_right)
        return
                
        pass




################# HELP_FUNCTIONS #####################


def id_handler(amount, id_list, max_id):
    """this will just return ordered id_list. In the case of id_list = [0]
     a random ordered id_list will be returned

    Args:
        amount (int): how many ids should be returned.
        id_list (ndarray): the chosen ids.
        max_id (int): what is the maximum id value possible. 
    """
    if (len(id_list) == 1 and id_list[0] == 0):
        id = np.ceil(max_id*np.random.rand([amount]))
    else:
        id = id_list
    sorted_id = id.sort()
    return(sorted_id)