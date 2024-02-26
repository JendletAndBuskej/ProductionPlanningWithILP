############## IMPORT ################
import numpy as np
import matplotlib.pyplot as plt
import os, json
from classes import Operation, Order
from Data_Converter.batch_data import Batch_Data
import pyomo as pyo
import pandas as pd

############# ENVIRONMENT_CLASS_##############
class Environment:
    def __init__(self, data_json: dict) -> None:
        # with open(data_json, "r") as f:
            # self.orders_json = json.load(f)
        # for data in self.orders_json:
            # self.orders_json[data]
        self.orders_json = data_json
        # class-variables that is zero should be set from json here and probably need another help function to do so
        machine_data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/Data/Raw/lines_tab.txt"
        self.machines = self.initialize_machines(machine_data_path)
        self.operations = self.initialize_operations()
        self.orders = self.initialize_orders()    # is it easy to make them ordered, from time low to high

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
        return(self.to_ilp(part_schedule, t_interval))#, locked_operations))

    def unlock_machine(self, amount, t_interval, machine_id=[0]):
        sorted_id = id_handler(amount, machine_id, len(self.machines) - 1)
        part_schedule = self.schedule_matrix[sorted_id, :, t_interval[0]:t_interval[1]]
        return(self.to_ilp(part_schedule, t_interval))

    def initialize_machines(self, data_path_machine_types : str):
        txt_to_np = np.genfromtxt(data_path_machine_types, skip_header=1, usecols=1, dtype=int)
        machine_id = 0
        machines = {}
        for iM, num_machine_type in enumerate(txt_to_np):
            machine_type = "machine_type_"+str(iM)
            for machine in range(num_machine_type):
                machines[machine_id] = machine_type
                machine_id += 1
        return machines

    def initialize_operations(self):
        operations = []
        for iOp, operation in enumerate(self.orders_json):
            operation_data = self.orders_json[operation]
            exec_time = operation_data["startup_time"] + operation_data["operation_time"]
            num_operators = operation_data["num_operator"]
            parent = operation_data["parent"]
            valid_machines = operation_data["linetype"]
            oper = Operation(iOp, exec_time, valid_machines, num_operators, parent)
            operations += [oper]
        return operations
    
    def initialize_orders(self):
        orders = {}
        for iOp, operation in enumerate(self.orders_json):
            operation_data = self.orders_json[operation]
            order_tree = operation_data["tree"]
            if order_tree not in orders:
                orders[order_tree] = []     
            orders[order_tree] += [self.operations[iOp]]
        
        return orders

    
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
    
    def to_ilp(self, part_schedule, t_interval):#, locked_operations=np.zeros[1,1,1]): #this preset is because machines don't need it
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

    def plot(t_interval):
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


if (__name__ == "__main__"):
    batch_size = 2
    batch_data = Batch_Data(batch_size=batch_size)
    batched_data = batch_data.get_batch()
    print(len(batched_data))
    env = Environment(batched_data)