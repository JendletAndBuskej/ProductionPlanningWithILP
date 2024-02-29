############## IMPORT ################
import numpy as np
import matplotlib.pyplot as plt
import os, json, re, math
from classes import Operation, Order
from Data_Converter.batch_data import Batch_Data
import pyomo as pyo
import pandas as pd

############# ENVIRONMENT_CLASS_##############
class Environment:
    def __init__(self, data_json: dict) -> None:
        self.orders_json = data_json
        machine_data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/Data/Raw/lines_tab.txt"
        self.machines = self.initialize_machines(machine_data_path)
        self.operations = self.initialize_operations()
        self.orders = self.initialize_orders()    # is it easy to make them ordered, from time low to high
        self.time = []
        self.schedule = self.initial_schedule()
        self.max_oper = 1
        self.time_step_size = 50 # this is just before we fix this
        #self.schedule_matrix = self.initial_schedule(self.orders, len(self.operations), 
        #                                        len(self.machines))    #order id = index in matrix
        self.last_sent_indices = []  #list of machines, list of operations, time_interval 

        ## Test ##
        #self.schedule_matrix = np.empty([3,3], dtype=object)
        ## machine
        #self.schedule_matrix[0,0] = 0
        #self.schedule_matrix[0,1] = 0
        #self.schedule_matrix[0,2] = 1
        ## oper
        #self.schedule_matrix[1,0] = self.operations[0]
        #self.schedule_matrix[1,1] = self.operations[1]
        #self.schedule_matrix[1,2] = self.operations[2]
        ## stime
        #self.schedule_matrix[2,0] = 0
        #self.schedule_matrix[2,1] = 1
        #self.schedule_matrix[2,2] = 2
        #self.to_ilp(unlocked_operations_indices=[1], locked_operations_indices=[0,2], time_interval=[0,2])
    
    def unlock_order(self, nr_orders, t_interval, order_idx=[0]):
        """this will return the input dict that ILP takes as input.
        it will unlock the amount of of operations given

        Args:
            amount (_type_): _description_
            t_interval (_type_): _description_
            order_idx (list, optional): _description_. Defaults to [0].
        """
        #unlocked operations in interval, just for you Theo my friend, know these comments will be removed
        locked_oper = []
        unlocked_oper = []
        if (len(order_idx) == 1 and order_idx[0] == 0):
            order_idx = np.ceil(len(self.orders)*np.random.rand(nr_orders))
        for oper_idx in range(len(self.schedule[0])):
            oper3 = [self.schedule[0][oper_idx],
                     self.schedule[1][oper_idx],
                     self.schedule[2][oper_idx]]
            if (oper3[2] > t_interval[0] and oper3[2] < t_interval[1] - self.max_oper):
                i = 0
                for idx in order_idx:
                    i += 1
                    if (oper3[1].order == self.orders[idx]):
                        unlocked_oper += [oper3[1]]
                        break
                    if (i == len(order_idx)):
                        locked_oper += [oper3[1]]
            elif (oper3[2] > t_interval[1] -self.max_oper and oper3[2] < t_interval[1]):
                exe_time = ceil(oper3[1].execution_time/self.time_step_size)
                if (oper3[2] + exe_time <= t_interval[1]):
                    unlocked_oper += [oper3[1]]
        locked_oper += self.line_check(t_interval[0])
        locked_oper += self.line_check(t_interval[1])
        return(unlocked_oper, locked_oper)

    def line_check(self, time_of_line):
        oper_on_line = []
        range_set = [time_of_line - self.max_oper, time_of_line]
        if (time_of_line - self.max_oper < 0):
            range_set = [0, time_of_line]
        for oper_idx in range(len(self.schedule[0])):
            oper3 = [self.schedule[0][oper_idx],
                     self.schedule[1][oper_idx],
                     self.schedule[2][oper_idx]]
            check1 = oper3[2] > range_set[0] and oper3[2] > range_set[1]
            exe_time = ceil(oper3[1].execution_time/self.time_step_size)
            check2 = oper3[2] + exe_time > time_of_line
            if (check1 and check2):
                oper_on_line += [oper3[1]]
        return(oper_on_line)

    def unlock_machine(self, amount, t_interval, machine_id=[0]):
        sorted_id = id_handler(amount, machine_id, len(self.machines) - 1)
        part_schedule = self.schedule_matrix[sorted_id, :, t_interval[0]:t_interval[1]]
        return(self.to_ilp(part_schedule, t_interval))

    def initialize_machines(self, data_path_machine_types : str) -> dict:
        """Extracts machine data from the machine txt file 'lines_tab.txt'
        and returns a dict with a machine ID as key and machine type as value

        Args:
            data_path_machine_types (str): Path to the data file containing machine data
        """
        txt_to_np = np.genfromtxt(data_path_machine_types, skip_header=1, usecols=1, dtype=int)
        machine_id = 0
        machines = {}
        for iM, num_machine_type in enumerate(txt_to_np):
            machine_type = "machine_type_"+str(iM)
            for machine in range(num_machine_type):
                machines[machine_id] = machine_type
                machine_id += 1
        return(machines)

    def initialize_operations(self) -> list:
        """Instanciates and returns all operations as a list, 
        where the elements are of the class 'Operation'. 
        The ID of an operation is the same as it's index in list.

        Args:
            None
        """
        # finds all machine ids of a machine type
        def get_valid_machine_ids(machine_types: list[str]) -> list[int]:
            valid_machine_ids = []
            for m_type in machine_types:
                m_type_as_int = self.find_int_in_string(m_type)
                for machine_id, machine_type in self.machines.items():
                    machine_type_as_int = self.find_int_in_string(machine_type)
                    if machine_type_as_int == m_type_as_int:
                        valid_machine_ids += [machine_id]
            return(valid_machine_ids)

        operations = []
        for iOp, operation in enumerate(self.orders_json):
            operation_data = self.orders_json[operation]
            exec_time = operation_data["startup_time"] + operation_data["operation_time"]
            num_operators = operation_data["num_operator"]
            parent_name = operation_data["parent"]
            valid_machines_types = operation_data["linetype"]
            valid_machine_ids = get_valid_machine_ids(valid_machines_types)
            oper = Operation(id=iOp, 
                             name=operation, 
                             execution_time=exec_time,
                             valid_machine_ids=valid_machine_ids, 
                             num_operators=num_operators, 
                             parent_name=parent_name)
            operations += [oper]
        # once every operation have been instanciated, the parents are set
        for operation in operations:
            operation.set_parent(operations)
        return(operations)
    
    def initialize_orders(self) -> list["Order"]:
        """Instanciates and returns a list of all Orders. 
        Each element of the list is of the class Order which in itself contains all the
        operations needed to be completed to finish the Order. The class Order also contains
        the Due Date and ID + Name 

        Args:
            None
        """
        orders = {}
        for iOp, operation in enumerate(self.orders_json):
            operation_data = self.orders_json[operation]
            order_tree = operation_data["tree"]
            if order_tree not in orders:
                orders[order_tree] = []     
            orders[order_tree] += [self.operations[iOp]]
            #self.operations[iOp].print_info()
        orders_list = []
        for iOrd, order_tree in enumerate(orders):
            ord = Order(id=iOrd,
                        name=order_tree,
                        operations=orders[order_tree])
            for operation in ord.operations:
                operation.set_order(ord)
                #operation.order.print_info()
            orders_list += [ord]
            ord.print_info()
        return(orders_list)

    
    ############# HELP_FUNCTIONS ###############
    def initial_schedule(self):
        """this will return a semi-bad schedule that dosn't break
        any constraints. It assumes that all operations is of the
        same length and that order.get_operations() is sorted so that 
        placing the operations in that order wont break precedence constraint.
        """
        def find_space(operation, min_time, schedule, schedule_2d):
            if (min_time < schedule_2d.shape[1]):
                for time in range(min_time, schedule_2d.shape[1]):
                    for machine in operation.get_valid_machine_ids():
                        if (schedule_2d[machine, time] == 0):
                            min_time = time + 1
                            schedule_2d[machine, time] = 1
                            schedule[0].append(machine)
                            schedule[1].append(operation) 
                            schedule[2].append(time) 
                            return(min_time, schedule, schedule_2d)
            min_time = schedule_2d.shape[1]
            schedule_2d = np.append(schedule_2d, np.zeros([schedule_2d.shape[0],1]), axis=1)
            return(find_space(operation, min_time, schedule, schedule_2d))

        schedule = [[],[],[]]
        schedule_2d = np.zeros([len(self.machines), 1])
        for order in self.orders:
            min_time = 0
            for operation in order.get_operations():
                min_time, schedule, schedule_2d = find_space(operation, min_time,
                                                            schedule, schedule_2d)
        return(schedule)
    
    def to_ilp(
            self, 
            unlocked_operations_indices: list[int], 
            locked_operations_indices: list[int],
            time_interval: list[int]
        ) -> dict:
        """Converts the data of the schedule to send to the ILP. Includes a list of operation index
        to optimize and a list of operation index to keep locked. Returns a dict

        Args:
            unlocked_operations_indices (list[int]): List of operation index to unlock from schedule_matrix.
            locked_operations_indices (list[int]): List of operation index to keep locked from schedule_matrix.
            time_interval (list[int]): Time interval to optimize over. (Currently seen as time indices)
        """
        def init_dict(dim_1: int, dim_2: int = 0) -> dict:
            init_dict = {}
            if dim_2 == 0:
                for iDim in range(1,dim_1+1):
                    init_dict[iDim] = 0
                return(init_dict)
            for iDim in range(1,dim_1+1):
                for jDim in range(1,dim_2+1):
                    init_dict[(iDim,jDim)] = 0
            return(init_dict)
        
        def get_valid_machines_and_exec_time() -> tuple[dict, dict]:
            valid_machines = init_dict(len(unlocked_operations_indices), len(self.machines))
            exec_time = init_dict(len(unlocked_operations_indices))
            for iUO, unlocked_operation_index in enumerate(unlocked_operations_indices):
                unlocked_operation = self.schedule_matrix[1,unlocked_operation_index]
                exec_time[iUO+1] = unlocked_operation.execution_time
                for machine_id in unlocked_operation.valid_machine_ids:
                    valid_machines[(iUO+1,machine_id)] = 1
            return(valid_machines, exec_time)

        def get_locked_operations_info() -> tuple[dict, dict, dict]:
            locked_operation_machine = init_dict(len(locked_operations_indices))
            locked_operation_start_time = init_dict(len(locked_operations_indices))
            locked_operation_exec_time = init_dict(len(locked_operations_indices))
            for iLO, locked_operation_index in enumerate(locked_operations_indices):
                locked_operation = self.schedule_matrix[:,locked_operation_index]
                locked_operation_machine[iLO+1] = locked_operation[0]
                locked_operation_exec_time[iLO+1] = locked_operation[1].execution_time
                locked_operation_start_time[iLO+1] = locked_operation[2]
            return(locked_operation_machine, locked_operation_exec_time, locked_operation_start_time)

        num_machines = { None: len(self.machines)}
        num_operations = { None: len(unlocked_operations_indices)}
        num_locked_operations = { None: len(locked_operations_indices)}
        num_time_indices = { None: time_interval[1] - time_interval[0]}
        valid_machines, exec_time = get_valid_machines_and_exec_time()
        locked_oper_machine, locked_oper_exec_time, locked_oper_start_time = get_locked_operations_info()
        ilp_input = {
            None: {
                "num_machines" : num_machines,
                "num_operations" : num_operations,
                "num_locked_operations" : num_locked_operations,
                "num_time_indices" : num_time_indices,
                "valid_machines" : valid_machines,
                "exec_time" : exec_time,
                "locked_oper_machine" : locked_oper_machine,
                "locked_oper_exec_time" : locked_oper_exec_time,
                "locked_oper_start_time" : locked_oper_start_time
            }
        }
        return(ilp_input)

    def update_from_ilp_instance(self, ilp_output):
        """_summary_

        Args:
            ilp_output (_type_): _description_
        """
        def instance_2_numpy(instance_data: pyo.Var | pyo.Param | pyo.Set | pyo.RangeSet, 
                             shape_array: np.ndarray | list = [] ) -> any:
            pass
        pass

    def plot(self, t_interval):
        pass
    
    def find_int_in_string(self, string):
        return [int(match) for match in re.findall(r'\d+', string)]


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
    env = Environment(batched_data)
    test1, test2 = env.unlock_order(1, [0,10])