############## IMPORT ################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import os, json, math, random
from datetime import datetime
from classes import Operation, Order
from Data_Converter.batch_data import BatchData
from ilp import instanciate_ilp_model, run_ilp
import pyomo as pyo
from pyomo.opt import SolverFactory
import pandas as pd
############# ENVIRONMENT_CLASS_##############
class Environment:
    def __init__(self, input_json: dict, weight_json: dict = {}) -> None:
        self.input_json = input_json
        self.machines = self.initialize_machines()
        self.initialize_operations_data()
        self.orders = self.initialize_orders()
        self.schedule, self.time_line = self.initial_schedule_and_time_line()
        self.time_step_size = self.time_line[1] - self.time_line[0]
        self.longest_oper = self.time_line[1] - self.time_line[0]
        self.precedence = self.initialize_precedence()
        self.model = self.create_ilp_model(weight_json)
        self.weight_json = weight_json
    
    ### INITIALIZATION ###
    def initialize_machines(self):
        """Instanciates the machines as a dict, keys as the IDS and the values as the machine type.
        Returns:
            dict: Machine IDS to type
        """
        def get_valid_machine_types():
            valid_machine_types = []
            for iOp, operation in enumerate(self.input_json):
                operation_valid_machine_types = self.input_json[operation]["linetype"]
                valid_machine_types += operation_valid_machine_types
                valid_machine_types = list(set(valid_machine_types))
            return (valid_machine_types)
        
        valid_machine_types = get_valid_machine_types()
        master_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = master_path + "/Data/"
        with open(data_path+"/Parsed_Json/machines.json", "r") as f:
            machines_json = json.load(f)  
        all_machine_types = list(machines_json.values())
        valid_machines = [x for x in all_machine_types if x in valid_machine_types]
        machines = {}
        for iMach, machine_type in enumerate(valid_machines):
            machines[iMach] = machine_type
        # machines = remove_redundent_machines(all_machine_types, valid_machine_types)
        # machines = {int(key): value for key, value in machines.items()}
        return machines
            
    def initialize_operations_data(self) -> None:
        """Instanciates and returns all operations as a list, 
        where the elements are of the class 'Operation'. 
        The ID of an operation is the same as it's index in list.
        """
        def get_valid_machine_ids(machine_types: list[str]) -> list[int]:
            valid_machine_ids = []
            for m_type in machine_types:
                for machine_id, machine_type in self.machines.items():
                    if machine_type == m_type:
                        valid_machine_ids += [machine_id]
            return (valid_machine_ids)
            
        operations = []
        for iOp, operation in enumerate(self.input_json):
            operation_data = self.input_json[operation]
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
        for operation in operations:
            operation.set_parent(operations)
            parent_oper = operation.parent
            if (parent_oper):
                parent_oper.set_children([operation]) 
        self.operations_data = operations
    
    def initialize_orders(self) -> list["Order"]:
        """Instanciates and returns a list of all Orders, where the operations 
        sorted within an order.
        Each element of the list is of the class Order which in itself contains all the
        operations needed to be completed to finish the Order. The class Order also contains
        the Due Date and ID + Name.
        
        Returns:
            list["Order"]: list of Orders
        """
        def sort_order(order: "Order") -> "Order":
            sort_help_dict = {0 : []}
            operations = order.get_operations()
            sorted_order = Order(id=order.id, name=order.name, operations=[], due_date=order.due_date)
            # finds all leaf operations
            for operation_ref in operations:
                leaf_operation = True
                for operation_check in operations:
                    if operation_check.parent == operation_ref:
                        leaf_operation = False
                        break
                if leaf_operation:
                    sort_help_dict[0] += [operation_ref]
            # adds all layers by traversing up from the leaf nodes
            for leaf_operation in sort_help_dict[0]:
                traverse_order(operation=leaf_operation, layer_level=0, sort_help_dict=sort_help_dict)
            # removes the duplicates between layers and sorts the order
            layers_descending = list(sort_help_dict.keys())[::-1]
            for iL,layer in enumerate(layers_descending):
                layer_operations = sort_help_dict[layer]
                sorted_order.operations += layer_operations
                for layer_operation in layer_operations:
                    for descended_layer in layers_descending[iL+1:]:
                        if layer_operation in sort_help_dict[descended_layer]:
                            sort_help_dict[descended_layer].remove(layer_operation)
            sorted_order.operations = sorted_order.operations[::-1]
            return (sorted_order)
        
        def traverse_order(operation: "Operation", layer_level: int, sort_help_dict: dict) -> None:
            parent = operation.parent
            if not parent:
                return ()
            layer_level += 1
            if not layer_level in sort_help_dict:
                sort_help_dict[layer_level] = []
            if not parent in sort_help_dict[layer_level]:
                sort_help_dict[layer_level] += [parent]
            traverse_order(parent, layer_level, sort_help_dict)
            
        orders = {}
        for iOp, operation in enumerate(self.input_json):
            operation_data = self.input_json[operation]
            order_tree = operation_data["tree"]
            if order_tree not in orders:
                orders[order_tree] = {}
                orders[order_tree]["operations"] = []
            if "due_date" in operation_data:
                orders[order_tree]["due_date"] = operation_data["due_date"]
            orders[order_tree]["operations"] += [self.operations_data[iOp]]
        orders_list = []
        for iOrd, order_tree in enumerate(orders):
            ord = Order(id=iOrd,
                        name=order_tree,
                        operations=orders[order_tree]["operations"],
                        due_date=orders[order_tree]["due_date"])
            for operation in ord.operations:
                operation.set_order(ord)
            ord = sort_order(ord)
            orders_list += [ord]
        for order in orders_list:
            for operation in order.get_operations():
                operation.order = order
        return (orders_list)
    
    def initial_schedule_and_time_line(self) -> tuple[np.ndarray, np.ndarray]:
        """This will return a schedule that doesn't break
        any constraints. It assumes that all operations is of the
        same length.
        
        Returns:
            np.ndarray: 2D schedule, where the rows corresponds to; 
            machine, operation and start time.
            np.ndarray: Time line representing real time.
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
                            return (min_time, schedule, schedule_2d)
            min_time = schedule_2d.shape[1]
            schedule_2d = np.append(schedule_2d, np.zeros([schedule_2d.shape[0],1]), axis=1)
            return (find_space(operation, min_time, schedule, schedule_2d))
    
        max_exec_time = 0
        schedule = [[],[],[]]
        schedule_2d = np.zeros([len(self.machines), 1])
        for order in self.orders:
            min_time = 0
            for operation in order.get_operations():
                if (operation.execution_time > max_exec_time):
                    max_exec_time = operation.execution_time
                min_time, schedule, schedule_2d = find_space(operation, min_time,
                                                             schedule, schedule_2d)
        time_step_size = max_exec_time
        time_line = time_step_size*np.arange(schedule_2d.shape[1]+1)
        schedule = np.array(schedule)
        return (schedule, time_line)
    
    def initialize_precedence(self) -> np.ndarray:
        """Instanciates and returns a numpy matrix of the precedence. Dim: [num_oper x num_oper].
        The value 1 at index (i,j) indicates that Operation j must be executed AFTER the completion
        of operation i. Otherwise the element is zero.  

        Returns:
            np.ndarray: 
        """
        num_operations = len(self.operations_data)
        precedence = np.zeros([num_operations, num_operations], dtype=int)
        operation_list = self.schedule[1,:]
        for iOp,operation in enumerate(operation_list):
            precedence_operation = operation.parent
            precedence_index = np.where(operation_list == precedence_operation)
            precedence[iOp,precedence_index] = 1
        return (precedence)
    
    def get_balance_weight(self):
        """this will calculate the weights to be used in the ilp, to make it consequent when dividing time line

        Returns:
            _type_: _description_
        """
        scaled_time_step = self.time_step_size/100000
        balance_json = {
        "make_span": 2*scaled_time_step/(len(self.schedule[1,:])),
        "make_span_real": scaled_time_step,
        "lead_time": scaled_time_step/(len(self.orders)),
        "lead_time_fake": 2*scaled_time_step/(len(self.schedule[1,:])),
        "operators": 10/self.weight_json["max_amount_operators"],
        "fake_operators": 1/len(self.time_line),
        "earliness": 1,
        "tardiness": 1,
        # "earliness": scaled_time_step/(len(self.orders)),
        # "tardiness": scaled_time_step/(len(self.orders)),
        }
        return (balance_json)
    
    ### SCHEDULE_UNLOCKING ###
    def unlock_order(
            self, 
            num_orders: int,
            t_interval: list[int],
            order_idx: list[int] = [0]
        ) -> tuple[list[int], list[int]]:
        """This will unlock every operation within an order within a given timespan and
        return two list; unlocked and locked operations indicies.
        
        Args:
            amount (int): number of orders to unlock
            t_interval (list[int]): time interval to unlock
            order_idx (list[int]): specifies order indicies to unlock. Defaults random order.
        
        Returns:
            tuple[list["Operation], list["Operation]]
        """
        statement = "order"
        unlocked_oper, locked_oper = self.unlock(num_orders, t_interval, 
                                                 statement, order_idx)
        return (unlocked_oper, locked_oper)

    def unlock_machine(
            self, 
            num_machines: int,
            t_interval: list[int],
            machine_idx: list[int] =[0]
        ) -> tuple[list["Operation"], list["Operation"]]:
        """This will unlock every operation within a machine within a given timespan and
        return two list; unlocked and locked operations indicies.
        
        Args:
            amount (int): number of machines to unlock
            t_interval (list[int]): time interval to unlock
            order_idx (list[int]): specifies a machine to unlock. Defaults random machine.
            
        Returns:
            tuple[list["Operation], list["Operation]]
        """
        statement = "machine"
        unlocked_oper, locked_oper = self.unlock(num_machines, t_interval, 
                                                 statement, machine_idx)
        return (unlocked_oper, locked_oper)
    
    def unlock(self, 
            num_objects: int,
            t_interval: list[int],
            check_statement: str,
            object_idx: list[int] =[]
        ) -> tuple[list["Operation"], list["Operation"]]:
        """Help function to unlock operations
        """
        valid_statements = ["order", "machine"]
        if not check_statement in valid_statements:
            print("error not a valid unlock statement")
            return ()
        if (check_statement == "order"):
            object_amount = len(self.orders)
        if (check_statement == "machine"):
            object_amount = len(self.machines)
        locked_oper = []
        unlocked_oper = []
        if (len(object_idx) == 0):
            object_idx = random.sample(range(object_amount), num_objects)
        for oper_idx in range(self.schedule.shape[1]):
            oper_info = self.schedule[:,oper_idx]
            exec_time = math.ceil(oper_info[1].execution_time/self.time_step_size)
            if not (oper_info[2] + exec_time > t_interval[0] and oper_info[2] < t_interval[1]):
                continue
            check_within_upper_bound = oper_info[2] + exec_time <= t_interval[1]
            check_within_lower_bound = oper_info[2] >= t_interval[0]
            for i,idx in enumerate(object_idx):
                if (check_statement == "order"):
                    check_same = (oper_info[1].order == self.orders[idx])
                if (check_statement == "machine"):
                    check_same = (oper_info[0] == idx)
                if (check_same and check_within_upper_bound and check_within_lower_bound):
                    unlocked_oper += [oper_idx]
                    break
                if (i+1 == len(object_idx)):
                    locked_oper += [oper_idx]
        return (unlocked_oper, locked_oper)
    
    ### TIMELINE_MANAGEMENT ###
    def divide_timeline(self, num_divisions = 1) -> None:
        """Updates the time line, increasing it's number of intervals. 

        Args:
            num_divisions (int, optional): Times to divide the time line in half. Defaults to 1.
        """
        self.time_step_size = self.time_step_size/(2**num_divisions)
        self.schedule[2,:] = 2*self.schedule[2,:]
        self.time_line = self.time_step_size * np.arange(2*self.time_line.shape[0]-1)
    
    def remove_excess_time(self) -> None:
        """Removes excess of the time line
        """
        def find_makespan():
            max_time = 0
            for iOper in range(self.schedule.shape[1]):
                oper_start_time = self.schedule[2,iOper]*self.time_step_size 
                oper_exec_time = self.schedule[1,iOper].execution_time
                oper_completed_time = oper_start_time + oper_exec_time
                if (oper_completed_time > max_time):
                    max_time = oper_completed_time
            return (max_time)

        makespan = find_makespan()
        new_timeline_max = makespan + self.longest_oper 
        timeline_indices_to_remove = np.where(self.time_line > new_timeline_max)
        self.time_line = np.delete(self.time_line, timeline_indices_to_remove)
        
    ### ILP_HANDLING ###
    def to_ilp(
            self, 
            unlocked_opers_indices: list[int], 
            locked_opers_indices: list[int],
            time_interval: list[int]
        ) -> dict:
        """Converts the data of the schedule to send to the ILP. Includes a list of operation index
        to optimize and a list of operation index to keep locked. Returns a dict

        Args:
            unlocked_operations_indices (list[int]): List of operation index to unlock from schedule.
            locked_operations_indices (list[int]): List of operation index to keep locked from schedule.
            time_interval (list[int]): Time interval to optimize over. (Currently seen as time indices)
        """
        if (len(unlocked_opers_indices) == 0):
            print("unlock list is empty, passing run")
            return([])
        def init_dict(dim_1: int, dim_2: int = 0, dim_3: int = 0) -> dict:
            init_dict = {}
            if dim_2 == 0 and dim_3 == 0:
                for iDim in range(1,dim_1+1):
                    init_dict[iDim] = 0
                return (init_dict)
            if dim_2 != 0 and dim_3 == 0:
                for iDim in range(1,dim_1+1):
                    for jDim in range(1,dim_2+1):
                        init_dict[(iDim,jDim)] = 0
                return (init_dict)
            for iDim in range(1,dim_1+1):
                for jDim in range(1,dim_2+1):
                    for kDim in range(1,dim_3+1):
                        init_dict[(iDim,jDim,kDim)] = 0
            return (init_dict)
        
        def get_valid_machines_and_exec_time() -> tuple[dict, dict]:
            valid_machines = init_dict(len(unlocked_opers_indices), len(self.machines))
            exec_time = init_dict(len(unlocked_opers_indices))
            for iUO, unlocked_operation_index in enumerate(unlocked_opers_indices):
                unlocked_operation = self.schedule[1,unlocked_operation_index]
                exec_time[iUO+1] = math.ceil(unlocked_operation.execution_time/self.time_step_size)
                for machine_id in unlocked_operation.valid_machine_ids:
                    valid_machines[(iUO+1,machine_id+1)] = 1
            return (valid_machines, exec_time)

        def get_locked_oper_info() -> tuple[dict, dict]:
            time_interval_length = time_interval[1] - time_interval[0]
            num_machines = len(self.machines)
            num_oper = len(locked_opers_indices)
            opers = init_dict(num_machines, num_oper, time_interval_length)
            exec_times = init_dict(num_oper)
            for iOper, oper_index in enumerate(locked_opers_indices):
                time_reduce  = 0
                oper = self.schedule[:,oper_index]
                oper_machine = oper[0] + 1
                oper_start_time = oper[2] + 1 - time_interval[0]
                if (oper_start_time < 1):
                    time_reduce = 1 - oper_start_time
                    oper_start_time = 1
                opers[(oper_machine, iOper+1, oper_start_time)] = 1
                exec_time = math.ceil(oper[1].execution_time/self.time_step_size) - time_reduce
                exec_times[iOper+1] = exec_time
            return (opers, exec_times)
        
        def map_unlocked_operations():
            self.mapping_unlocked_opers.clear()
            self.mapping_unlocked_opers = unlocked_opers_indices
        
        def get_precedence() -> dict:
            def sub_precedence(row, col):
                if (len(col) == 0 or len(row) == 0):
                    return ({})
                prece = init_dict(len(row),len(col))
                for i,iIdx in enumerate(row):
                    for j,jIdx in enumerate(col):
                        prece[(i+1,j+1)] = self.precedence[int(iIdx),int(jIdx)]
                return (prece)
            precedence = sub_precedence(unlocked_opers_indices, unlocked_opers_indices)
            prece_locked_before = sub_precedence(locked_opers_indices, unlocked_opers_indices)
            prece_locked_after = sub_precedence(unlocked_opers_indices, locked_opers_indices)
            return (precedence, prece_locked_before, prece_locked_after)
        
        def get_unlocked_amount_operators() -> dict:
            amount_operators = init_dict(len(unlocked_opers_indices))
            num_operators_np = np.array([oper.num_operators for oper in self.schedule[1,unlocked_opers_indices]])
            for iOper, num_oper in enumerate(num_operators_np):
                amount_operators[iOper+1] = num_oper
            return (amount_operators)
        
        def get_locked_amount_operators() -> dict:
            amount_operators = init_dict(len(locked_opers_indices))
            num_operators_np = np.array([oper.num_operators for oper in self.schedule[1,locked_opers_indices]])
            for iOper, num_oper in enumerate(num_operators_np):
                amount_operators[iOper+1] = num_oper
            return (amount_operators)
        
        def get_previous_schedule() -> dict:
            num_machines = len(self.machines)
            num_opers = len(unlocked_opers_indices)
            time_interval_len = time_interval[1] - time_interval[0] 
            previous_schedule = init_dict(num_machines,num_opers,time_interval_len)
            for iOper, oper in enumerate(unlocked_opers_indices):
                oper_machine = self.schedule[0,oper] + 1
                oper_start_time = self.schedule[2,oper] - time_interval[0] + 1
                previous_schedule[(oper_machine,iOper+1,oper_start_time)] = 1
            return (previous_schedule)
        
        def get_orders_within_interval() -> list["Order"]:
            all_indicies = np.append(unlocked_opers_indices, 
                                     locked_opers_indices).astype(int)
            orders_within_interval = []
            for iOper, oper in enumerate(all_indicies):
                order = self.schedule[1,oper].order
                if order in orders_within_interval:
                    continue
                orders_within_interval += [order]
            sorted_orders = []
            for iOrder, order in enumerate(self.orders):
                is_order_in = np.isin(np.array(orders_within_interval), order)
                if (is_order_in.any()):
                    sorted_orders += [order]
            return (sorted_orders) 
        
        def get_unlocked_orders_within_interval() -> list["Order"]:
            all_indicies = np.array(unlocked_opers_indices).astype(int)
            orders_within_interval = []
            for iOper, oper in enumerate(all_indicies):
                order = self.schedule[1,oper].order
                if order in orders_within_interval:
                    continue
                orders_within_interval += [order]
            sorted_orders = []
            for iOrder, order in enumerate(self.orders):
                is_order_in = np.isin(np.array(orders_within_interval), order)
                if (is_order_in.any()):
                    sorted_orders += [order]
            return (sorted_orders)
        
        def get_final_order_in(unlock_order_within: list["Order"]) -> dict:
            final_order_in = init_dict(len(unlock_order_within))
            for iOrd, order in enumerate(unlock_order_within):
                opers_np = np.array(order.operations)
                oper_indices = np.where(np.isin(self.schedule[1,:], opers_np))
                final_order_start_time = np.max(self.schedule[2,oper_indices])
                if (final_order_start_time < time_interval[1]):
                    final_order_in[iOrd+1] = 1
            return (final_order_in)
        
        def get_init_order_in(order_within: list["Order"]) -> dict:
            init_order_in = init_dict(len(order_within))
            for iOrd, order in enumerate(order_within):
                opers_np = np.array(order.operations)
                oper_indices = np.where(np.isin(self.schedule[1,:], opers_np))
                exec_times = np.array([math.ceil(oper.execution_time/self.time_step_size) 
                                       for oper in self.schedule[1,oper_indices[0]]])
                finished_times = self.schedule[2,oper_indices[0]] + exec_times
                init_order_finished_time = np.min(finished_times)
                if (init_order_finished_time > time_interval[0]):
                    init_order_in[iOrd+1] = 1
            return (init_order_in)
        
        def get_order_unlocked_oper(order_within: list["Order"]) -> dict:
            is_oper_in_order = init_dict(len(unlocked_opers_indices), len(order_within))
            for iOper, oper in enumerate(unlocked_opers_indices):
                oper_order = self.schedule[1,oper].order
                iOrd = np.where(np.isin(order_within, oper_order))[0]
                is_oper_in_order[(iOper+1,iOrd[0]+1)] = 1
            return (is_oper_in_order)
        
        def get_order_locked_oper(order_within: list["Order"]) -> dict:
            is_locked_in_order = init_dict(len(locked_opers_indices), len(order_within))
            for iOper, oper in enumerate(locked_opers_indices):
                oper_order = self.schedule[1,oper].order
                iOrd = np.where(np.isin(order_within, oper_order))[0]
                is_locked_in_order[(iOper+1,iOrd[0]+1)] = 1
            return (is_locked_in_order)
        
        def get_order_due_dates(order_within: list["Order"]):
            order_due_dates = init_dict(len(order_within))
            for iOrd, order in enumerate(order_within):
                order_due_dates[iOrd+1] = 1 + math.floor(order.due_date/self.time_step_size)
            return (order_due_dates)
        
        def get_orders_finished_time(order_within: list["Order"]) -> dict:
            orders_finished_time = init_dict(len(order_within))
            for iOrd, order in enumerate(order_within):
                opers = np.array(order.operations)
                oper_indices = np.where(np.isin(self.schedule[1,:], opers))
                exec_times = np.array([math.ceil(oper.execution_time/self.time_step_size)
                                       for oper in self.schedule[1,oper_indices[0]]]) 
                order_finished = np.max(self.schedule[2,oper_indices] + exec_times)
                orders_finished_time[iOrd+1] = order_finished
            return (orders_finished_time)

        self.mapping_unlocked_opers = [] 
        num_machines = { None: len(self.machines)}
        num_opers = { None: len(unlocked_opers_indices)}
        orders_within_interval = get_orders_within_interval()
        num_orders = { None: len(orders_within_interval)}
        map_unlocked_operations()
        previous_schedule = get_previous_schedule()
        num_locked_opers = { None: len(locked_opers_indices)}
        num_time_indices = { None: time_interval[1] - time_interval[0]}
        time_index_to_real = { None: int(self.time_step_size)}
        amount_operators = get_unlocked_amount_operators()
        valid_machines, exec_time = get_valid_machines_and_exec_time()
        precedence, prece_locked_before, prece_locked_after = get_precedence()
        locked_schedule, locked_oper_exec_time = get_locked_oper_info()
        locked_amount_operators = get_locked_amount_operators()
        is_final_order_in = get_final_order_in(get_unlocked_orders_within_interval())
        is_init_order_in = get_init_order_in(orders_within_interval)
        is_oper_in_order = get_order_unlocked_oper(orders_within_interval)
        # print("hello world")
        # oper = self.schedule[1,unlocked_opers_indices[0]]
        # order = oper.order.name
        # print(order)
        # penis = [order.name for order in self.orders]
        # print(penis)
        # print(is_oper_in_order)
        is_locked_in_order = get_order_locked_oper(orders_within_interval)
        order_due_dates = get_order_due_dates(orders_within_interval)
        orders_finished_time = get_orders_finished_time(orders_within_interval)
        balance_json = self.get_balance_weight()
        # print(exec_time)
        # print(locked_oper_exec_time)
        ilp_input = {
            None: {
                "num_machines" : num_machines,
                "num_opers" : num_opers,
                "num_orders" : num_orders,
                "num_locked_opers" : num_locked_opers,
                "num_time_indices" : num_time_indices,
                "valid_machines" : valid_machines,
                "previous_schedule" : previous_schedule,
                "precedence" : precedence, 
                "locked_prece_before": prece_locked_before,
                "locked_prece_after": prece_locked_after,
                "exec_time" : exec_time,
                "locked_exec_time" : locked_oper_exec_time,
                "locked_schedule" : locked_schedule,
                "amount_operators" : amount_operators,
                "locked_amount_operators" : locked_amount_operators,
                "is_final_order_in" : is_final_order_in,
                "is_init_order_in" : is_init_order_in,
                "is_oper_in_order" : is_oper_in_order,
                "is_locked_in_order" : is_locked_in_order,
                "order_due_dates" : order_due_dates,
                "balance_make_span" : { None: balance_json["make_span"] },
                "balance_make_real" : { None: balance_json["make_span_real"] },
                "balance_lead_time" : { None: balance_json["lead_time"] },
                "balance_lead_fake" : { None: balance_json["lead_time_fake"] },
                "balance_operator" : { None: balance_json["operators"] },
                "balance_operator_fake" : { None: balance_json["fake_operators"] },
                "balance_earliness" : { None: balance_json["earliness"] },
                "balance_tardiness" : { None: balance_json["tardiness"] },
                # "orders_finished_time" : orders_finished_time,
            }
        }
        return (ilp_input)
    
    def create_ilp_model(self, ilp_dict: dict = {}):
        """Create ILP abstract model.

        Args:
            ilp_dict (dict, optional): Sets weights and different settings for the Abstract Model.
                                    Defaults to {} := weights are 1 and all constraints included.

        Returns:
            UNKNOWN: The Pyomo Abstract Model used for ILP
        """
        model = instanciate_ilp_model(ilp_dict)
        return (model)

    def objective_function_weights(self, weight: dict) -> None:
        self.model = instanciate_ilp_model(weight)

    def run_ilp_instance(self, ilp_data: dict, timelimit: int | None = None) -> None:
        """Creates an instance of the abstract model and solves that instance.

        Args:
            ilp_data (dict): _description_
            timelimit (int | None, optional): _description_. Defaults to None.
        """
        if (len(ilp_data) == 0):
            return ([])
        solved_instance = run_ilp(self.model, ilp_data, timelimit=timelimit)
        return (solved_instance)

    def update_from_ilp_solution(self, ilp_solution, t_interval) -> None:
        """Updates Environment's class variable self.schedule according to the solution of the ILP.

        Args:
            ilp_solution (UNKNOWN): the solved instance of the ilp
        """
        if (len(ilp_solution) == 0):
            return()

        def instance_2_numpy(
                # instance_data: pyo.Var | pyo.Param | pyo.Set | pyo.RangeSet, 
                instance_data, 
                shape_array: np.ndarray | list = [] 
            ) -> float | np.ndarray:
            """Converts parameters, variables or ints that starts with "instance." and has a lower dimension than 4.
            The return will be a numpy array/matrix but just the value in the case of a single value (dimension of 0).
            In the case of a single value the shape_array should be an empty array "[]".

            Args:
                instance_data (pyomo.Var, pyomo.Param or pyomo.Set): This is your input data ex. "instance.num_machines" and should always start with "instance.".
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
        
        def update_unlocked_operations(ilp_solution: np.ndarray, num_unlocked_oper: int):
            for operation in range(num_unlocked_oper):
                machine, start_time = np.where(ilp_solution[:,operation,:] == 1)
                self.schedule[0,self.mapping_unlocked_opers[operation]] = machine[0]
                self.schedule[2,self.mapping_unlocked_opers[operation]] = start_time[0] + t_interval[0]
            
        num_machines = len(self.machines)
        num_unlocked_oper = instance_2_numpy(ilp_solution.num_opers)
        num_time_indices = instance_2_numpy(ilp_solution.num_time_indices)
        solution_shape = [num_machines, num_unlocked_oper, num_time_indices]
        ilp_solution_np = instance_2_numpy(ilp_solution.assigned, solution_shape)
        update_unlocked_operations(ilp_solution_np, num_unlocked_oper)

    def get_objective_value(self, weight_json):
        """returns objective function value and then the actual time without weights.

        Args:
            settings_json (_type_): _description_
            weight_json (_type_): _description_

        Returns:
            _type_: _description_
        """
        balance_json = self.get_balance_weight()
        def make_span_real():
            start_time = 0
            end_time = 0
            for i in range(len(self.schedule[2,:])):
                exec_time = np.ceil(self.schedule[1,i].execution_time/self.time_step_size)
                time = self.schedule[2,i] + exec_time
                if (time > end_time):
                    end_time = time
            weighted = (end_time-start_time)*weight_json["make_span"]*balance_json["make_span_real"]
            return (weighted, (end_time-start_time))
        
        def make_span():
            return(np.sum(self.schedule[2,:])*weight_json["make_span"]*balance_json["make_span"],np.sum(self.schedule[2,:]))
            # start_time = np.zeros([len(self.schedule[2,:])])
            # end_time = np.zeros([len(self.schedule[2,:])])
            # for i in range(len(self.schedule[2,:])):
            #     exec_time = np.ceil(self.schedule[1,i].execution_time/self.time_step_size)
            #     end_time[i] = self.schedule[2,i] #+ exec_time
            # cum_sum = np.sum(end_time)- np.sum(start_time)
            # weighted = cum_sum*weight_json["make_span"]*balance_json["make_span"]
            # return (weighted, cum_sum)

        def lead_time():
            order_lead_time = np.zeros([len(self.orders)])
            for iOrder, order in enumerate(self.orders):
                opers = np.array(order.operations)
                oper_indices = np.where(np.isin(self.schedule[1,:],opers))[0]
                order_start_time = np.min(self.schedule[2,oper_indices])
                order_end_time = 0
                for oper in oper_indices:
                    oper_exec_time = np.ceil(self.schedule[1,oper].execution_time/self.time_step_size)
                    oper_end_time = self.schedule[2,oper] + oper_exec_time
                    if (oper_end_time > order_end_time):
                        order_end_time = oper_end_time
                order_lead_time[iOrder] = (order_end_time - order_start_time)
            weighted = order_lead_time*weight_json["lead_time"]*balance_json["lead_time"]
            return(weighted, order_lead_time)

        def operators():
            operators_per_time = np.zeros([len(self.time_line)])
            for time_index in range(len(self.time_line)):
                for iOper, oper in enumerate(self.schedule[1,:]):
                    oper_exec = np.ceil(oper.execution_time/self.time_step_size)
                    oper_end_time = self.schedule[2,iOper] + oper_exec
                    oper_start_time = self.schedule[2,iOper]
                    if (time_index >= oper_start_time and time_index < oper_end_time):
                        operators_per_time[time_index] += oper.num_operators
            operator_diff = (operators_per_time 
                             - weight_json["max_amount_operators"]*np.ones([len(self.time_line)]))
            operator_diff = np.where(operator_diff < 0, 0, operator_diff)
            weighted = operator_diff*weight_json["operators"]*balance_json["operators"]
            return(weighted ,operators_per_time)
        
        def operators_fake():
            operators_per_time = np.zeros([len(self.time_line)])
            for time_index in range(len(self.time_line)):
                for iOper, oper in enumerate(self.schedule[1,:]):
                    oper_exec = np.ceil(oper.execution_time/self.time_step_size)
                    oper_end_time = self.schedule[2,iOper] + oper_exec
                    oper_start_time = self.schedule[2,iOper]
                    if (time_index >= oper_start_time and time_index < oper_end_time):
                        operators_per_time[time_index] += oper.num_operators
            operator_diff = (operators_per_time 
                             - weight_json["max_amount_operators"]*np.ones([len(self.time_line)]))
            operator_diff = np.where(operator_diff < 0, 0, operator_diff)
            weighted = operator_diff*weight_json["fake_operators"]*balance_json["fake_operators"]
            return(weighted ,operators_per_time)

        def earliness_and_tardiness():
            #is due date ceil, floor or just int. I used ceil
            """returns earliness, tardiness and last how late in time every order is.

            Returns:
                _type_: _description_
            """
            order_due_date = np.zeros(len(self.orders))
            fg_time = np.zeros(len(self.orders))
            for iOrder, order in enumerate(self.orders):
                for oper in order.operations:
                    if (oper.parent == None):
                        idx = np.where(np.isin(self.schedule[1,:], oper))[0]
                        order_due_date[iOrder] = np.floor(order.due_date/self.time_step_size)
                        oper_exec = np.ceil(oper.execution_time/self.time_step_size)
                        fg_time[iOrder] = self.schedule[2,idx] + oper_exec
                        break
            delta_time = order_due_date - fg_time
            earl_per_order = np.zeros(len(self.orders))
            tard_per_order = np.zeros(len(self.orders))
            for iDelta, delta in enumerate(delta_time):
                if (delta > 0):
                    earl_per_order[iDelta] = weight_json["earliness"]*balance_json["earliness"]*delta
                else:
                    tard_per_order[iDelta] = -weight_json["tardiness"]*balance_json["tardiness"]*delta
            return (earl_per_order, tard_per_order, delta_time)
        objective_value_short = {
            "total_value": (make_span()[0] 
                            + sum(lead_time()[0]) 
                            + max(operators()[0])
                            + sum(operators_fake()[0])
                            + sum(earliness_and_tardiness()[0])
                            + sum(earliness_and_tardiness()[1])),
            "make_span": make_span()[0],
            "lead_time": sum(lead_time()[0]),
            "operators": max(operators()[0]),
            "operators_fake": sum(operators_fake()[0]),
            "earliness": sum(earliness_and_tardiness()[0]),
            "tardiness": sum(earliness_and_tardiness()[1]),
        }
        objective_value_long = {
            "make_span": make_span()[0],
            "lead_time": lead_time()[0],
            "operators": operators()[0],
            "earliness": earliness_and_tardiness()[0],
            "tardiness": earliness_and_tardiness()[1],
        }
        behavior_long = {
            "make_span": make_span()[1],
            "lead_time": lead_time()[1],
            "operators": operators()[1],
            "earliness and tardiness": earliness_and_tardiness()[2],
        }
        return (objective_value_short, objective_value_long, behavior_long)


    def plot(self, real_size = True, save_plot = False, hide_text = False) -> None:
        """Plots the scheme

        Args:
            real_size (bool, optional): Weather or not the real length of each operation should be
                                        displayed or the length of time_step_size. Defaults to True.
            save_plot (bool, optional): Weather or not the plot should be saved or shown. 
                                        Defaults to Show, False.
            hide_text (bool, optional): Weather or not the plot should contain text or not. 
                                        Defaults to contain text, False.
            t_interval (list[int], optional): The time interval to plot within. Default is the
                                              entire interval.
        """
        def set_and_get_order_color(order_tree: str, seed: int) -> tuple[float,float,float,float]:
            if not order_tree in order_tree_dict:
                random.seed(seed)
                random_red = random.random()
                random_green = random.random()
                random_blue = random.random()
                order_tree_dict[order_tree] = (random_red, random_green, random_blue, 1)
            random.seed(None)
            return order_tree_dict[order_tree]
        
        def get_operation_text(operation: "Operation") -> str:
            operation_text = ""
            if operation.order.operations[-1] == operation:
                operation_text = operation.name.split("FG_")[-1]
            return operation_text
        
        def get_machine_ticks() -> list[str]:
            machine_ids = list(self.machines.keys())
            machine_types = list(self.machines.values())
            machine_ticks = [f"{mtype} (ID: {mID})" for mID, mtype in zip(machine_ids, machine_types)]
            return (machine_ticks)
        
        def get_order_and_due_date() -> dict:
            orders_due_dates = {}
            for iOper, oper in enumerate(self.schedule[1,:]):
                order = oper.order
                if not (order in orders_due_dates):
                    finished_time = self.schedule[2,iOper] + math.ceil(oper.execution_time/self.time_step_size)
                    orders_due_dates[order.name] = {"due_date" : order.due_date,
                                                    "finished_time" : finished_time,
                                                    "final_oper" : oper}
                oper_finished_time = self.schedule[2,iOper] + math.ceil(oper.execution_time/self.time_step_size)
                if (oper_finished_time >= orders_due_dates[order.name]["finished_time"]):
                    orders_due_dates[order.name] = {"due_date" : order.due_date,
                                                    "finished_time" : oper_finished_time,
                                                    "final_oper" : oper}
            return (orders_due_dates)
        
        plt.clf(), plt.cla(), plt.close()
        order_due_dates = get_order_and_due_date()
        fig, ax = plt.subplots(figsize=(16,9))
        order_tree_dict = {}
        num_seed = 42
        for operation_index in range(self.schedule.shape[1]):
            machine_id = self.schedule[0,operation_index]
            operation = self.schedule[1,operation_index]
            start_time = self.schedule[2,operation_index]
            exec_time = self.time_step_size + (real_size)*(operation.execution_time - self.time_step_size)
            offset = start_time*self.time_step_size
            order_color = set_and_get_order_color(operation.order.name, seed=num_seed)
            finished_operation_text = get_operation_text(operation)
            plt.barh(y=machine_id, width=exec_time, left=offset, alpha=0.4, 
                     color=order_color, edgecolor='black', linewidth=0.7)
            if (not hide_text):
                plt.text(x=offset, y=machine_id+0.3, s=finished_operation_text)
            num_seed += 1
        plt.title("Scheduled Operations", fontsize=20)
        ax.set_xlabel(" Time", fontsize=16)
        ax.set_ylabel("Machines", fontsize=16)
        plt.gca().invert_yaxis()
        
        # preferred_ticks = 15
        # num_ticks = min(preferred_ticks,self.time_line.shape[0])
        # tick_distances = math.ceil(self.time_line.shape[0]/num_ticks)
        # actual_num_ticks = (self.time_line.shape[0]+1)//tick_distances
        # xticks = np.arange(actual_num_ticks)
        # for iXtick, xtick in enumerate(self.time_line):
        #     if (iXtick%tick_distances==0):
        #        xticks[iXtick//tick_distances] = xtick 
        # ax.set_xticks(xticks)
        # xlim_upper = self.time_line[-1]
        # ax.set_xlim([0, xlim_upper])
        xticks = self.time_line
        ax.set_xticks(xticks)

        machine_ticks = get_machine_ticks()
        ax.set_yticks(np.arange(len(self.machines)))
        ax.set_yticklabels(machine_ticks, fontsize = 12)
        machine_colors = ["linen", "lavender"]
        machine_color = machine_colors[0]
        previous_machine_type = ""
        new_ytick = []
        for ytick in ax.get_yticklabels():
            current_machine_type = ytick.get_text().split(" ")[0]
            new_ytick += [ytick.get_text().split("(")[1].split(")")[0]]
            if (current_machine_type != previous_machine_type):
                new_ytick[-1] = ytick.get_text().replace("(", "").replace(")", "")
                previous_machine_type = current_machine_type
                machine_color = next(item for item in machine_colors if item != machine_color)
            ytick.set_backgroundcolor(machine_color)
        ax.set_yticklabels(new_ytick, fontsize = 12)
        ax.xaxis.grid(True, alpha=0.5)
        due_dates = [order_due_dates[order]["due_date"] for order in order_due_dates.keys()]
        due_dates_np = np.array(due_dates)
        order_names = [order for order in order_due_dates.keys()]
        order_final_oper = [order_due_dates[order]["final_oper"] for order in order_due_dates.keys()]
        due_date_duplicates = np.zeros(len(due_dates_np))
        for iDue_date, due_date in enumerate(due_dates_np):
            duplicate_indices = np.where(due_dates_np == due_date)[0]
            due_date_duplicates[duplicate_indices] += 1
        num_seed = 42
        for iDue_date, due_date in enumerate(due_dates):
            num_duplicates = due_date_duplicates[iDue_date]
            label = order_final_oper[iDue_date].name.split("FG_")[-1]
            final_oper_index = np.where(np.isin(self.schedule[1,:], order_final_oper[iDue_date]))[0]
            final_oper_machine = self.schedule[0,final_oper_index]
            ymin = final_oper_machine-2
            ymax = final_oper_machine+2
            order_name = order_names[iDue_date]
            order_color = set_and_get_order_color(order_name, seed=num_seed)
            random_offset = random.randint(-int(self.time_line[-1]), int(self.time_line[-1]))
            random_offset /= 100
            # ax.vlines(x=due_date+random_offset, ymin=ymin, ymax=ymax, colors=order_color, 
            ax.vlines(x=due_date, ymin=ymin, ymax=ymax, colors=order_color, 
                      ls='--', lw=3, label=label, alpha=0.6)
            num_seed += 1
        ax.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
        if (hide_text):
            ax.xaxis.grid(False)
            ax.set_xticklabels([""])
        if save_plot:
            plot_path = os.path.dirname(os.path.abspath(__file__))+"/Plots/"
            plot_name = datetime.now().strftime("%H_%M_%S")
            print(plot_name)
            plt.savefig(plot_path+plot_name+".png")
            return
        plt.show()
        
    def schedule_to_csv(self) -> None:
        schedule_np = np.empty([5,self.schedule.shape[1]], dtype=object)
        cols = []
        for operation_index in range(self.schedule.shape[1]):
            operation = self.schedule[1,operation_index]
            oper_name = operation.name
            order = operation.order.id
            machine_id = self.schedule[0,operation_index]
            start_time = self.schedule[2,operation_index] * self.time_step_size
            finished_time = start_time + operation.execution_time
            schedule_np[0,operation_index] = oper_name
            schedule_np[1,operation_index] = machine_id
            schedule_np[2,operation_index] = start_time
            schedule_np[3,operation_index] = finished_time
            schedule_np[4,operation_index] = order
            cols += ["Oper_"+str(operation_index)]
        schedule_pd = pd.DataFrame(schedule_np, columns=cols, index=["Name", "Machine ID", "Start Time", "Finished Time", "Order ID"])
        csv_path = os.path.dirname(os.path.abspath(__file__))+"/CSV/"
        if not os.path.exists(csv_path):
            os.makedirs(csv_path)
        schedule_pd.to_csv(csv_path+"schedule.csv", index=True, sep="\t", encoding="utf-8")