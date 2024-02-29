############## IMPORT ################
from platform import machine
import numpy as np
import matplotlib.pyplot as plt
import os, json, re, math, random
from classes import Operation, Order
from Data_Converter.batch_data import Batch_Data
from ilp import create_and_run_ilp
import pyomo as pyo
import pandas as pd

############# ENVIRONMENT_CLASS_##############
class Environment:
    def __init__(self, data_json: dict) -> None:
        self.orders_json = data_json
        machine_data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/Data/Raw/lines_tab.txt"
        self.machines = self.initialize_machines(machine_data_path)
        self.initialize_operations_data()
        self.orders = self.initialize_orders()
        self.time = []
        self.time_step_size = None
        self.schedule = self.initial_schedule()
        #self.precedence = self.initialize_precedence()
        self.time = []
        self.max_oper = 1
        
        # list of operations that are sent into the ilp 
        self.mapping_unlocked_operations = []  # [ActualIndex]   
    
    ### INITIALIZATION ###
    def initialize_machines(self, data_path_machine_types: str) -> dict:
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
        return (machines)

    def initialize_operations_data(self) -> list["Operation"]:
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
            return (valid_machine_ids)

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
        self.operations_data = operations
    
    
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
            orders[order_tree] += [self.operations_data[iOp]]
            #self.operations_data[iOp].print_info()
        orders_list = []
        for iOrd, order_tree in enumerate(orders):
            ord = Order(id=iOrd,
                        name=order_tree,
                        operations=orders[order_tree])
            for operation in ord.operations:
                operation.set_order(ord)
                #operation.order.print_info()
            orders_list += [ord]
            #ord.print_info()
        return (orders_list)
    
    list_valid_statements = ["S1", "S2"]
    
    def initialize_precedence(self) -> np.ndarray:
        """Instanciates and returns a numpy matrix of the precedence. Dim: [num_oper x num_oper].
        The value 1 at index (i,j) indicates that Operation j must be executed AFTER the completion
        of operation i. Otherwise the element is zero.  

        Args:
            None
        """
        num_operations = len(self.operations_data)
        precedence = np.zeros([num_operations, num_operations])
        operation_list = self.schedule[1,:]
        for iOp,operation in enumerate(operation_list):
            precedence_operation = operation.parent
            precedence_index = np.where(operation_list == precedence_operation)
            precedence[iOp,precedence_index] = 1
        return precedence
    
    def initial_schedule(self) -> np.ndarray:
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
        self.time_step_size = max_exec_time
        self.time_line = self.time_step_size*np.arange(schedule_2d.shape[1])
        schedule = np.array(schedule)
        return (schedule)
    
    ### SCHEDULE_UNLOCKING ###
    def unlock_order(
            self, 
            num_orders: int,
            t_interval: list[int],
            order_idx: list[int] = [0]
        ) -> tuple[list[int], list[int]]:
        """this will return the input dict that ILP takes as input.
        it will unlock the amount of of operations given

        Args:
            amount (_type_): _description_
            t_interval (_type_): _description_
            order_idx (list, optional): _description_. Defaults to [0].
        """
        statement = "order"
        unlocked_oper, locked_oper = self.unlock(num_orders, t_interval, 
                                                 statement, order_idx)
        return (unlocked_oper, locked_oper)

    def unlock_machine(self, 
            num_machines: int,
            t_interval: list[int],
            machine_idx: list[int] =[0]
        ) -> tuple[list["Operation"], list["Operation"]]:
        statement = "machine"
        unlocked_oper, locked_oper = self.unlock(num_machines, t_interval, 
                                                 statement, machine_idx)
        return (unlocked_oper, locked_oper)
    
    def unlock(self, 
            num_objects: int,
            t_interval: list[int],
            check_statement: str,
            object_idx: list[int] =[0]
        ) -> tuple[list["Operation"], list["Operation"]]:
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
        if (len(object_idx) == 1 and object_idx[0] == 0):
            object_idx = random.sample(range(object_amount), num_objects)
        for oper_idx in range(len(self.schedule[0])):
            oper_info = [self.schedule[0][oper_idx],
                        self.schedule[1][oper_idx],
                        self.schedule[2][oper_idx]]
            if (oper_info[2] >= t_interval[0] and oper_info[2] < t_interval[1]):
                exe_time = math.ceil(oper_info[1].execution_time/self.time_step_size)
                check2 = oper_info[2] + exe_time <= t_interval[1]
                i = 0
                for idx in object_idx:
                    i += 1
                    idx = int(idx)
                    check1 = (oper_info[1].order == self.orders[idx])
                    if (check_statement == "machine"):
                        check1 = (oper_info[0] == idx)
                    if (check1 and check2):
                        unlocked_oper += [oper_idx]
                        break
                    if (i == len(object_idx)):
                        locked_oper += [oper_idx]
        locked_oper += self.line_check(t_interval[0])
        return(unlocked_oper, locked_oper)

    ### TIMELINE_MANAGEMENT ###
    def divide_timeline(self, num_divisions = 1) -> None:
        self.time_step_size = self.time_step_size/np.exp(2)
        pass
    
    ### ILP_HANDLING ###    
    def to_ilp(
            self, 
            unlocked_operations_indices: list[int], 
            locked_operations_indices: list[int],
            time_interval: list[int]
        ) -> dict:
        """Converts the data of the schedule to send to the ILP. Includes a list of operation index
        to optimize and a list of operation index to keep locked. Returns a dict

        Args:
            unlocked_operations_indices (list[int]): List of operation index to unlock from schedule.
            locked_operations_indices (list[int]): List of operation index to keep locked from schedule.
            time_interval (list[int]): Time interval to optimize over. (Currently seen as time indices)
        """
        def init_dict(dim_1: int, dim_2: int = 0) -> dict:
            init_dict = {}
            if dim_2 == 0:
                for iDim in range(1,dim_1+1):
                    init_dict[iDim] = 0
                return (init_dict)
            for iDim in range(1,dim_1+1):
                for jDim in range(1,dim_2+1):
                    init_dict[(iDim,jDim)] = 0
            return (init_dict)
        
        def get_valid_machines_and_exec_time() -> tuple[dict, dict]:
            valid_machines = init_dict(len(unlocked_operations_indices), len(self.machines))
            exec_time = init_dict(len(unlocked_operations_indices))
            for iUO, unlocked_operation_index in enumerate(unlocked_operations_indices):
                unlocked_operation = self.schedule[1,unlocked_operation_index]
                exec_time[iUO+1] = math.ceil(unlocked_operation.execution_time/self.time_step_size)
                for machine_id in unlocked_operation.valid_machine_ids:
                    valid_machines[(iUO+1,machine_id)] = 1
            return (valid_machines, exec_time)

        def get_locked_operations_info() -> tuple[dict, dict, dict]:
            locked_operation_machine = init_dict(len(locked_operations_indices))
            locked_operation_start_time = init_dict(len(locked_operations_indices))
            locked_operation_exec_time = init_dict(len(locked_operations_indices))
            for iLO, locked_operation_index in enumerate(locked_operations_indices):
                locked_operation = self.schedule[:,locked_operation_index]
                locked_operation_machine[iLO+1] = locked_operation[0]
                exec_time = math.ceil(locked_operation[1].execution_time/self.time_step_size)
                locked_operation_exec_time[iLO+1] = exec_time
                locked_operation_start_time[iLO+1] = locked_operation[2] + 1 
            return (locked_operation_machine, locked_operation_exec_time, locked_operation_start_time)
        
        def map_unlocked_operations():
            self.mapping_unlocked_operations.clear()
            self.mapping_unlocked_operations = unlocked_operations_indices
        
        def get_precedence():
            pass
        
        num_machines = { None: len(self.machines)}
        num_operations = { None: len(unlocked_operations_indices)}
        map_unlocked_operations()
        num_locked_operations = { None: len(locked_operations_indices)}
        num_time_indices = { None: time_interval[1] - time_interval[0]}
        valid_machines, exec_time = get_valid_machines_and_exec_time()
        #precedence = get_precedence()
        locked_oper_machine, locked_oper_exec_time, locked_oper_start_time = get_locked_operations_info()
        ilp_input = {
            None: {
                "num_machines" : num_machines,
                "num_operations" : num_operations,
                "num_locked_operations" : num_locked_operations,
                "num_time_indices" : num_time_indices,
                "valid_machines" : valid_machines,
                # "precedence" : precedence,
                "exec_time" : exec_time,
                "locked_oper_machine" : locked_oper_machine,
                "locked_oper_exec_time" : locked_oper_exec_time,
                "locked_oper_start_time" : locked_oper_start_time
            }
        }
        # with open(os.path.dirname(os.path.abspath(__file__))+"/Test_ilp.json", 'w') as json_file:
            # json.dump(ilp_input, json_file, indent=4)
        print(ilp_input)
        return (ilp_input)

    # Don't know the class of the solution atm
    #def run_ilp(self, ilp_dict: dict) -> pyo.AbstractModel | pyo.ConcreteModel:
    def run_ilp(self, ilp_dict: dict):
        """Runs the ILP, from the ILP file. Returns the solved instance of the ILP.
        
        Args:
            ilp_dict (dict): a dict corresponding to the content of a .dat file.
        """
        ilp_solution = create_and_run_ilp(ilp_dict)
        return (ilp_solution)
        
    # Don't know the class of instance atm
    def update_from_ilp_solution(self, ilp_solution) -> None:
        """Updates Environment's class variable self.schedule according to the solution of the ILP.

        Args:
            ilp_solution (UNKNOWN): the solved instance of the ilp
        """
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
            
        num_machines = len(self.machines)
        num_unlocked_oper = instance_2_numpy(ilp_solution.num_operations)
        num_time_indices = instance_2_numpy(ilp_solution.num_time_indices)
        solution_shape = [num_machines, num_unlocked_oper, num_time_indices]
        ilp_solution_np = instance_2_numpy(ilp_solution.assigned, solution_shape)
        for operation in range(num_unlocked_oper):
            machine, start_time = np.where(ilp_solution_np[:,operation,:] == 1)
            self.schedule[0,self.mapping_unlocked_operations[operation]] = machine[0]
            self.schedule[2,self.mapping_unlocked_operations[operation]] = start_time[0]

    ### MISC ###
    # def plot(self, t_interval: list[int]) -> None:
    def plot(self) -> None:
        # set t_interval default value to be the complete scheme
        fig, ax = plt.subplots()
        for operation_index in range(self.schedule.shape[1]):
            machine_id = self.schedule[0,operation_index]
            operation = self.schedule[1,operation_index]
            start_time = self.schedule[2,operation_index]
            exec_time = operation.execution_time
            #plt.barh(y=machine_id, width=exec_time, left=start_time*self.time_step_size, alpha=0.4)#, color=team_colors[row['team']], alpha=0.4)
            plt.barh(y=machine_id, width=self.time_step_size, left=start_time*self.time_step_size, alpha=0.4)#, color=team_colors[row['team']], alpha=0.4)
        plt.title('Project Management Schedule of Project X', fontsize=15)
        plt.gca().invert_yaxis()
        ax.set_xticks(self.time_line)
        ax.xaxis.grid(True, alpha=0.5)
        # ax.legend(handles=patches, labels=team_colors.keys(), fontsize=11)
        plt.show()
        plt.close()
    
    ############# HELP_FUNCTIONS ###############
    def line_check(self, time_of_line: int) -> list[int]:
        oper_on_line = []
        for oper_idx in range(len(self.schedule[0])):
            oper_info = [self.schedule[0][oper_idx],
                     self.schedule[1][oper_idx],
                     self.schedule[2][oper_idx]]
            check1 = oper_info[2] < time_of_line
            exe_time = math.ceil(oper_info[1].execution_time/self.time_step_size)
            check2 = oper_info[2] + exe_time >= time_of_line
            if (check1 and check2):
                oper_on_line += [oper_idx]
        return (oper_on_line)
    
    def find_int_in_string(self, string):
        return [int(match) for match in re.findall(r'\d+', string)]

############# TESTING ###############
if (__name__ == "__main__"):
    num_orders = 15
    batched_orders = Batch_Data(batch_size=num_orders)
    batched_data = batched_orders.get_batch()
    #with open(os.path.dirname(os.path.abspath(__file__))+"/Test.json", 'w') as json_file:
        # json.dump(batched_data, json_file, indent=4)
    env = Environment(batched_data)
    time_interval = [1,100]
    env.plot()
    num_runs = 4
    for iRun in range(num_runs):
        unlocked_operations, locked_operations = env.unlock_order(1, time_interval)
        ilp_dict = env.to_ilp(unlocked_operations,locked_operations,time_interval)
        ilp_solution = env.run_ilp(ilp_dict=ilp_dict)
        env.update_from_ilp_solution(ilp_solution)
    env.plot()