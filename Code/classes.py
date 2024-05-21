############### IMPORT #################
import numpy as np
import math
############### CLASSES ################

class Operation:
    def __init__(self, id: int, name: str, execution_time: float, valid_machine_ids: list[int],
                 num_operators: float, parent_name: str) -> None:
        self.id = id
        self.name = name
        self.execution_time = execution_time
        self.valid_machine_ids = valid_machine_ids
        self.num_operators = num_operators
        self.parent_name = parent_name
        if parent_name == "":
            self.parent_name = None
        self.parent = None
        self.children = []
        self.order = None

    def get_id(self) -> int:
        return(self.id)
    
    def get_name(self) -> str:
        return(self.name)
    
    def get_execution_time(self) -> float:
        return(self.execution_time)
    
    def get_valid_machine_ids(self) -> list[int]:
        return(self.valid_machine_ids)
    
    def get_num_operators(self) -> float:
        return(self.num_operators)
    
    def get_parent(self) -> "Operation":
        return(self.parent)
    
    def set_parent(self, all_operations: list["Operation"]):
        for operation in all_operations:
            if operation.name == self.parent_name:
                self.parent = operation
                break
            
    def set_children(self, children: list["Operation"]):
        # if not isinstance(children, Operation):
            # raise TypeError("The Operation's order variable must be of the class 'Order'")
        self.children += children
    
    def set_order(self, order: "Order"):
        if not isinstance(order, Order):
            raise TypeError("The Operation's order variable must be of the class 'Order'")
        self.order = order
        
    def print_info(self):
        print(  f"Operation Name: {self.name}\n"
              + f"Operation ID: {self.id}\n"
              + f"Execution Time: {self.execution_time}\n"
              + f"Valid Machine IDs: {self.valid_machine_ids}\n"
              + f"Number of Operators: {self.num_operators}\n"
              + f"Parent: {self.parent}\n"
              + f"Children: {self.children}\n"
              + f"Parent Name: {self.parent_name}\n")
              #+ f"Order: {self.order}\n")

class Order:
    def __init__(self, id: int, name: str, operations: list["Operation"],
                 due_date: float = None) -> None:
        self.id = id
        self.name = name
        self.operations = operations
        self.due_date = due_date

    def get_id(self) -> int:
        return(self.id)
    
    def get_name(self) -> str:
        return(self.name)

    def get_order_size(self) -> int:
        return(len(self.operations))

    def get_operations(self) -> list["Operation"]:
        return(self.operations)
    
    def get_due_date(self) -> float:
        return(self.due_date)
    
    def print_info(self):
        print(  f"Order Tree Name: {self.name}\n"
              + f"Order ID: {self.id}\n"
              + f"Operations: {self.operations}\n"
              + f"Order Size: {self.get_order_size()}\n"
              + f"Due Date: {self.due_date}\n")

class Machine:
    def __init__(self, id: int, m_type: str) -> None:
        self.id = id
        self.m_type = m_type
        self.operations = None
        self.start_times = None
        self.finish_times = None
        self.max_time = 0.0
        
    def add_operation(self, operation: "Operation", start_time: float, indexed_time: float | None = None):
        if not isinstance(operation, Operation):
            raise TypeError("The Operation added to a machine must be of the class 'Operation'")
        operation_np = np.array(operation)
        start_time_np = np.array(start_time)
        finish_time_np = np.array(start_time+operation.execution_time)
        if (start_time < self.max_time):
            raise ValueError("Can only add operations to the left of the max_time on a machine")
        if (self.operations is None):
            self.operations = np.empty([1,], dtype=object)
            self.operations[0] = operation
            self.start_times = np.empty([1,], dtype=object)
            self.start_times[0] = start_time
            self.finish_times = np.empty([1,], dtype=object)
            self.finish_times[0] = start_time + operation.execution_time
        else:
            self.operations = np.hstack((self.operations, operation_np))
            self.start_times = np.hstack((self.start_times, start_time_np))
            self.finish_times = np.hstack((self.finish_times, finish_time_np))
        self.max_time = start_time + operation.execution_time
        if (indexed_time is not None): 
            self.max_time = start_time + indexed_time*math.ceil(operation.execution_time/indexed_time)
        
    def print_info(self):
        if (self.operations is None): return
        for iOper, oper in enumerate(self.operations):
            print(  f"Operation Name: {oper.name}\n"
                  + f"Operation ID: {oper.id}\n"
                  + f"Operation Start Time: {self.start_times[iOper]}\n"
                  + f"Operation Finish Time: {self.finish_times[iOper]}\n")
        print(f"Machine Max Time: {self.max_time}")