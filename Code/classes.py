############### IMPORT #################
import numpy as np

############### CLASSES ################

class Operation:
    def __init__(self, id: int, name: str, execution_time: float, 
                 valid_machine_ids: list[int], num_operators: float, parent_name: str) -> None:
        self.id = id
        self.name = name
        self.execution_time = execution_time
        self.valid_machine_ids = valid_machine_ids
        self.num_operators = num_operators
        self.parent_name = parent_name
        if parent_name == "":
            self.parent_name = None
        self.parent = None

    def get_id(self) -> int:
        return(self.id)
    
    def get_name(self) -> str:
        return(self.name)
    
    def get_execution_time(self) -> float:
        return(self.execution_time)
    
    def get_valid_machine_ids(self) -> int:
        return(self.valid_machine_ids)
    
    def get_num_operators(self):
        return(self.num_operators)
    
    def get_parent(self):
        return(self.parent)
    
    def print_info(self):
        print(  f"Operation Name: {self.name}\n"
              + f"Operation ID: {self.id}\n"
              + f"Execution Time: {self.execution_time}\n"
              + f"Valid Machine IDs: {self.valid_machine_ids}\n"
              + f"Number of Operators: {self.num_operators}\n"
              + f"Parent: {self.parent}\n"
              + f"Parent Name: {self.parent_name}\n")
    
    def set_parent(self, all_operations: list["Operation"]):
        for operation in all_operations:
            if operation.name == self.parent_name:
                self.parent = operation
                break


class Order:
    def __init__(self, id: int, name: str, operations: list["Operation"], due_date: float = None) -> None:
        self.id = id
        self.name = name
        self.operations = operations
        self.due_date = due_date

    def get_id(self):
        return(self.id)
    
    def get_name(self):
        return(self.name)

    def get_order_size(self):
        return(len(self.operations))

    def get_operations(self):
        return(self.operations)
    
    def get_due_date(self):
        return(self.due_date)
    
    def print_info(self):
        print(  f"Order Tree Name: {self.name}\n"
              + f"Order ID: {self.id}\n"
              + f"Operations: {self.operations}\n"
              + f"Order Size: {self.get_order_size()}\n"
              + f"Due Date: {self.due_date}\n")