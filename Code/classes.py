############### IMPORT #################
import numpy as np

############### CLASSES ################

class Operation:
    def __init__(self, id: int, name: str, execution_time: float, 
                 valid_machines: list, num_operators: float, parent_name: str) -> None:
                 #valid_machines: list, num_operators: float, parent: "Operation") -> None:
        self.id = id
        self.name = name
        self.execution_time = execution_time
        self.valid_machines = valid_machines
        self.num_operators = num_operators
        self.parent_name = parent_name
        if parent_name == "":
            self.parent_name = None
        self.parent = None

    def get_id(self):
        return(self.id)
    
    def get_name(self):
        return(self.name)
    
    def get_execution_time(self):
        return(self.execution_time)
    
    def get_valid_machines(self):
        return(self.valid_machines)
    
    def get_num_operators(self):
        return(self.num_operators)
    
    def get_parent(self):
        return(self.parent)
    
    def print_info(self):
        print(  f"Name: {self.name}\nID: {self.id}\n"
              + f"Execution Time: {self.execution_time}\n"
              + f"Valid Machines: {self.valid_machines}\n"
              + f"Number of Operators: {self.num_operators}\n"
              + f"Parent: {self.parent}\n"
              + f"Parent Name: {self.parent_name}\n")
    
    def set_parent(self, all_operations: list["Operation"]):
        for operation in all_operations:
            if operation.name == self.parent_name:
                self.parent = operation
                break
        pass
    

class Order:
    def __init__(self, id: int, name: str, operations: list, due_date: float) -> None:
        self.id = id
        self.name = name
        self.operations = operations
        self.due_date = due_date

    def get_id(self):
        return(self.id)
    
    def get_name(self):
        return(self.name)

    def get_operations(self):
        return(self.operations)
    
    def get_due_date(self):
        return(self.due_date)