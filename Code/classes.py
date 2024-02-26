############### IMPORT #################
import numpy as np

############### CLASSES ################

class Operation:
    def __init__(self, id: int, execution_time: float, 
                 valid_machines: list, num_operators: float, parent: str) -> None:
        self.id = id
        self.execution_time = execution_time
        self.valid_machines = valid_machines
        self.num_operators = num_operators
        self.parent = parent

    def get_id(self):
        return(self.id)
    
    def get_execution_time(self):
        return(self.execution_time)
    
    def get_valid_machines(self):
        return(self.valid_machines)
    
    def get_num_operators(self):
        return(self.num_operators)
    
    def get_parent(self):
        return(self.parent)
    

class Order:
    def __init__(self, id: int, operations: list, due_date: float) -> None:
        self.id = id
        self.operations = operations
        self.due_date = due_date

    def get_id(self):
        return(self.id)

    def get_operations(self):
        return(self.operations)
    
    def get_due_date(self):
        return(self.due_date)