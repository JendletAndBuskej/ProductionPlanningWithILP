############### IMPORT #################
import numpy as np

############### CLASSES ################

class Operation:
    def __init__(self,id,execution_time,valid_machines,nr_operators) -> None:
        self.id = id
        self.execution_time = execution_time
        self.valid_machines = valid_machines
        self.nr_operators = nr_operators

    def get_id(self):
        return(self.id)
    
    def get_execution_time(self):
        return(self.execution_time)
    
    def get_valid_machines(self):
        return(self.valid_machines)
    
    def get_nr_operators(self):
        return(self.nr_operators)
    
class Order:
    def __init__(self, operation_list, due_date) -> None:
        self.operations = operation_list
        self.due_date = due_date

    def get_operations(self):
        return(self.operations)
    
    def get_due_date(self):
        return(self.due_date)