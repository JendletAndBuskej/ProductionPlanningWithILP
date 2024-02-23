############## IMPORT ################
from math import ceil
import string
import numpy as np
import matplotlib.pyplot as plt
import os, json
import "classes.py"

############# ENVIRONMENT_CLASS_##############

class Environment:

    def __init__(self, data_jason: string) -> None:
        with open(data_jason, "r") as f:
            orders_json = json.load(f)
        self.operations = np.array()
        self.orders = np.array()
        self.on_machine = np.array()
        self.on_time = np.array()
        self.valid_machines = np.matrix()
        

    def unlock_order(self, amount, t_interval , order_idx=[0]):
        if (len(idx) == 1 and idx[0] == 0):
            idx = ceil(len(self.orders)*np.random.rand([amount]))
        else:
            idx = order_idx
        unlock_operations = []
        for i in idx:
            order = self.orders[i]
            unlock_operations.append(order.get_operations())
        operations = self.orders[idx]
        return(operations_to_ilp(operations))
    
    def operations_to_ilp(self, operations, t_interval):
        """this should return the input for ilp.
        remember that this should be with active locked
        operations as well


        Args:
            operations (_type_): _description_
            t_interval (_type_): _description_
        """
        pass



        # dim = len(self.operations)
        # self.presidence = np.zeros([dim, dim])
        # for i in range(dim):
        #     for j in range(dim):
        #         self.presidence[i,j] = 