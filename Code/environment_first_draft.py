
#####################################################################
## THIS_IS_JUST_TO_TAKE_SPARE_PARTS_WORK_ON_ENVIRONMENT.PY_INSTEAD ##
#####################################################################


############## IMPORT ################
from math import ceil
import string
import numpy as np
import matplotlib.pyplot as plt
import os, json
from classes import *
import pyomo as pyo
import pandas as pd

############# ENVIRONMENT_CLASS_##############

class Environment:

    def __init__(self, data_json: string) -> None:
        with open(data_json, "r") as f:
            orders_json = json.load(f)
        self.operations = np.array()
        self.orders = np.array()
        
        self.schedule_matrix = initial_scedule()
        self.last_sent_operations = []

    def unlock_order(self, amount, t_interval , order_idx=[0]):
        """this will return the input dict that ILP takes as input.
        it will unlock the amount of of operations given

        Args:
            amount (_type_): _description_
            t_interval (_type_): _description_
            order_idx (list, optional): _description_. Defaults to [0].
        """
        if (len(idx) == 1 and idx[0] == 0):
            idx = ceil(len(self.orders)*np.random.rand([amount]))
        else:
            idx = order_idx
        unlock_operations = []
        for i in idx:
            order = self.orders[i]
            unlock_operations += order.get_operations()
        self.last_sent_operations = unlock_operations
        return(operations_to_ilp(unlock_operations, t_interval))

    def unlock_machine(self, amount, t_interval , machine_idx=[0]):
        if (len(idx) == 1 and idx[0] == 0):
            idx = ceil(len(self.orders)*np.random.rand([amount]))
        else:
            idx = machine_idx
        unlock_operations = []
        for i in idx:
            
        self.last_sent_operations = unlock_operations
        return(operations_to_ilp(unlock_operations, t_interval))
        
    
    def operations_to_ilp(self, operations, t_interval):
        """this should return the input for ilp.
        remember that this should be with active locked
        operations as well


        Args:
            operations (_type_): _description_
            t_interval (_type_): _description_
        """
        pass

    def uppdate_from_ilp_instance(self, ilp_output):
        """_summary_

        Args:
            ilp_output (_type_): _description_
        """
        def instance_2_numpy(instance_data: pyo.Var | pyo.Param | pyo.Set | pyo.RangeSet, 
                             shape_array: np.ndarray | list = [] ) -> any:
            """Converts parameters, variables or ints that starts with "instance." and has a lower dimension than 4.
            The return will be a numpy array/matrix but just the value in the case of a single value (dimension of 0).
            In the case of a single value the shape_array should be an empty array "[]".

            Args:
                instance_data (pyomo.Var, pyomo.Param or pyomo.set): This is your input data ex. "instance.num_machines" and should always start with "instance.".
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
                            flat_idx = shape_array[1]*shape_array[2]*i + shape_array[2]*j + k
                            solution_matrix[i,j,k] = solution_flat_matrix[flat_idx,0]
            return(solution_matrix)


    def initial_scedule(self):
        """this will create the first scedule and return it as a 3x3 pandas with 
        """

        pass

    def plot(t_intervall):
        pass
            '''
        def plot_gantt(gantt_matrix, operations_times): #operation_time?
            fig, ax = plt.subplots()
            gantt_dims = gantt_matrix.shape
            for operation in range(gantt_dims[1]):
                gantt_of_operation = gantt_matrix[:,operation,:]
                machine, start_of_operation = np.where(gantt_of_operation == 1)
                plt.barh(y=machine, width=operations_times[operation], left= start_of_operation)#, color=team_colors[row['team']], alpha=0.4)
            plt.title('Project Management Schedule of Project X', fontsize=15)
            plt.gca().invert_yaxis()
            ax.xaxis.grid(True, alpha=0.5)
            #ax.legend(handles=patches, labels=team_colors.keys(), fontsize=11)
            plt.show()
            '''



        # dim = len(self.operations)
        # self.presidence = np.zeros([dim, dim])
        # for i in range(dim):
        #     for j in range(dim):
        #         self.presidence[i,j] = 