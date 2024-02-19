from matplotlib import pyplot as plt
import yaml, os, re, numpy as np, ast

code_path = os.path.dirname(os.path.abspath(__file__))
master_path = os.path.dirname(code_path)
results_path = code_path + "/Results/" 
pyomo_data_path = master_path + "/Data/Pyomo/pyo_data.dat"



def get_data_dims():
    with open(pyomo_data_path, 'r') as file:
        pyomo_data = file.read()
    
    list_dims = {}
    pyomo_rows = pyomo_data.split("\n")
    
    for pyo_data in pyomo_rows:
        if len(list_dims) == 3:
            break
        if "num_machines" in pyo_data:
            list_dims["num_machines"] = find_integer_in_string(pyo_data)
        if "num_operations" in pyo_data:
            list_dims["num_operations"] = find_integer_in_string(pyo_data)
        if "num_time_indices" in pyo_data:
            list_dims["num_time_indices"] = find_integer_in_string(pyo_data)
        
    return list_dims



def get_operation_times(num_operations):
    with open(pyomo_data_path, 'r') as file:
        pyomo_data = file.read()
    
    operations_times = np.zeros([num_operations])
    pyomo_rows = pyomo_data.split("\n")
    
    for iRow, pyo_data in enumerate(pyomo_rows):
        if "operation_time" in pyo_data:
            for ot in range(num_operations):
                operations_times[ot] = int(pyomo_rows[iRow+ot+1].split(" ")[-1])
        
    return operations_times
    


def find_integer_in_string(input_string):
    match = re.search(r'\d+', input_string)
    if match:
        return int(match.group())
    else:
        return None
    
    
    
def get_yaml_data(yaml_filename): 
    with open(results_path+yaml_filename, 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
        
    return data



def get_variable_yaml(yaml_filename, dims):
    all_data = get_yaml_data(yaml_filename)
    solution_data = all_data["Solution"]

    variable_data = None
    for sol_data in solution_data:
        if "Variable" in sol_data:
            variable_data = sol_data["Variable"]
            break
        
    if variable_data == None:
        print("Could not find Variable data of the file: ",yaml_filename)
        return 
        
    
    num_machines = dims["num_machines"]
    num_operations = dims["num_operations"]
    num_time_indices = dims["num_time_indices"]
    indicies = num_operations*[[]]
    gantt_matrix = np.zeros([num_machines,num_operations,num_time_indices])

    for iVar, variable_info in enumerate(variable_data):
        indicies[iVar] = get_variable_idx(variable_info)
        machine = indicies[iVar][0] - 1 
        operation = indicies[iVar][1] - 1
        time_index = indicies[iVar][2] - 1
        gantt_matrix[machine, operation, time_index] = 1
        
    return gantt_matrix
   

def get_variable_idx(variable_string):
    return ast.literal_eval(variable_string.split("assigned")[-1])
    
    

def plot_gantt(gantt_matrix, operations_times):
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
        

def create_and_visualize_gantt_schem(yaml_filename):
    dims = get_data_dims()
    gantt = get_variable_yaml(yaml_filename, dims)
    operation_times = get_operation_times(dims["num_operations"])
    plot_gantt(gantt,operation_times)
    
    
    
if __name__ == "__main__":
    create_and_visualize_gantt_schem("results4.yml")