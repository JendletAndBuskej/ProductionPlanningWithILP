import os

###### global variables ######
file_to_run = "Code/main.py"
data_file = "Data/Pyomo/pyo_data.dat"

created_results_file = "results.yml"
results_folder = "Code/Results/"
solver = "glpk"


###### functions ######
def run_script(file_name, data_file_name, solver = solver):
    run_command = "pyomo solve " + file_name + " " + data_file_name + " --solver=" + solver
    os.system(run_command)
    iteration_int = find_latest_iteration()
    new_file_name = created_results_file[:-4] + str(iteration_int) + created_results_file[-4:]
    os.rename(created_results_file,results_folder + new_file_name)

def find_latest_iteration():
    new_iteration_of_results = 0
    files_in_results_folder = os.listdir(results_folder)

    if len(files_in_results_folder) == 0:
        return(new_iteration_of_results)
    
    iterations_array = []
    start = len(created_results_file) -4
    for file in files_in_results_folder:
        end = len(file) - 4
        iterations_array.append(int(file[start:end]))
    new_iteration_of_results = max(iterations_array) + 1
    
    return(new_iteration_of_results)


###### main ######
if __name__ == "__main__":
    run_script(file_to_run,data_file)




"""
import os

###### global variables ######
file_to_run = "Code/main.py"
data_file = "Data/Pyomo/pyo_data.dat"

created_results_file = "results.yml"
results_folder = "Code/Results/"
solver = "glpk"


###### functions ######
def run_script(file_name, data_file_name, solver = solver):
    run_command = "pyomo solve " + file_name + " " + data_file_name + " --solver=" + solver
    os.system(run_command)
    iteration_int = find_latest_iteration()
    new_file_name = created_results_file[:-4] + str(iteration_int) + created_results_file[-4:]
    os.rename(created_results_file,results_folder + new_file_name)

def find_latest_iteration():
    new_iteration_of_results = 0
    files_in_results_folder = os.listdir(results_folder)

    if len(files_in_results_folder) == 0:
        return(new_iteration_of_results)
    
    iterations_array = []
    start = len(created_results_file) -4
    for file in files_in_results_folder:
        end = len(file) - 4
        iterations_array.append(int(file[start:end]))
    new_iteration_of_results = max(iterations_array) + 1
    
    return(new_iteration_of_results)


###### main ######
if __name__ == "__main__":
    run_script(file_to_run,data_file)
"""