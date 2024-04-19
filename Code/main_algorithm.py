import numpy as np
from environment import Environment
from Data_Converter.batch_data import BatchData
import json, os, math

### INITIALIZATION ###
num_orders = 5 # roughly 3 days 
data_bank = BatchData(num_orders)
input_json = data_bank.get_batch()
input_json = data_bank.generate_due_dates("", [1, 1])
folder_path = os.path.dirname(__file__)
# with open(folder_path+"/test1337.json", "r") as f:
    # input_json = json.load(f)   
env = Environment(input_json)
env.create_ilp_model()    

### HELP_FUNCTIONS ###
def compute_idle_time():
    pass

# def chronological_order_of_operations():
    # chronological_order = np.empty([2,env.schedule.shape[1]])
    # chronological_order = np.argsort(to_sort[1])
    
def distance_weights():
    weights = np.empty([1,env.schedule.shape[1]])
    for iMachine in env.machines:
        opers_on_machine_indices = np.where(env.schedule[0,:] == iMachine)
        for opers in opers_on_machine_indices:
            to_sort = np.empty([2,opers.shape[0]])
            to_sort[0,:] = opers
            to_sort[1,:] = env.schedule[2,opers]
            sorted_indices = np.argsort(to_sort[1])
            sorted = to_sort[:, sorted_indices]
            to_subtract = np.zeros([1,sorted.shape[1]]) 
            to_subtract[0,1:] = sorted[1,:-1]
            oper_exec_time = np.array([math.ceil(oper.execution_time/env.time_step_size) 
                                       for oper in env.schedule[1,sorted[0,:].astype(int)]])
            to_subtract[0,1:] += oper_exec_time[:-1]
            weights[0,sorted[0,:].astype(int)] = sorted[1,:] - to_subtract
    return (weights) 

def create_weight_array():
    dist_weights = distance_weights()
    # print("\nDistances: ", dist_weights)
    weights_size = np.sum(dist_weights) + dist_weights.shape[1]
    weights = np.empty(weights_size.astype(int), dtype=int)
    index = 0
    for i, value in enumerate(dist_weights[0,:].astype(int)):
        for _ in range(value+1):
            weights[index] = i
            index += 1
    return (weights)

def get_random_unlocking(t_interval_size: int) -> tuple[int, int, list[int,int]]:
    num_opers = env.schedule.shape[1]
    weighted_array = create_weight_array()
    # print("Weights: ",weighted_array)
    random_oper = np.random.choice(weighted_array, 1)[0]
    # print("Selected: ",random_oper)
    random_oper_machine = env.schedule[0,random_oper]
    random_oper_order = env.schedule[1,random_oper].order.id
    random_oper_time_interval = get_random_time_interval(random_oper, t_interval_size)
    return (random_oper_machine, random_oper_order, random_oper_time_interval)
    
def get_random_time_interval(random_oper: int, t_interval_size: int) -> list[int,int]:
    oper_start_time = env.schedule[2,random_oper]
    lower_time_interval = oper_start_time - int(t_interval_size/2)
    to_shift_right = min(0,lower_time_interval)
    lower_time_interval = oper_start_time - int(t_interval_size/2) - to_shift_right
    upper_time_interval = oper_start_time + int(t_interval_size/2) - to_shift_right
    t_interval = [lower_time_interval, upper_time_interval]
    return (t_interval)

def unlock_with_probability(
        probs: list[int, int],
        num_entity_unlocks: int,
        t_interval: list[int, int],
        given_machines: int,
        given_order: int
    ) -> list[int,int]:
    """Randomizes what unlock method to use, according to a given probability.

    Args:
        probs (list[int, int]): Probability of using unlock_order and unlock_machine respectively.
        num_entity_unlocks (int): How many of the certain unlocking technique to use.
        t_interval (list[int, int]): Time interval to unlock within.

    Returns:
        tuple[int, int]: unlocked indices
    """
    if (sum(probs) > 1):
        print("Probs doesnt add upp to 1")
        return
    random_prob = np.random.random()
    if (probs[0] < random_prob):
        #print("unlock_order selected")
        unlocked_indices, locked_indices = env.unlock_order(num_entity_unlocks, t_interval)
        return (unlocked_indices, locked_indices)
    #print("unlock_machine selected")
    unlocked_indices, locked_indices = env.unlock_machine(num_entity_unlocks, t_interval)
    return (unlocked_indices, locked_indices)


### MAIN_LOOP ###
iRun = 1
factor = 15
max_runs = factor*25
# max_runs = 2
# env.plot(real_size=True, save_plot=True, hide_text=True)
env.divide_timeline()
run_complete = False
while True:
    if (iRun >= max_runs):
        break        
    if (iRun%(factor*8)==0 and run_complete):
        print(f"Timeline divided, step size is {env.time_step_size}...")
        run_complete = False
        env.plot(real_size=True, save_plot=True, hide_text=False)
        env.divide_timeline()
    max_t_interval = int(env.time_line[-1]/env.time_step_size)
    t_interval_size = max(1,int(0.1*len(env.time_line)))
    oper_machine, oper_order, oper_time_interval = get_random_unlocking(t_interval_size)
    # t_interval_upper_bound = np.random.randint(t_interval_size, max_t_interval)
    # t_interval_lower_bound = t_interval_upper_bound - t_interval_size
    # t_interval = [t_interval_lower_bound, t_interval_upper_bound]
    num_entity_unlocks = 1
    unlocked_indices, locked_indices = unlock_with_probability(
                                                               [0.25, 0.75],
                                                               num_entity_unlocks,
                                                               oper_time_interval,
                                                               oper_machine,
                                                               oper_order)
    if (unlocked_indices == []):
        continue
    
    iRun += 1
    run_complete = True
    print(f"Run {iRun-1} out of {max_runs-1}...")
    ilp_data = env.to_ilp(unlocked_indices, locked_indices, oper_time_interval)
    timelimit = 10
    instance_solution = env.run_ilp_instance(ilp_data, timelimit)
    env.update_from_ilp_solution(instance_solution, oper_time_interval)
    env.remove_excess_time()
env.plot(real_size=True, save_plot=True, hide_text=False)
"""
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

if not os.path.exists(results_folder):
        os.makedirs(results_folder)
        
    iteration_int = find_latest_iteration()
    new_file_name = created_results_file[:-4] + str(iteration_int) + created_results_file[-4:]
    
    os.rename(created_results_file,results_folder + new_file_name)
"""