import numpy as np
from environment import Environment
from Data_Converter.batch_data import BatchData

### INITIALIZATION ###
num_orders = 60
data_bank = BatchData(num_orders)
input_json = data_bank.get_batch()
env = Environment(input_json)
env.create_ilp_model()    

### HELP_FUNCTIONS ###
def unlock_with_probability(probs: list[int, int], num_entity_unlocks: int, t_interval: list[int, int]):
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
        print("unlock_order selected")
        unlocked_indices, locked_indices = env.unlock_order(num_entity_unlocks, t_interval)
        return (unlocked_indices, locked_indices)
    print("unlock_machine selected")
    unlocked_indices, locked_indices = env.unlock_machine(num_entity_unlocks, t_interval)
    return (unlocked_indices, locked_indices)
    

### MAIN_LOOP ###
iRun = 1
factor = 5
max_runs = factor*60
env.plot(real_size=True, save_plot=True, hide_text=True)
run_complete = False
env.divide_timeline()
while True:
    if (iRun >= max_runs):
        break        
    if (iRun%10 == 0):
        env.plot(real_size=True, save_plot=True, hide_text=True)
    if (iRun%(factor*30)==0 and run_complete):
        print(f'Timeline divided, step size is {env.time_step_size}...')
        run_complete = False
        env.divide_timeline()
    max_t_interval = int(env.time_line[-1]/env.time_step_size)
    t_interval_size = int(0.1*len(env.time_line))
    t_interval_upper_bound = np.random.randint(t_interval_size, max_t_interval)
    t_interval_lower_bound = t_interval_upper_bound - t_interval_size
    t_interval = [t_interval_lower_bound, t_interval_upper_bound]
    num_entity_unlocks = 1
    unlocked_indices, locked_indices = unlock_with_probability([0.6, 0.4], num_entity_unlocks, t_interval)
    if (unlocked_indices == []):
        continue
    
    iRun += 1
    run_complete = True
    print(f'Run {iRun-1} out of {max_runs-1}...')
    ilp_data = env.to_ilp(unlocked_indices, locked_indices, t_interval)
    instance_solution = env.run_ilp_instance(ilp_data)
    env.update_from_ilp_solution(instance_solution, t_interval)
env.plot(real_size=True, save_plot=True, hide_text=True)

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