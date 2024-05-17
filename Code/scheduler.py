import numpy as np
from environment import Environment
from Data_Converter.batch_data import BatchData
import json, os, math, timeit

class Scheduler:
    def __init__(self, num_orders: int | None = None, weight_json: dict = {}) -> None:
        project_path = os.path.dirname(os.path.dirname(__file__))
        if not (weight_json):
            self.weight_json = {
                "max_amount_operators": 10,
                "make_span": 1,
                "lead_time": 1,
                "operators": 0,
                "earliness": 0,
                "tardiness": 0
                }
        with open(project_path+"/Data/Parsed_Json/batched.json", "r") as f:
            input_json = json.load(f)
        if (num_orders):
            data_bank = BatchData(num_orders)
            input_json = data_bank.generate_batch().generate_due_dates(1000000, 2/3)
            data_bank.save_as_json(input_json, "/Parsed_Json/batched.json")
        self.env = Environment(input_json)
        self.env.create_ilp_model(weight_json)
    
    def unlock_with_probability(
            self,
            probs: list[int, int],
            t_interval: list[int, int],
            given_machines: int,
            given_order: int,
            num_entity_unlocks: int = 1,
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
            unlocked_indices, locked_indices = self.env.unlock_order(num_entity_unlocks, t_interval, given_order)
            return (unlocked_indices, locked_indices)
        unlocked_indices, locked_indices = self.env.unlock_machine(num_entity_unlocks, t_interval, given_machines)
        return (unlocked_indices, locked_indices)
    
    def compress(self):
        def chronological_order_of_operations():
            chronological_order = np.argsort(self.env.schedule[2,:])
            return (chronological_order)

        def get_random_sequence_of_operations():
            random_sequence = np.arange(self.env.schedule.shape[1])
            np.random.shuffle(random_sequence)
            return (random_sequence)

        def centralize_time_interval_of_operation(operation_index: int, time_interval_size: int) -> list[int, int]:
            operation_start_time = self.env.schedule[2,operation_index]
            lower_time_interval = operation_start_time - int(time_interval_size/2)
            to_shift_right = min(0,lower_time_interval)
            lower_time_interval = operation_start_time - int(time_interval_size/2) - to_shift_right
            upper_time_interval = operation_start_time + int(time_interval_size/2) - to_shift_right
            centralized_time_interval = [lower_time_interval, upper_time_interval]
            return (centralized_time_interval)
        
        chronological_order = chronological_order_of_operations()
        random_sequence = get_random_sequence_of_operations()
        t_interval_size = max(4,int((self.env.time_line[-1]/self.env.time_step_size)/(3*self.current_divide)))
        timelimit = 10
        print(f"Num operations is {chronological_order.shape[0]}, the expected max time is {timelimit*2*chronological_order.shape[0]}.")
        for oper_index in chronological_order:
            oper_time_interval = centralize_time_interval_of_operation(oper_index, t_interval_size)
            oper_machine = [self.env.schedule[0,oper_index]]
            oper_order = [self.env.schedule[1,oper_index].order.id]
            unlocked_indices, locked_indices = self.unlock_with_probability([0.15, 0.85],
                                                                            oper_time_interval,
                                                                            oper_machine,
                                                                            oper_order,
                                                                            1)
            if (unlocked_indices == []):
                continue
            ilp_data = self.env.to_ilp(unlocked_indices, locked_indices, oper_time_interval)
            # instance_solution = self.env.run_ilp_instance(ilp_data)
            start_time = timeit.default_timer() 
            instance_solution = self.env.run_ilp_instance(ilp_data, timelimit)
            finished_time = timeit.default_timer()
            print("Time of a chrono run: ",finished_time - start_time)
            self.env.update_from_ilp_solution(instance_solution, oper_time_interval)
        for oper_index in random_sequence:
            oper_time_interval = centralize_time_interval_of_operation(oper_index, t_interval_size)
            oper_machine = [self.env.schedule[0,oper_index]]
            oper_order = [self.env.schedule[1,oper_index].order.id]
            unlocked_indices, locked_indices = self.unlock_with_probability([0.15, 0.85],
                                                                            oper_time_interval,
                                                                            oper_machine,
                                                                            oper_order,
                                                                            1)
            if (unlocked_indices == []):
                continue
            ilp_data = self.env.to_ilp(unlocked_indices, locked_indices, oper_time_interval)
            start_time = timeit.default_timer() 
            instance_solution = self.env.run_ilp_instance(ilp_data, timelimit)
            finished_time = timeit.default_timer()
            print("Time of a random run: ",finished_time - start_time)
            self.env.update_from_ilp_solution(instance_solution, oper_time_interval)
        self.env.remove_excess_time()
    
    def schedule(self, scaler, runs_factor, num_divides):
        iRun = 1
        max_runs = scaler*runs_factor
        self.current_divide = 1
        self.env.plot(real_size=True, save_plot=True, hide_text=False)
        # print("Returning in schedule and skipping compress to skip ctrl-c")
        # return
        self.env.divide_timeline()
        compress_start_time = timeit.default_timer() 
        while True: 
            if (iRun >= max_runs): break        
            if (iRun%(scaler*runs_factor//num_divides)==0):
                print(f"Timeline divided, step size is {self.env.time_step_size}...")
                self.env.plot(real_size=True, save_plot=True, hide_text=False)
                self.env.divide_timeline()
                self.current_divide += 1
            iRun += 1
            print(f"Compressing, Run {iRun-1} out of {max_runs-1}...")
            self.compress()
        compress_finished_time = timeit.default_timer()
        self.env.plot(real_size=True, save_plot=True, hide_text=False)
        self.env.schedule_to_csv()
        print(compress_finished_time - compress_start_time)
        return (self)
    
    def compute_theoretical_max(self) -> int:
        print("theoretical_best_makespan: ", self.theoretical_best_makespan())
        print("theoretical_best_makespan_due_date: ", self.theoretical_best_makespan_due_date())
        print("theoretical_best_lead_time: ", self.theoretical_best_lead_time())
        print("theoretical_best_num_operators: ", self.theoretical_best_num_operators())
        print("theoretical_best_due_dates: ", self.theoretical_best_due_dates())
    
    def theoretical_best_makespan(self) -> int:
        total_time = 0
        for iOper, oper in enumerate(self.env.schedule[1,:]):
            total_time += oper.execution_time
        total_time_in_index = math.ceil(total_time/self.env.time_step_size)
        theoretical_best_makespan = math.ceil(total_time_in_index/len(self.env.machines))
        return (theoretical_best_makespan)
    
    def theoretical_best_makespan_due_date(self) -> int:
        orders = np.array(self.env.orders)
        orders_due_dates = np.array([order.due_date for order in orders])
        last_due_date = math.floor(np.amax(orders_due_dates)/self.env.time_step_size)
        return (last_due_date)
        
    def theoretical_best_lead_time(self) -> np.ndarray:
        def traverse_child(operation, current_lead_time: float):
            children = operation.children
            current_best_lead_time = current_lead_time + operation.execution_time
            if (not children):
                if (current_best_lead_time > self.lead_time):
                    self.lead_time = current_best_lead_time
                    return
            for child in children:
                traverse_child(child, current_best_lead_time)
            
        orders_best_lead_time = np.empty([2,len(self.env.orders)], dtype=object)
        orders_best_lead_time[0,:] = np.array(self.env.orders)
        for iOrder, order in enumerate(self.env.orders):
            opers_np = np.array(order.operations)
            oper_indices = np.where(np.isin(self.env.schedule[1,:], opers_np))[0]
            final_oper_of_order_index = oper_indices[np.amax(self.env.schedule[2,oper_indices])]
            final_oper_of_order = self.env.schedule[1,final_oper_of_order_index]
            children = final_oper_of_order.children
            self.lead_time = 0
            current_lead_time = final_oper_of_order.execution_time
            for child in children:
                traverse_child(child, current_lead_time)
            orders_best_lead_time[1,iOrder] = self.lead_time
        return (orders_best_lead_time)                
    
    def theoretical_best_num_operators(self) -> int:
        return 0
    
    def theoretical_best_due_dates(self) -> int:
        return 0
    
if __name__ == "__main__":
    # scheduler = Scheduler(num_orders=8)
    scheduler = Scheduler()
    # scheduler.compute_theoretical_max()
    scheduler.schedule(scaler=2,runs_factor=2,num_divides=3)
### INITIALIZATION ###
# num_orders = 5 # roughly 3 days 
# data_bank = BatchData(num_orders)
# input_json = data_bank.get_batch()
# input_json = data_bank.generate_due_dates("", [1, 1])
# data_bank.save_as_json(input_json, "/Parsed_Json/batched.json")
# project_path = os.path.dirname(os.path.dirname(__file__))
# with open(project_path+"/Data/Parsed_Json/batched.json", "r") as f:
    # input_json = json.load(f)   
# env = Environment(input_json)
# env.create_ilp_model(weight_json) 
### HELP_FUNCTIONS ###
# def compute_idle_time():
    # pass
# 
# def compress():
    # chronological_order = chronological_order_of_operations()
    # random_sequence = get_random_sequence_of_operations()
    # t_interval_size = max(4,int((env.time_line[-1]/env.time_step_size)/(3*current_divide)))
    # timelimit = 10
    # for oper_index in chronological_order:
        # oper_time_interval = centralize_time_interval_of_operation(oper_index, t_interval_size)
        # oper_machine = [env.schedule[0,oper_index]]
        # oper_order = [env.schedule[1,oper_index].order.id]
        # unlocked_indices, locked_indices = unlock_with_probability([0.5, 0.5],
                                                                #    1,
                                                                #    oper_time_interval,
                                                                #    oper_machine,
                                                                #    oper_order)
        # if (unlocked_indices == []):
            # continue
        # ilp_data = env.to_ilp(unlocked_indices, locked_indices, oper_time_interval)
        # instance_solution = env.run_ilp_instance(ilp_data)#, timelimit)
        # env.update_from_ilp_solution(instance_solution, oper_time_interval)
    # for oper_index in random_sequence:
        # oper_time_interval = centralize_time_interval_of_operation(oper_index, t_interval_size)
        # oper_machine = [env.schedule[0,oper_index]]
        # oper_order = [env.schedule[1,oper_index].order.id]
        # unlocked_indices, locked_indices = unlock_with_probability([0.5, 0.5],
                                                                #    1,
                                                                #    oper_time_interval,
                                                                #    oper_machine,
                                                                #    oper_order)
        # if (unlocked_indices == []):
            # continue
        # ilp_data = env.to_ilp(unlocked_indices, locked_indices, oper_time_interval)
        # instance_solution = env.run_ilp_instance(ilp_data, timelimit)
        # env.update_from_ilp_solution(instance_solution, oper_time_interval)
    # env.remove_excess_time()
# 
# def chronological_order_of_operations():
    # chronological_order = np.argsort(env.schedule[2,:])
    # return (chronological_order)
# 
# def get_random_sequence_of_operations():
    # random_sequence = np.arange(env.schedule.shape[1])
    # np.random.shuffle(random_sequence)
    # return (random_sequence)
# 
# def centralize_time_interval_of_operation(operation_index: int, time_interval_size: int) -> list[int, int]:
    # operation_start_time = env.schedule[2,operation_index]
    # lower_time_interval = operation_start_time - int(time_interval_size/2)
    # to_shift_right = min(0,lower_time_interval)
    # lower_time_interval = operation_start_time - int(time_interval_size/2) - to_shift_right
    # upper_time_interval = operation_start_time + int(time_interval_size/2) - to_shift_right
    # centralized_time_interval = [lower_time_interval, upper_time_interval]
    # return (centralized_time_interval)