######### IMPORTS ##############
import numpy as np
import pandas as pd
import os, json, math, random, time
from environment import Environment
from datetime import datetime
from Data_Converter.batch_data import BatchData
from pyomo.opt import SolverFactory, TerminationCondition

class Scheduler:
    def __init__(self, num_orders: int | None = None, weight_json: dict | None = None, num_divides: int = 3):
        self.master_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.weight_json = weight_json
        if (weight_json is None):
            self.weight_json = {
                "max_amount_operators": 50,
                "make_span": 10,
                "lead_time": 100,
                "operators": 0,
                "fake_operators": 0,
                "earliness": 0,
                "tardiness": 0,
            }
        # with open(self.master_path+"/Data/Parsed_Json/batched_same_operator.json", "r") as f:
        with open(self.master_path+"/Data/Parsed_Json/batched.json", "r") as f:
            input_json = json.load(f)
        if (num_orders is not None):
            data_bank = BatchData(num_orders)
            input_json = data_bank.generate_batch().generate_due_dates(1000000, 2/3)
            data_bank.save_as_json(input_json, "/Parsed_Json/batched.json")
        self.num_divides = 3
        self.timeout = 60
        self.iter_per_divide = 2*np.ones([num_divides], dtype=int)
        self.env = Environment(input_json, self.weight_json)
        self.num_machines = len(self.env.machines)
        self.total_loops = 2*sum(self.iter_per_divide)
        self.loops_count = 0
        self.init_obj_value = self.env.get_objective_value()
        num_time_steps_good_test = 28
        num_oper_good_test = 15
        num_machines_good_test = 35
        self.max_permutations = math.factorial(num_oper_good_test)*math.comb(num_machines_good_test*num_time_steps_good_test, num_oper_good_test)
        self.balance_json = self.env.get_balance_weight()
        
    def is_too_complex(self, num_time_steps, num_unlock_oper):
        """Calculates if the complexity of the planned unlocked ILP is smaller or greater than a good reference.
        This returns true if the reference has a lower complexity and false otherwise. num_time_steps_good_test and 
        num_oper_good_test is used to calculate the good reference value.

        Args:
            num_time_steps (int): this is the planed amount of unlocked time steps.
            num_unlock_oper (int): this is the planed amount of unlocked operations.
        """
        possibilities = math.factorial(num_unlock_oper)*math.comb(self.num_machines*num_time_steps,num_unlock_oper)
        if (possibilities > self.max_permutations):
            return (True)
        return (False)

    def run_t_interval(self, type, type_list, t_interval):
        obj_value = self.env.get_objective_value()
        self.env.plot(real_size = True, save_plot = True)
        #print(obj_value[0] + "Plotted: ",datetime.now().strftime("%H_%M_%S"))
        sub_run_counter = 0
        t_interval2 = [t_interval[0], t_interval[1]]
        if (type == "machine"):
            num_types = self.num_machines
        elif (type == "order"):
            num_types = len(self.env.orders)
        else:
            print("not a valid type")
            return ()
        while (t_interval2[0] != t_interval2[1]):
            num_time_steps = int(t_interval2[1] - t_interval2[0])
            unlock, lock = self.env.unlock(0, t_interval2, type, type_list)
            if not (self.is_too_complex(num_time_steps, len(unlock))):
                if (sub_run_counter > 0):
                    print("\t\tsub run: "  + str(sub_run_counter + 1))
                if (len(unlock) == 0):
                    return ()
                dict = self.env.to_ilp(unlock, lock, t_interval2)
                self.csv_save(True, unlock, lock)
                instance, timed_out = self.env.run_ilp_instance(dict, timelimit=self.timeout)
                self.update_or_rerun(timed_out=timed_out, instance=instance, unlocked=unlock, locked=lock, t_interval=t_interval2)
                # self.env.update_from_ilp_solution(instance, t_interval2)
                self.csv_save(False, unlock, lock)
                return ()
            if (sub_run_counter == 0):
                print("\t\tsplit time axis: ")
            time_cut = int(t_interval2[0])
            step_length = int(num_time_steps)
            for i in range(num_time_steps):
                step_length = int(np.floor(step_length/2))
                if (step_length <= 0):
                    while (self.is_too_complex(num_time_steps, len(unlock))):
                        time_cut += 1
                        num_time_steps = int(t_interval2[1] - time_cut)
                        unlock, lock = self.env.unlock(0, [time_cut, t_interval2[1]], type, type_list)
                    sub_run_counter += 1
                    print("\t\tsub run: " + str(sub_run_counter))
                    print("time_interval: " + str(t_interval2[1]) + "," + str(time_cut))
                    break
                num_time_steps = int(t_interval2[1] - time_cut)
                unlock, lock = self.env.unlock(0, [time_cut, t_interval2[1]], type, type_list)
                if not (self.is_too_complex(num_time_steps, len(unlock))):
                    time_cut -= step_length
                else:
                    time_cut += step_length
                time_cut = int(time_cut)
            unlock, lock = self.env.unlock(0, [time_cut, t_interval2[1]], type, type_list)
            if (len(unlock) == 0):
                unlock, lock = self.env.unlock(0, t_interval2, type, type_list)
                self.run_semi_batch(unlock, lock, t_interval2)
                print("deadlock saved")
                return (0)
            if (num_time_steps <= 1):
                print("trying to unlock to many operations at one time, skipping this iteration and continues")
                return ()
            dict = self.env.to_ilp(unlock, lock, [time_cut, t_interval2[1]])
            instance, timed_out = self.env.run_ilp_instance(dict, timelimit=self.timeout)
            # self.env.update_from_ilp_solution(instance, [time_cut, t_interval2[1]])
            self.update_or_rerun(timed_out=timed_out, instance=instance, unlocked=unlock, locked=lock, t_interval=[time_cut, t_interval2[1]])
            t_interval2[1] = int(np.floor(3/4*(t_interval2[1] - time_cut)) + time_cut)

    def run_semi_batch(self, unlock, lock, t_interval):
        """this calculates if the complexity of the planned ILP run is to big and if so it will run multiple smaller
        ILP solves to get a resonable run time.

        Args:
            env (_type_): _description_
            unlock (_type_): _description_
            lock (_type_): _description_
            t_interval (_type_): _description_
        """
        unlock = np.array(unlock)
        num_time_steps = t_interval[1] - t_interval[0]
        num_unlock = len(unlock)
        too_big = self.is_too_complex(num_time_steps, num_unlock)
        if not (too_big):
            dict = self.env.to_ilp(unlock, lock, t_interval)
            instance, timed_out = self.env.run_ilp_instance(dict, timelimit=self.timeout)
            self.env.update_from_ilp_solution(instance, t_interval)
            is_schedule_improved =  self.env.is_schedule_improved()
            if not (is_schedule_improved):
                self.env.revert_update_from_ilp()
            return ()
        preferred_unlock_amount = num_unlock
        while (self.is_too_complex(num_time_steps, preferred_unlock_amount)):
            preferred_unlock_amount -= 1
        done_matrix = np.zeros([num_unlock,num_unlock]).astype(int)
        for diagonal in range(num_unlock):
            done_matrix[diagonal,diagonal] = 1
        while ((done_matrix == 0).any()):
            chosen_idx = np.array([]).astype(int)
            for unlock_idx in range(num_unlock):
                if (len(chosen_idx) >= preferred_unlock_amount):
                    break
                not_done_idx = np.where(done_matrix[unlock_idx,:] == 0)[0]
                if (len(not_done_idx) == 0):
                    continue
                chosen_idx = np.append(chosen_idx, [unlock_idx])
                chosen_idx = np.unique(chosen_idx)
                if (len(chosen_idx) + len(not_done_idx) <= preferred_unlock_amount):
                    chosen_idx = np.append(chosen_idx, not_done_idx)
                    chosen_idx = np.unique(chosen_idx)
                elif (len(chosen_idx) < preferred_unlock_amount):
                    random_idx = np.array(random.sample(not_done_idx.tolist(), 
                                                        preferred_unlock_amount - len(chosen_idx)))
                    chosen_idx = np.append(chosen_idx, random_idx)
                    chosen_idx = np.unique(chosen_idx)
                    break
            for i in chosen_idx:
                for j in chosen_idx:
                    done_matrix[i,j] = 1
            sub_batch = unlock[chosen_idx]
            locked_part = np.setdiff1d(unlock, sub_batch)
            if (len(lock) > 0):
                locked_part = np.append(locked_part, lock)
            dict = self.env.to_ilp(sub_batch, locked_part, t_interval)
            instance, _ = self.env.run_ilp_instance(dict, timelimit=self.timeout)
            self.env.update_from_ilp_solution(instance, t_interval)
            is_schedule_improved =  self.env.is_schedule_improved()
            if not (is_schedule_improved):
                self.env.revert_update_from_ilp()

    def run_group(self, type, num_unlock, t_interval, is_time_based=True):
        """This runs all orders or machines in random order and can
        be grouped together if num_unlock is not 1. it will also cut the 
        time at the end.

        Args:
            type (string): machine or order
            num_unlock (int): how many orders or machines should be unlocked together
            env (environment): environment
            t_interval (list[int,int]): time interval that you want to run.
        """
        if (type == "machine"):
            num_types = self.num_machines
        elif (type == "order"):
            num_types = len(self.env.orders)
        else:
            print("not a valid type")
            return ()
        shuffled_type = list(range(num_types))
        random.shuffle(shuffled_type)
        num_runs = math.ceil(num_types/num_unlock)
        runs_list = []
        for i_run in range(num_runs):
            run = np.array([]).astype(int)
            for j_type in range(num_unlock):
                idx = i_run*num_unlock + j_type
                if (idx == num_types):
                    break
                run = np.append(run, shuffled_type[idx])
            runs_list.append(run)
        for iRun, run in enumerate(runs_list):
            if (type == "order"):
                pass 
                # print("Unlocked orders: ",)[]
            run_start_time = time.time()
            if (is_time_based):
                self.run_t_interval(type, run, t_interval)
            else:
                unlock, lock = self.env.unlock(num_unlock, t_interval, type, run)
                if(len(unlock) == 0):
                    continue
                self.run_semi_batch(unlock, lock, t_interval)
            print("\trun: " + str(iRun + 1) + "/" + str(len(runs_list)) 
                      + "    " + self.time_to_string(run_start_time))

    def update_or_rerun(self, timed_out: bool, instance, unlocked: list[int],
                        locked: list[int], t_interval: list[int,int]) -> None:
        self.env.update_from_ilp_solution(instance, t_interval)
        is_schedule_improved =  self.env.is_schedule_improved()
        if not (is_schedule_improved):
            self.env.revert_update_from_ilp()
        if (timed_out):
            print("Maximum time limit reached")
            max_permutations_scaler = 5
            self.max_permutations = int(self.max_permutations/max_permutations_scaler)
            self.run_semi_batch(unlocked, locked, t_interval)
            self.max_permutations *= max_permutations_scaler

    def compute_theoretical_max(self) -> int:
        theoretical_max = {}
        make_span = self.theoretical_best_makespan()
        lead_time = self.theoretical_best_lead_time()
        operators = self.theoretical_best_num_operators()
        due_dates = self.theoretical_best_due_dates()
        theoretical_max["make_span"] = self.theoretical_best_makespan()
        theoretical_max["lead_time"] = self.theoretical_best_lead_time()
        theoretical_max["operators"] = self.theoretical_best_num_operators()
        theoretical_max["due_dates"] = self.theoretical_best_due_dates()
        theoretical_max["total_value"] = make_span + lead_time + operators + due_dates
        return (theoretical_max)
    
    def theoretical_best_makespan(self) -> int:
        total_time = 0
        for _, oper in enumerate(self.env.schedule[1,:]):
            total_time += math.ceil(oper.execution_time/self.env.time_step_size)
        factor = self.weight_json["make_span"]*self.balance_json["make_span"]
        # theoretical_best_makespan = factor*math.ceil(total_time/self.env.time_step_size)
        theoretical_best_makespan = factor*total_time
        # theoretical_best_makespan = math.ceil(total_time_in_index/len(self.env.machines))
        return (theoretical_best_makespan)
    
    def theoretical_best_makespan_due_date(self) -> int:
        orders = np.array(self.env.orders)
        orders_due_dates = np.array([order.due_date for order in orders])
        last_due_date = math.floor(np.amax(orders_due_dates)/self.env.time_step_size)
        return (last_due_date)
    
    def theoretical_best_lead_time(self) -> np.ndarray:
        def traverse_child(operation, current_lead_time: float):
            children = operation.children
            # current_best_lead_time = current_lead_time + operation.execution_time
            current_best_lead_time = current_lead_time + math.ceil(operation.execution_time/self.env.time_step_size)
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
            final_oper_of_order_index = oper_indices[np.argmax(self.env.schedule[2,oper_indices])]
            final_oper_of_order = self.env.schedule[1,final_oper_of_order_index]
            children = final_oper_of_order.children
            self.lead_time = 0
            current_lead_time = math.ceil(final_oper_of_order.execution_time/self.env.time_step_size)
            # current_lead_time = final_oper_of_order.execution_time
            for child in children:
                traverse_child(child, current_lead_time)
            # orders_best_lead_time[1,iOrder] = math.ceil(self.lead_time/self.env.time_step_size)
            orders_best_lead_time[1,iOrder] = self.lead_time
        factor = self.weight_json["lead_time"]*self.balance_json["lead_time"]
        theoretical_best_lead_time = factor*np.sum(orders_best_lead_time[1,:])
        return (theoretical_best_lead_time)                
    
    def theoretical_best_num_operators(self) -> int:
        return 0
    
    def theoretical_best_due_dates(self) -> int:
        return 0
    
    def print_progress(self):
        self.loops_count += 1
        print("############ Main Loop: " + str(self.loops_count) 
            + "/" + str(self.total_loops) + " ############")

    def time_to_string(self, start_time):
        time_value = time.time() - start_time
        time_value = np.ceil(10*time_value)/10
        string = str(time_value) + "s"
        return (string)

    def csv_save(self, is_data_new, unlock, lock):
        schedule = self.env.schedule
        np_save = np.zeros([4,len(schedule[1,:])])
        for i in range(len(schedule[1,:])):
            np_save[0,i] = schedule[0,i]
            np_save[1,i] = i
            np_save[2,i] = schedule[2,i]
        for i in unlock:
            np_save[3,i] = 1
        for i in lock:
            np_save[3,i] = -1
        df = pd.DataFrame(np_save)
        csv_path = self.master_path + "/Data/CSV/"
        if not os.path.exists(csv_path):
            os.makedirs(csv_path)
        if (is_data_new):
            df.to_csv(csv_path+"new.csv")
        else:
            df.to_csv(csv_path+"old.csv")

    def get_t_intervals(self, num_partitions: int):
        max_t_interval = len(self.env.time_line)
        amount_intervals = 2*num_partitions - 1
        single_interval_length = max_t_interval/num_partitions
        t_intervals = np.zeros([2,amount_intervals],dtype=int)
        for iTI in range(t_intervals.shape[1]):
            t_intervals[0,iTI] = round(single_interval_length/2*iTI)
            t_intervals[1,iTI] = round(single_interval_length/2*(iTI+2))
        return (t_intervals)

    def schedule(self):
        self.main_start_time = time.time()
        self.env.divide_timeline()
        for iDivide in range(self.num_divides):
            if (iDivide != 0):
                print("\n############ Divided Timeline ############")
                self.env.remove_excess_time()
                self.env.divide_timeline()
            num_time_axises = len(self.env.time_line)
            t_interval = [0, num_time_axises]
            for _ in range(self.iter_per_divide[iDivide]):
                num_unlock = 2
                self.print_progress()
                self.run_group("order", num_unlock=num_unlock-1, t_interval=t_interval)
                print(self.env.get_objective_value()[0])
                self.print_progress()
                self.run_group("order", num_unlock=num_unlock, t_interval=t_interval)
                print(self.env.get_objective_value()[0])

if __name__ == "__main__":
    scheduler = Scheduler()
    scheduler.env.plot()
    scheduler.schedule()
    print("Total Run Time: " + scheduler.time_to_string(scheduler.main_start_time))
    final_obj_value = scheduler.env.get_objective_value()
    print("initial objective value:\n",scheduler.init_obj_value[0])
    print("final objective value:\n",final_obj_value[0])
    theoretical_max = scheduler.compute_theoretical_max()
    scheduler.env.plot(real_size=True,save_plot=True)
    export_name = str(len(scheduler.env.orders))+"_operations"
    for key, value in scheduler.weight_json.items():
        export_name += "___"+key+"_"+str(value)
    scheduler.env.schedule_to_csv(export_name, theoretical_max=theoretical_max)