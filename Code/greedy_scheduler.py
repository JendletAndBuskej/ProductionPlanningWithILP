import numpy as np, json, os, random, math
from environment import Environment
import matplotlib.pyplot as plt 
from classes import Machine, Operation
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

class GreedyScheduler:
    def __init__(self, input_json: dict, settings_json: dict) -> None:
        self.env = Environment(input_json)
        self.operations = self.env.schedule[1,:]
        self.time_step_size = settings_json["time_step_size"]
        self.num_operators = settings_json["num_operators"]
        self.max_timeline = settings_json["max_timeline"]
        self.w_makespan = settings_json["make_span"]
        self.w_lead_time = settings_json["lead_time"]
        self.w_operators = settings_json["operators"]
        self.w_fake_operators = settings_json["fake_operators"]
        self.w_earliness = settings_json["earliness"]
        self.w_tardiness = settings_json["tardiness"]
        self.orders = self.env.orders
        self.operations_precedence_time = np.zeros([1,self.operations.shape[0]],dtype=float)
        self.machines = self.instanciate_machines()
        self.oper_num_childs = self.instanciate_num_childs()
        self.selection_mask = self.get_selection_mask()
        self.due_date_prio = self.instanciate_due_date_prio()
        
    def instanciate_machines(self):
        machine_info = self.env.machines
        machines = []
        for m_id, m_type in machine_info.items():
            machines += [Machine(id=m_id,m_type=m_type)]
        return (np.array(machines)) 
    
    def instanciate_num_childs(self):
        oper_num_childs = np.zeros([2,self.operations.shape[0]], dtype=object)
        oper_num_childs[0,:] = self.operations
        for iOper, operation in enumerate(self.operations):
            parent = operation.parent
            if not (parent):
                continue
            parent_index = np.where(np.isin(oper_num_childs[0,:], parent))
            oper_num_childs[1,parent_index] += 1
        return (oper_num_childs)
    
    def instanciate_due_date_prio(self):
        due_dates = np.array([[order, order.due_date] for order in self.orders])
        due_dates = due_dates[np.argsort(due_dates[:, 1])]
        due_date_prio = np.zeros([1,self.operations.shape[0]], dtype=int)
        num_orders = len(self.orders)
        for order in due_dates[:,0]:
            oper = np.array(order.operations)
            order_indicies = np.where(np.isin(self.operations, oper))[0]
            due_date_prio[0,order_indicies] += num_orders
            num_orders -= 1
        return (due_date_prio)
    
    def reset_machines(self):
        for machine in self.machines:
            machine.operations = None
            machine.start_times = None
            machine.finish_times = None
            machine.max_time = 0.0
    
    def reset_scheduler(self):
        self.operations_precedence_time = np.zeros([1,self.operations.shape[0]],dtype=float)
        self.oper_num_childs = self.instanciate_num_childs()
        self.selection_mask = self.get_selection_mask()
        self.due_date_prio = self.instanciate_due_date_prio()
        self.reset_machines()

    def get_selection_mask(self):
        valid_selections = np.where(self.oper_num_childs[1,:] == 0)[0]
        selection_mask = np.zeros([1,self.operations.shape[0]],dtype=int)
        selection_mask[0,valid_selections] = 1
        return (selection_mask)
    
    def update_selection_mask(self, selected_operation: Operation):
        parent = selected_operation.parent
        operation_index = np.where(np.isin(self.oper_num_childs[0,:], selected_operation))
        self.oper_num_childs[1,operation_index] -= 1
        if not (parent):
            self.selection_mask = self.get_selection_mask()
            return
        parent_index = np.where(np.isin(self.oper_num_childs[0,:], parent))
        self.oper_num_childs[1,parent_index] -= 1
        self.selection_mask = self.get_selection_mask()
    
    def update_precedence_times(self, selected_operation: Operation, end_time: float):
        parent = selected_operation.parent
        if not (parent):
            return
        parent_index = np.where(np.isin(self.oper_num_childs[0,:], parent))
        current_precedence_time = self.operations_precedence_time[0,parent_index]
        if (end_time >= current_precedence_time):
            self.operations_precedence_time[0,parent_index] = end_time

    def get_earliest_machine_id(self, machine_ids: list[int], precedence_time: float) -> int:
        earliest_time = abs(self.machines[machine_ids[0]].max_time - precedence_time)
        earliest_machine_id = machine_ids[0]
        for machine_id in machine_ids:
            time_to_earliest_placement = abs(self.machines[machine_id].max_time - precedence_time)
            if (time_to_earliest_placement < earliest_time):
                earliest_time = time_to_earliest_placement
                earliest_machine_id = machine_id
        return (earliest_machine_id)
    
    def get_obj_value(self):
        scaled_time_step = self.env.time_step_size/100000
        balance_json = {
            "make_span": 2*scaled_time_step/(self.env.schedule.shape[1]),
            "lead_time": scaled_time_step/(len(self.env.orders)),
            "operators": 1/self.num_operators,
            "fake_operators": 1/len(self.env.time_line),
            "earliness": scaled_time_step/(len(self.env.orders)),
            "tardiness": scaled_time_step/(len(self.env.orders)),
        }
        obj_value = {}
        obj_value["lead_time"] = self.w_lead_time*balance_json["lead_time"]*self.get_obj_lead_time()
        # obj_value["make_span"] = balance_json["make_span"]*self.get_obj_make_span()
        obj_value["make_span"] = self.w_makespan*balance_json["make_span"]*self.get_obj_make_span_fake()
        # obj_value["due_date"] = self.w_makespan*balance_json["make_span"]*self.get_obj_due_date()
        obj_value["tardiness"] = self.w_tardiness*balance_json["tardiness"]*self.get_obj_tardiness()
        obj_value["earliness"] = self.w_earliness*balance_json["earliness"]*self.get_obj_earliness()
        obj_value["operators"] = self.w_operators*balance_json["operators"]*self.get_obj_operators()
        obj_value["total"] = np.sum(np.array([value for key, value in obj_value.items()]))
        return obj_value

    def get_obj_lead_time(self):
        orders_lead_time = np.ones([3,len(self.orders)], dtype=object)
        orders_lead_time[0,:] = self.orders
        orders_lead_time[1,:] *= 10**10
        for machine in self.machines:
            opers = machine.operations
            if (opers is None): continue
            for iOper, oper in enumerate(opers):
                start_time = math.ceil(machine.start_times[iOper]/self.time_step_size)
                finish_time = math.ceil(machine.finish_times[iOper]/self.time_step_size)
                order = oper.order
                order_index = np.where(np.isin(orders_lead_time[0,:], order))[0]
                order_lead_time_start = orders_lead_time[1,order_index]
                order_lead_time_finish = orders_lead_time[2,order_index]
                if (start_time < order_lead_time_start):
                    orders_lead_time[1,order_index] = start_time
                if (finish_time > order_lead_time_finish): 
                    orders_lead_time[2,order_index] = finish_time
        lead_time = np.sum(np.diff(orders_lead_time[1:,:], axis=0))
        return lead_time

    def get_obj_make_span(self):
        machines_max_time = np.array([machine.max_time for machine in self.machines])
        make_span = np.amax(machines_max_time)
        return make_span

    def get_obj_make_span_fake(self):
        make_span_fake = 0
        for machine in self.machines:
            if (machine.start_times is None): continue
            start_times = np.array([math.ceil(start_time/self.time_step_size) for start_time in machine.start_times])
            # start_times = np.array(math.ceil(machine.start_times/self.time_step_size))
            start_times_sum = np.sum(start_times)
            make_span_fake += start_times_sum
        return make_span_fake

    def get_obj_operators(self):
        width_schedule = self.get_obj_make_span()
        num_time_intervals = math.ceil(width_schedule/self.time_step_size)
        num_operators = np.zeros([1,num_time_intervals], dtype=float)
        for machine in self.machines:
            opers = machine.operations
            if (opers is None): continue
            for iOper, oper in enumerate(opers):
                start_time_interval = math.floor(machine.start_times[iOper]/self.time_step_size)
                finish_time_interval = math.ceil(machine.finish_times[iOper]/self.time_step_size)
                time_interval = np.arange(start_time_interval, finish_time_interval)
                num_operators[0,time_interval] += oper.num_operators
        remove_target = self.num_operators*np.ones([1,num_time_intervals])
        # overspill_operators = num_operators - remove_target
        overspill_operators = num_operators
        print("\n",overspill_operators)
        overspill_operators[overspill_operators < self.num_operators] = self.num_operators
        print(overspill_operators)
        operators_value = np.sum(overspill_operators)
        return operators_value

    def get_obj_due_date(self):
        orders_due_date = np.ones([3,len(self.orders)], dtype=object)
        orders_due_date[0,:] = self.orders
        orders_due_date[2,:] = np.array([order.due_date for order in self.orders])
        for machine in self.machines:
            opers = machine.operations
            if (opers is None): continue
            for iOper, oper in enumerate(opers):
                finish_time = machine.finish_times[iOper]
                order = oper.order
                order_index = np.where(np.isin(orders_due_date[0,:], order))[0]
                order_finished_time = orders_due_date[1,order_index]
                if (finish_time > order_finished_time): 
                    orders_due_date[1,order_index] = finish_time
        due_date = np.sum(np.diff(orders_due_date[1:,:], axis=0))
        return due_date
    
    def get_obj_tardiness(self):
        orders_due_date = np.ones([3,len(self.orders)], dtype=object)
        orders_due_date[0,:] = self.orders
        orders_due_date[2,:] = np.array([order.due_date for order in self.orders])
        for machine in self.machines:
            opers = machine.operations
            if (opers is None): continue
            for iOper, oper in enumerate(opers):
                finish_time = math.ceil(machine.finish_times[iOper]/self.time_step_size)
                order = oper.order
                order_index = np.where(np.isin(orders_due_date[0,:], order))[0]
                order_finished_time = math.ceil(orders_due_date[1,order_index]/self.time_step_size)
                if (finish_time > order_finished_time): 
                    orders_due_date[1,order_index] = finish_time
        tardiness = 0
        for iOrd in range(orders_due_date.shape[1]):
            if orders_due_date[1,iOrd] - orders_due_date[2,iOrd] >= 0:
                tardiness += orders_due_date[1,iOrd] - orders_due_date[2,iOrd]
        return tardiness
    
    def get_obj_earliness(self):
        orders_due_date = np.ones([3,len(self.orders)], dtype=object)
        orders_due_date[0,:] = self.orders
        orders_due_date[2,:] = np.array([order.due_date for order in self.orders])
        for machine in self.machines:
            opers = machine.operations
            if (opers is None): continue
            for iOper, oper in enumerate(opers):
                finish_time = math.ceil(machine.finish_times[iOper]/self.time_step_size)
                order = oper.order
                order_index = np.where(np.isin(orders_due_date[0,:], order))[0]
                order_finished_time = math.ceil(orders_due_date[1,order_index]/self.time_step_size)
                if (finish_time > order_finished_time): 
                    orders_due_date[1,order_index] = finish_time
        earliness = 0
        for iOrd in range(orders_due_date.shape[1]):
            if orders_due_date[1,iOrd] - orders_due_date[2,iOrd] <= 0:
                earliness += orders_due_date[2,iOrd] - orders_due_date[1,iOrd]
        return earliness
    
    def make_span_scheduling(self):
        self.reset_scheduler()
        while True:
            if not (np.any(self.selection_mask)): break
            prio_selections = self.due_date_prio*self.selection_mask
            valid_operations = np.where(prio_selections == np.amax(prio_selections))[1]
            valid_operations_precedence_time = self.operations_precedence_time[0,valid_operations]
            selected_operation_index = valid_operations[np.argmin(valid_operations_precedence_time)]
            selected_operation = self.operations[selected_operation_index]
            selected_operation_mIDS = selected_operation.valid_machine_ids
            selected_operation_precedence_time = self.operations_precedence_time[0,selected_operation_index]
            selected_machine_id = self.get_earliest_machine_id(selected_operation_mIDS, selected_operation_precedence_time)
            selected_machine = self.machines[selected_machine_id]
            operation_start_time = max(selected_machine.max_time, selected_operation_precedence_time)
            operation_end_time = operation_start_time + selected_operation.execution_time
            if (self.time_step_size is not None):
                operation_end_time = self.time_step_size*math.ceil(operation_end_time/self.time_step_size)
            indexed_time = self.time_step_size
            selected_machine.add_operation(selected_operation, operation_start_time, indexed_time)
            self.update_selection_mask(selected_operation)
            self.update_precedence_times(selected_operation, operation_end_time)
        self.get_obj_value()

    def _get_best_comb_operators(self, spill_over: float):
        target = self.num_operators - spill_over
        elements = np.array([oper.num_operators for oper in self.operations])*self.selection_mask
        elements = elements[0]
        possible_sums = {0: []}
        closest_sum_value = 0
        closest_combination_indices = []
        for idx, elem in enumerate(elements):
            if elem == 0:
                continue
            new_sums = {}
            for current_sum, combination in possible_sums.items():
                new_sum = current_sum + elem
                if new_sum not in possible_sums:
                    new_combination = combination + [idx]
                    new_sums[new_sum] = new_combination
                    # if new_sum <= target and new_sum > closest_sum_value:
                    if abs(target - new_sum) <= abs(target - closest_sum_value):
                        closest_sum_value = new_sum
                        closest_combination_indices = new_combination        
            possible_sums.update(new_sums)
        return closest_combination_indices

    def _get_total_amount_operators(self):
        total_amount_operators = 0
        for oper in self.operations:
            total_amount_operators += oper.num_operators
        return total_amount_operators
    
    def operators_scheduling(self):
        self.reset_scheduler()
        spill_overs = [0.0]
        iTI = 0
        while True:
            if not (np.any(self.selection_mask)): break
            if (iTI + 1 > len(spill_overs)): spill_overs += [0.0]
            spill_over = spill_overs[iTI]  
            selected_operations_indices = self._get_best_comb_operators(spill_over)
            for iOper, selected_operation_index in enumerate(selected_operations_indices):
                selected_operation = self.operations[selected_operation_index]
                selected_operation_mIDS = selected_operation.valid_machine_ids
                selected_operation_precedence_time = self.operations_precedence_time[0,selected_operation_index]
                selected_machine_id = self.get_earliest_machine_id(selected_operation_mIDS, selected_operation_precedence_time)
                selected_machine = self.machines[selected_machine_id]
                operation_start_time = max(selected_machine.max_time, selected_operation_precedence_time, (iTI+1)*self.time_step_size)
                operation_end_time = operation_start_time + selected_operation.execution_time
                if (self.time_step_size is not None):
                    operation_end_time = self.time_step_size*math.ceil(operation_end_time/self.time_step_size)
                num_spill_overs_forward = math.ceil(operation_end_time/self.time_step_size) - iTI - 1
                if (len(spill_overs) < num_spill_overs_forward + iTI + 1):
                    to_add = num_spill_overs_forward + iTI + 1 - len(spill_overs)
                    spill_overs += to_add*[0.0]
                for iSpill, spill_over_elem in enumerate(spill_overs[iTI+1:iTI+num_spill_overs_forward+1]):
                    spill_over_elem += selected_operation.num_operators
                    spill_overs[iTI+1+iSpill] = spill_over_elem
                indexed_time = self.time_step_size
                selected_machine.add_operation(selected_operation, operation_start_time, indexed_time)
                self.update_selection_mask(selected_operation)
                self.update_precedence_times(selected_operation, operation_end_time)
            iTI += 1
            
    def plot(self, save_plot = False) -> None:
        """Plots the scheme

        Args:
            save_plot (bool, optional): Weather or not the plot should be saved or shown. 
                                        Defaults to Show, False.
        """
        def set_and_get_order_color(order_tree: str, seed: int) -> tuple[float,float,float,float]:
            if not order_tree in order_tree_dict:
                random.seed(seed)
                random_red = random.random()
                random_green = random.random()
                random_blue = random.random()
                order_tree_dict[order_tree] = (random_red, random_green, random_blue, 1)
            random.seed(None)
            return order_tree_dict[order_tree]
        
        def get_operation_text(operation: "Operation") -> str:
            operation_text = ""
            if operation.order.operations[-1] == operation:
                operation_text = operation.name.split("FG_")[-1]
            return operation_text
        
        def get_machine_ticks() -> list[str]:
            machine_ids = []
            machine_types = []
            for machine in self.machines:
                machine_ids += [machine.id]
                machine_types += [machine.m_type]
            machine_ticks = [f"{mtype} (ID: {mID})" for mID, mtype in zip(machine_ids, machine_types)]
            return (machine_ticks)
        
        def get_order_and_due_date() -> dict:
            orders_due_dates = {}
            for machine in self.machines:
                opers = machine.operations
                if (opers is None): continue
                for iOper, oper in enumerate(opers):
                    order = oper.order
                    if not (order in orders_due_dates):
                        finished_time = machine.start_times[iOper] + math.ceil(oper.execution_time/self.time_step_size)
                        orders_due_dates[order.name] = {
                            "due_date" : order.due_date,
                            "finished_time" : finished_time,
                            "final_oper" : oper,
                            "machine" : machine.id
                        }
                    oper_finished_time = machine.start_times[iOper] + math.ceil(oper.execution_time/self.time_step_size)
                    if (oper_finished_time >= orders_due_dates[order.name]["finished_time"]):
                        orders_due_dates[order.name] = {
                            "due_date" : order.due_date,
                            "finished_time" : oper_finished_time,
                            "final_oper" : oper,
                            "machine" : machine.id
                        }
            return (orders_due_dates)
        
        order_due_dates = get_order_and_due_date()
        plt.clf(), plt.cla(), plt.close()
        fig, ax = plt.subplots(figsize=(16,9))
        order_tree_dict = {}
        num_seed = 42
        for iMach, machine in enumerate(self.machines):
            machine_id = machine.id
            operations = machine.operations
            start_times = machine.start_times
            finish_times = machine.finish_times
            if (operations is None): continue
            for iOper, oper in enumerate(operations):
                order_color = set_and_get_order_color(oper.order.name, seed=num_seed)
                finished_operation_text = get_operation_text(oper)
                plt.barh(y=machine_id, width=finish_times[iOper]-start_times[iOper], left=start_times[iOper], alpha=0.4, 
                         color=order_color, edgecolor='black', linewidth=0.7)
                plt.text(x=start_times[iOper], y=machine_id+0.3, s=finished_operation_text)
                num_seed += 1
        plt.title("Scheduled Operations", fontsize=20)
        ax.set_xlabel(" Time", fontsize=16)
        ax.set_ylabel("Machines", fontsize=16)
        plt.gca().invert_yaxis()
        max_time = self.get_obj_make_span()
        num_ticks = math.ceil(max_time/self.time_step_size)
        xticks = self.time_step_size*np.arange(num_ticks+1)
        ax.set_xticks(xticks)
        ax.xaxis.grid(True, alpha=0.5)
        # preferred_ticks = 15
        # num_ticks = min(preferred_ticks,self.time_line.shape[0])
        # tick_distances = math.ceil(self.time_line.shape[0]/num_ticks)
        # actual_num_ticks = (self.time_line.shape[0]+1)//tick_distances
        # xticks = np.arange(actual_num_ticks)
        # for iXtick, xtick in enumerate(self.time_line):
            # if (iXtick%tick_distances==0):
            #    xticks[iXtick//tick_distances] = xtick 
        # ax.set_xticks(xticks)
        # xlim_upper = self.time_line[-1]
        # xlim_upper = xticks[-1]
        # ax.set_xlim([0, xlim_upper])
        machine_ticks = get_machine_ticks()
        ax.set_yticks(np.arange(len(self.machines)))
        ax.set_yticklabels(machine_ticks, fontsize = 12)
        machine_colors = ["linen", "lavender"]
        machine_color = machine_colors[0]
        previous_machine_type = ""
        new_ytick = []
        for ytick in ax.get_yticklabels():
            current_machine_type = ytick.get_text().split(" ")[0]
            new_ytick += [ytick.get_text().split("(")[1].split(")")[0]]
            if (current_machine_type != previous_machine_type):
                new_ytick[-1] = ytick.get_text().replace("(", "").replace(")", "")
                previous_machine_type = current_machine_type
                machine_color = next(item for item in machine_colors if item != machine_color)
            ytick.set_backgroundcolor(machine_color)
        ax.set_yticklabels(new_ytick, fontsize = 12)
        due_dates = [order_due_dates[order]["due_date"] for order in order_due_dates.keys()]
        due_dates_np = np.array(due_dates)
        order_names = [order for order in order_due_dates.keys()]
        order_final_oper = [order_due_dates[order]["final_oper"] for order in order_due_dates.keys()]
        order_final_machine = [order_due_dates[order]["machine"] for order in order_due_dates.keys()]
        due_date_duplicates = np.zeros(len(due_dates_np))
        for iDue_date, due_date in enumerate(due_dates_np):
            duplicate_indices = np.where(due_dates_np == due_date)[0]
            due_date_duplicates[duplicate_indices] += 1
        num_seed = 42
        for iDue_date, due_date in enumerate(due_dates):
            num_duplicates = due_date_duplicates[iDue_date]
            label = order_final_oper[iDue_date].name.split("FG_")[-1]
            ymin = order_final_machine[iDue_date]-2
            ymax = order_final_machine[iDue_date]+2
            order_name = order_names[iDue_date]
            order_color = set_and_get_order_color(order_name, seed=num_seed) 
            # ax.vlines(x=due_date, ymin=ymin, ymax=ymax, colors=order_color, 
                    #   ls='--', lw=3, label=label, alpha=0.6)
            num_seed += 1
        # ncol = math.ceil(len(due_dates)/48)
        # ax.legend(bbox_to_anchor=(1.0, 1), loc='upper left',ncol=ncol)
        if save_plot:
            plot_path = os.path.dirname(os.path.abspath(__file__))+"/Plots/GreedyScheduler/"
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)
            plot_name = datetime.now().strftime("%H_%M_%S")
            plt.savefig(plot_path+plot_name+".png")
            return
        plt.show()

if __name__ == "__main__":
    project_path = os.path.dirname(os.path.dirname(__file__))
    with open(project_path+"/Data/Parsed_Json/batched.json", "r") as f:
            input_json = json.load(f)
    settings_json = {
        "time_step_size" : 10719.675,
        "num_operators" : 30,
        "max_timeline" : 700000,
        "make_span": 10,
        "lead_time": 10,
        "operators": 10,
        "fake_operators": 10,
        "earliness": 0,
        "tardiness": 100,
    }
    gs = GreedyScheduler(input_json,settings_json)
    gs.make_span_scheduling()
    print(gs.get_obj_value())
    gs.plot(save_plot=True)
    print("### OPER ###")
    gs.operators_scheduling()
    print(gs.get_obj_value())
    gs.plot(save_plot=True)