import numpy as np, json, os, random, math
from environment import Environment
import matplotlib.pyplot as plt 
from classes import Machine, Operation
from datetime import datetime

class GreedyScheduler:
    def __init__(self, input_json: dict) -> None:
        env = Environment(input_json)
        self.operations = env.schedule[1,:]
        self.orders = env.orders
        self.operations_precedence_time = np.zeros([1,self.operations.shape[0]],dtype=float)
        self.machines = self.instanciate_machines(env.machines)
        self.oper_num_childs = self.instanciate_num_childs()
        self.selection_mask = self.get_selection_mask()
        self.due_date_prio = self.instanciate_due_date_prio()
        
    def instanciate_machines(self, machine_info):
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
    
    def make_span_scheduling(self):
        # i = 0
        while True:
            # i += 1
            # if (i >= 25): break
            if not (np.any(self.selection_mask)): break
            prio_selections = self.due_date_prio*self.selection_mask
            print("\n",self.selection_mask)
            print(prio_selections)
            valid_operations = np.where(prio_selections == np.amax(prio_selections))[1]
            valid_operations_precedence_time = self.operations_precedence_time[0,valid_operations]
            selected_operation_index = valid_operations[np.argmin(valid_operations_precedence_time)]
            selected_operation = self.operations[selected_operation_index]
            selected_operation_mIDS = selected_operation.valid_machine_ids
            selected_operation_precedence_time = self.operations_precedence_time[0,selected_operation_index]
            print("selected_operation_index: ",selected_operation_index)
            print("selected_operation: ",selected_operation.name)
            print("selected_operation_mIDS: ",selected_operation_mIDS)
            selected_machine_id = self.get_earliest_machine_id(selected_operation_mIDS, selected_operation_precedence_time)
            print("selected_machine_id: ",selected_machine_id)
            selected_machine = self.machines[selected_machine_id]
            print("selected_machine: ",selected_machine)
            operation_start_time = selected_machine.max_time
            operation_end_time = operation_start_time + selected_operation.execution_time
            print("operation_start_time: ",operation_start_time)
            selected_machine.add_operation(selected_operation, operation_start_time)
            self.update_selection_mask(selected_operation)
            self.update_precedence_times(selected_operation, operation_end_time)            

    # def plot(self, real_size = True, save_plot = False, hide_text = False) -> None:
    def plot(self, save_plot = False) -> None:
        """Plots the scheme

        Args:
            real_size (bool, optional): Weather or not the real length of each operation should be
                                        displayed or the length of time_step_size. Defaults to True.
            save_plot (bool, optional): Weather or not the plot should be saved or shown. 
                                        Defaults to Show, False.
            hide_text (bool, optional): Weather or not the plot should contain text or not. 
                                        Defaults to contain text, False.
            t_interval (list[int], optional): The time interval to plot within. Default is the
                                              entire interval.
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
            machine_ids = list(self.machines.keys())
            machine_types = list(self.machines.values())
            machine_ticks = [f"{mtype} (ID: {mID})" for mID, mtype in zip(machine_ids, machine_types)]
            return (machine_ticks)
        
        def get_order_and_due_date() -> dict:
            orders_due_dates = {}
            for iOper, oper in enumerate(self.schedule[1,:]):
                order = oper.order
                if not (order in orders_due_dates):
                    finished_time = self.schedule[2,iOper] + math.ceil(oper.execution_time/self.time_step_size)
                    orders_due_dates[order.name] = {"due_date" : order.due_date,
                                                    "finished_time" : finished_time,
                                                    "final_oper" : oper}
                oper_finished_time = self.schedule[2,iOper] + math.ceil(oper.execution_time/self.time_step_size)
                if (oper_finished_time >= orders_due_dates[order.name]["finished_time"]):
                    orders_due_dates[order.name] = {"due_date" : order.due_date,
                                                    "finished_time" : oper_finished_time,
                                                    "final_oper" : oper}
            return (orders_due_dates)
        
        plt.clf(), plt.cla(), plt.close()
        # order_due_dates = get_order_and_due_date()
        fig, ax = plt.subplots(figsize=(16,9))
        order_tree_dict = {}
        num_seed = 42
        for iMach, machine in enumerate(self.machines):
            machine_id = machine.id
            m_type = machine.m_type
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
        # ax.set_xlim([0, xlim_upper])
        # machine_ticks = get_machine_ticks()
        # ax.set_yticks(np.arange(len(self.machines)))
        # ax.set_yticklabels(machine_ticks, fontsize = 12)
        # machine_colors = ["linen", "lavender"]
        # machine_color = machine_colors[0]
        # previous_machine_type = ""
        # new_ytick = []
        # for ytick in ax.get_yticklabels():
        #     current_machine_type = ytick.get_text().split(" ")[0]
        #     new_ytick += [ytick.get_text().split("(")[1].split(")")[0]]
        #     if (current_machine_type != previous_machine_type):
        #         new_ytick[-1] = ytick.get_text().replace("(", "").replace(")", "")
        #         previous_machine_type = current_machine_type
        #         machine_color = next(item for item in machine_colors if item != machine_color)
        #     ytick.set_backgroundcolor(machine_color)
        # ax.set_yticklabels(new_ytick, fontsize = 12)
        # ax.xaxis.grid(True, alpha=0.5)
        # due_dates = [order_due_dates[order]["due_date"] for order in order_due_dates.keys()]
        # due_dates_np = np.array(due_dates)
        # order_names = [order for order in order_due_dates.keys()]
        # order_final_oper = [order_due_dates[order]["final_oper"] for order in order_due_dates.keys()]
        # due_date_duplicates = np.zeros(len(due_dates_np))
        # for iDue_date, due_date in enumerate(due_dates_np):
        #     duplicate_indices = np.where(due_dates_np == due_date)[0]
        #     due_date_duplicates[duplicate_indices] += 1
        # num_seed = 42
        # for iDue_date, due_date in enumerate(due_dates):
        #     num_duplicates = due_date_duplicates[iDue_date]
        #     label = order_final_oper[iDue_date].name.split("FG_")[-1]
        #     final_oper_index = np.where(np.isin(self.schedule[1,:], order_final_oper[iDue_date]))[0]
        #     final_oper_machine = self.schedule[0,final_oper_index]
        #     ymin = final_oper_machine-2
        #     ymax = final_oper_machine+2
        #     order_name = order_names[iDue_date]
        #     order_color = set_and_get_order_color(order_name, seed=num_seed)
        #     random_offset = random.randint(-int(self.time_line[-1]), int(self.time_line[-1]))
        #     random_offset /= 100
        #     # ax.vlines(x=due_date+random_offset, ymin=ymin, ymax=ymax, colors=order_color, 
        #     ax.vlines(x=due_date, ymin=ymin, ymax=ymax, colors=order_color, 
        #               ls='--', lw=3, label=label, alpha=0.6)
        #     num_seed += 1
        # ax.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
        if save_plot:
            plot_path = os.path.dirname(os.path.abspath(__file__))+"/Plots/"
            plot_name = datetime.now().strftime("%H_%M_%S")
            plt.savefig(plot_path+plot_name+".png")
            return
        plt.show()

if __name__ == "__main__":
    project_path = os.path.dirname(os.path.dirname(__file__))
    with open(project_path+"/Data/Parsed_Json/batched.json", "r") as f:
            input_json = json.load(f)
    gs = GreedyScheduler(input_json)
    gs.make_span_scheduling()
    gs.plot()
    # gs.machines[0].add_operation(gs.operations[0],start_time=0.0)
    # gs.machines[1].add_operation(gs.operations[1],start_time=22140)
    # gs.machines[2].add_operation(gs.operations[2],start_time=22140)
    # gs.machines[3].add_operation(gs.operations[3],start_time=22140)
    # gs.machines[4].add_operation(gs.operations[3],start_time=22140)
    # gs.machines[5].add_operation(gs.operations[3],start_time=22140)
    # gs.machines[6].add_operation(gs.operations[3],start_time=22140)
    # gs.machines[7].add_operation(gs.operations[3],start_time=22140)
    # gs.machines[8].add_operation(gs.operations[3],start_time=22140)
    # gs.get_earliest_machine_id(["line_12", "line_11", "line_10"])
    # for machine in gs.machines:
    #     machine.print_info()