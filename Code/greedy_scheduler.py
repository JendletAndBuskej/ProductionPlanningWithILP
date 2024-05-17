import numpy as np, json, os
from environment import Environment
from classes import Machine, Operation

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
            return
        parent_index = np.where(np.isin(self.oper_num_childs[0,:], parent))
        self.oper_num_childs[1,parent_index] -= 1
        self.selection_mask = self.get_selection_mask()
    
    def update_precedence_times(self, selected_operation: Operation, end_time: float):
        parent = selected_operation.parent
        if not (parent):
            return
        parent_index = np.where(np.isin(self.oper_num_childs[0,:], parent))
        
    
    def get_earliest_machine_id(self, machine_types: list[str]) -> int:
        def id_earliest_machine_mtype(machine_type: str) -> int:
            machine_ids = [machine.id for machine in self.machines if machine.m_type == machine_type]
            if (not machine_ids):
                raise ValueError("Machine type passed doesn't exist.")
            earliest_time = self.machines[machine_ids[0]].max_time
            earliest_machine_id = machine_ids[0]
            for machine_id in machine_ids:
                machine_max_time = self.machines[machine_id].max_time
                if (machine_max_time < earliest_time):
                    earliest_time = machine_max_time
                    earliest_machine_id = machine_id
            return (earliest_machine_id)
        
        id_earliest_per_mtype = []
        for machine_type in machine_types:
            id_earliest_per_mtype += [id_earliest_machine_mtype(machine_type)]
        earliest_time = self.machines[id_earliest_per_mtype[0]].max_time
        earliest_machine_id = id_earliest_per_mtype[0]
        for machine_id in id_earliest_per_mtype:
            machine_max_time = self.machines[machine_id].max_time
            if (machine_max_time < earliest_time):
                earliest_time = machine_max_time
                earliest_machine_id = machine_id
        return (earliest_machine_id)

    def make_span_scheduling(self):
        print(self.selection_mask)
        while True:
            if not (np.any(self.selection_mask)): break
            prio_selections = self.due_date_prio*self.selection_mask
            print(prio_selections)
            selected_operation = np.argmax(prio_selections)
            print(selected_operation)
            
                
if __name__ == "__main__":
    project_path = os.path.dirname(os.path.dirname(__file__))
    with open(project_path+"/Data/Parsed_Json/batched.json", "r") as f:
            input_json = json.load(f)
    gs = GreedyScheduler(input_json)
    gs.make_span_scheduling()
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