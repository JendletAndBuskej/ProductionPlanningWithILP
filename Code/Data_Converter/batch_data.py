import numpy as np
import os, json

class BatchData:
    def __init__(self, batch_size = 100):
        self.batch_size = batch_size
        master_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.data_path = master_path + "/Data/"
        self.valid_FG_lines = np.array(["line_01","line_02","line_04","line_08"], dtype=str)
        order_list_path = self.data_path+"Raw/simulation_orderlist.txt" 
        order_list = np.loadtxt(order_list_path,dtype=str,delimiter="\t",skiprows=1)
        unique_orders = np.unique(order_list[:,0])
        all_FG_orders = np.array(unique_orders[np.core.defchararray.find(unique_orders.astype(str), 'FG') != -1])
        self.selected_orders = np.random.choice(all_FG_orders, size=self.batch_size, replace=False)
        self.queue = self.selected_orders
        self.batch_order_list = {}
        with open(self.data_path+"/Parsed_Json/all_orders.json", "r") as f:
            self.all_orders_json = json.load(f)      
        with open(self.data_path+"/Parsed_Json/machines.json", "r") as f:
            self.machines = json.load(f)  
        self.machines = {int(key): value for key, value in self.machines.items()}

    def get_order_preds(self,order,randomize_FG):
        all_orders = self.all_orders_json[order]
        pred_list = all_orders["pred_list"]
        if isinstance(pred_list, str):
            pred_list = [pred_list]
        linetype = all_orders["linetype"]
        startup_time = all_orders["startup_time"]
        operation_time = all_orders["operation_time"]
        num_operator = all_orders["num_operator"]
        tree = all_orders["tree"]
        parent = ""
        
        if "parent" in all_orders:
            parent = all_orders["parent"]
        else:
            if randomize_FG:
                linetype = np.random.choice(self.valid_FG_lines)
        
        self.batch_order_list[order] = {
            "pred_list" : pred_list,
            "linetype" : linetype,
            "startup_time" : startup_time,
            "operation_time" : operation_time,
            "num_operator" : num_operator,
            "parent" : parent,
            "tree" : tree }
        
        self.add_preds_to_queue_list(np.array(pred_list))    

    def add_preds_to_queue_list(self,pred_list):
        if pred_list.shape == ():
            self.queue = np.append(self.queue,pred_list)
            return ()

        for pred in pred_list:
            self.queue = np.append(self.queue,pred)
            
    def get_batch(self, randomize_FG = False):    
        while True:
            # breaks when all orders have been found
            if self.queue.shape[0] == 0:
                break
            order = self.queue[0]
            self.queue = np.delete(self.queue,0)
            self.get_order_preds(order, randomize_FG)
        return (self.batch_order_list)

    def get_machines(self):
        return (self.machines)

    def generate_due_dates(self, distribution: str, time_interval: list[int, int]):
        total_time = sum(self.batch_order_list[oper]["operation_time"] 
                         + self.batch_order_list[oper]["startup_time"]
                         for oper in self.batch_order_list)
        num_orders_per_due_date_unit = 2
        due_dates_size = total_time/(num_orders_per_due_date_unit*len(self.selected_orders))
        due_dates = np.arange(1, len(self.selected_orders)/num_orders_per_due_date_unit + 1)
        due_dates = due_dates_size * np.repeat(due_dates, num_orders_per_due_date_unit)  
        for iOrd, order in enumerate(self.selected_orders):
            self.batch_order_list[order]["due_date"] = due_dates[iOrd]
        return (self.batch_order_list)
    
    def save_as_json(self, data, path):
        with open(self.data_path+path, "w") as f:
            json.dump(data, f, indent=4)

if __name__ == "__main__":
    batch_data = BatchData(batch_size=1)
    batched_data = batch_data.get_batch()
    batch_data.generate_due_dates("uniform", [0,10])
    # batch_data.save_as_json(batched_data, "/Parsed_Json/batched.json")
    print(batch_data.get_machines())



