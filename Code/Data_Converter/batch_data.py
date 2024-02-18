import numpy as np
import os, json

#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class Batch_Data:

    def __init__(self, batch_size = 100):
        self.batch_size = batch_size
        master_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.data_path = master_path + "/Data/"
        self.valid_FG_lines = np.array(["line_01","line_02","line_04","line_08"], dtype=str)

        order_list_path = self.data_path+"Raw/simulation_orderlist.txt" 
        order_list = np.loadtxt(order_list_path,dtype=str,delimiter="\t",skiprows=1)
        unique_orders = np.unique(order_list[:,0])

        all_FG_orders = np.array(unique_orders[np.core.defchararray.find(unique_orders.astype(str), 'FG') != -1])
        batch_FG_orders = np.random.choice(all_FG_orders, size=self.batch_size, replace=False)
        #batch_order_list = np.empty([0,4],dtype=object)

        self.queue = batch_FG_orders
        self.batch_order_list = {}
        #self.taillard

        #self.all_orders_json = np.loadtxt(self.data_path+"/raw/all_orders_json.txt",dtype=str,delimiter="\t")      
        with open(self.data_path+"/Parsed_Json/all_orders.json", "r") as f:
            self.all_orders_json = json.load(f)      


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

        """
        for iOrd, all_orders in enumerate(self.all_orders_json):
            if all_orders[0] == order:

                #self.batch_order_list = np.r_[self.batch_order_list,[all_orders]]
                linetype = all_orders[2][2:-2]
                if all_orders[1][1:-1] == "":
                    pred_list = np.empty([0,0])
                else:
                    pred_list = np.array(ast.literal_eval(all_orders[1][1:-1]))
                    if pred_list.shape == ():
                        pred_list = np.array([pred_list])

                self.batch_order_list[all_orders[0]] = {"pred_list" : pred_list.tolist(),
                                                     "linetype" : linetype,
                                                     "time" : float(all_orders[3])}
                
                self.add_preds_to_queue_list(pred_list)
                break
        """

    

    def add_preds_to_queue_list(self,pred_list):
        """
        if len(pred_list) == 0:
            return
        """

        if pred_list.shape == ():
            self.queue = np.append(self.queue,pred_list)
            return

        for pred in pred_list:
            self.queue = np.append(self.queue,pred)
            


    def get_batch(self, randomize_FG = False):    
        while True:
            # breaks when all orders have been found
            if self.queue.shape[0] == 0:
                break
            
            # "pops" queue
            order = self.queue[0]
            self.queue = np.delete(self.queue,0)

            # gets preds of order poped
            self.get_order_preds(order, randomize_FG)


        #batchPath = self.data_path+"/batched/infoMatrixBatched.txt"
        #np.savetxt(batchPath, self.batch_order_list ,delimiter="\t",fmt='%s')
        return self.batch_order_list
    
    

    def save_as_json(self):
        with open(self.data_path+"/Parsed_Json/batched.json", "w") as f:
            json.dump(self.batch_order_list, f, indent=4)


batchData = Batch_Data(batch_size=8)
batchedData = batchData.get_batch()
#print(batchedData)
batchData.save_as_json()



