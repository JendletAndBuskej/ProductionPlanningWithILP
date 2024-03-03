import numpy as np
import os, json

def compress_txt_file(raw_data_path, parsed_file_name):

    if parsed_file_name in os.listdir("Data/Raw/"):
        compressed_txt_file = np.loadtxt(raw_data_path+parsed_file_name+".txt")

    else:
        def get_pre_req(compressed_txt_file,pre_reqs):

            for prod in pre_reqs:
                if not prod[0] in compressed_txt_file[:,0]:
                    continue

                idx = np.where(compressed_txt_file[:,0] == prod[0])[0][0]
                compressed_txt_file[idx,1] += [prod[1]]

            return compressed_txt_file


        def get_linetypes_and_times(compressed_txt_file,lines_and_times):
            for prod in lines_and_times:
                idx = np.where(compressed_txt_file[:,0] == prod[0])[0][0]

                compressed_txt_file[idx,2] += [prod[1]]
                compressed_txt_file[idx,3] = float(prod[4])
                compressed_txt_file[idx,4] = float(prod[5])
                compressed_txt_file[idx,5] = float(prod[3])

            
            return compressed_txt_file

        pre_req_path = raw_data_path+"order_preds_tab.txt"
        linetypes_and_time_path = raw_data_path+"order_line_types_tab.txt"
        pre_reqs = np.loadtxt(fname=pre_req_path, skiprows=1, dtype=str,delimiter="\t")
        lines_and_times = np.loadtxt(fname=linetypes_and_time_path, skiprows=1, dtype=str, delimiter="\t")

        compressed_txt_file = np.empty([np.unique(lines_and_times[:,0]).shape[0],6],dtype=object)
        compressed_txt_file[:,0] = np.unique(lines_and_times[:,0])
        
        for prnp in compressed_txt_file:
            prnp[1] = []
            prnp[2] = []

        compressed_txt_file = get_linetypes_and_times(compressed_txt_file,lines_and_times)
        compressed_txt_file = get_pre_req(compressed_txt_file,pre_reqs)
        return compressed_txt_file
        #np.savetxt(raw_data_path+parsed_file_name+".txt", compressed_txt_file,delimiter="\t",fmt="%s")


def add_parent(parsed_json):
    for order in parsed_json:
        for pred in parsed_json[order]["pred_list"]:
            if pred != "PP_6065":
                parsed_json[pred]["parent"] = order
                #print(type(pred))

    return parsed_json



def add_tree_to_child(parsed_json, order, tree_index):
    parsed_json[order]["tree"] = "T"+str(tree_index)
    for pred in parsed_json[order]["pred_list"]:
        if pred == "PP_6065":
            continue
        add_tree_to_child(parsed_json, pred, tree_index)
   
          

def add_tree(parsed_json):
    fgs = [key for key, val in parsed_json.items() if "FG" in key]

    tree_index = 0
    for fg in fgs:
        add_tree_to_child(parsed_json,fg,tree_index)
        tree_index += 1

    return parsed_json



def create_json(compressed):
    #compressed_txt = np.loadtxt(compressed_file_path,dtype=str,delimiter="\t")

    parsed_json = {}
    for order in compressed:

        orderNum = order[0]
        pred_list = order[1]
        
        lineType = order[2]
        startup_time = order[3]
        operation_time = order[4]
        num_operator = order[5]

        parsed_json[orderNum] = {
            "pred_list" : pred_list,
            "linetype" : lineType,
            "startup_time" : float(startup_time),
            "operation_time" : float(operation_time),
            "num_operator" : float(num_operator)}

    parsed_json = add_parent(parsed_json)
    parsed_json = add_tree(parsed_json)

    if not os.path.exists(master_path+"/Data/Parsed_Json"):
        os.makedirs(master_path+"/Data/Parsed_Json")
    
    with open(master_path+"/Data/Parsed_Json/all_orders.json", "w") as f:
        json.dump(parsed_json, f, indent=4)

def convert_machines(data_path_machine_types: str) -> None:
        """Extracts machine data from the machine txt file 'lines_tab.txt'
        and returns a dict with a machine ID as key and machine type as value

        Args:
            data_path_machine_types (str): Path to the data file containing machine data
        """
        txt_to_np = np.genfromtxt(data_path_machine_types, skip_header=1, usecols=1, dtype=int)
        machine_to_np = np.genfromtxt(data_path_machine_types, skip_header=1, usecols=0, dtype=str)
        machine_id = 0
        machines = {}
        for iM, num_machine_type in enumerate(txt_to_np):
            machine_type = machine_to_np[iM]
            for machine in range(num_machine_type):
                machines[machine_id] = machine_type
                machine_id += 1
        
        with open(master_path+"/Data/Parsed_Json/machines.json", "w") as f:
            json.dump(machines, f, indent=4)

if __name__ == "__main__":
    this_path = os.path.dirname(os.path.abspath(__file__))
    master_path = os.path.dirname(os.path.dirname(this_path))
    raw_data_path = master_path + "/Data/Raw/"
    parsed_file_name = "compressed_txt_file"
    compressed_file_path = raw_data_path + parsed_file_name+".txt"

    compressed = compress_txt_file(raw_data_path, parsed_file_name)
    create_json(compressed)
    convert_machines(raw_data_path+"lines_tab.txt")


    


