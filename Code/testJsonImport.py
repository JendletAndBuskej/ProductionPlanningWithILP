import string
import numpy as np
import matplotlib.pyplot as plt
import os, json


with open("Data/Parsed_Json/batched.json", "r") as f:
    orders_json = json.load(f)

print(np.array())