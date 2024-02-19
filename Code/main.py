import pyomo.environ as pyo
import pandas as pd



model = pyo.AbstractModel()

model.num_machines = pyo.Param(within=pyo.NonNegativeIntegers)
model.num_operations = pyo.Param(within=pyo.NonNegativeIntegers)
model.num_time_indices = pyo.Param(within=pyo.NonNegativeIntegers)

model.machines = pyo.RangeSet(1, model.num_machines)
model.operations = pyo.RangeSet(1, model.num_operations)
model.time_indices = pyo.RangeSet(1, model.num_time_indices)

model.operation_valid_machine_types = pyo.Param(model.operations, model.machines)
model.operation_time = pyo.Param(model.operations)

model.assigned = pyo.Var(model.machines, model.operations, model.time_indices, domain=pyo.Binary)

def objective_rule(model):
    return sum(t * model.assigned[m, o, t] for m in model.machines for o in model.operations for t in model.time_indices)  

model.objective = pyo.Objective(rule=objective_rule)

# placed at once only
def no_duplicate_const(model, operation):
    return sum(model.assigned[m,operation,t] for m in model.machines for t in model.time_indices) == 1

model.no_duplicate = pyo.Constraint(model.operations, rule=no_duplicate_const)

def machine_type_const(model, machine, operation):
    return sum(model.assigned[machine,operation,t] for t in model.time_indices) <= model.operation_valid_machine_types[operation, machine]

model.machine_type_const = pyo.Constraint(model.machines, model.operations, rule=machine_type_const)




"""

instance = model.create_instance('Data/Pyomo/pyo_data.dat')
opt = SolverFactory('glpk')
result = opt.solve(instance)

df = pd.DataFrame.from_dict(model.assigned.extract_values(), orient='index', columns=[str(model.assigned)])
print(df)
"""