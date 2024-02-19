import pyomo.environ as pyo

model = pyo.AbstractModel()

### Sets, Params and Variables 
model.num_machines = pyo.Param(within=pyo.NonNegativeIntegers)
model.num_operations = pyo.Param(within=pyo.NonNegativeIntegers)
model.num_time_indices = pyo.Param(within=pyo.NonNegativeIntegers)

model.machines = pyo.RangeSet(1, model.num_machines)
model.operations = pyo.RangeSet(1, model.num_operations)
model.time_indices = pyo.RangeSet(1, model.num_time_indices)

model.operation_valid_machine_types = pyo.Param(model.operations, model.machines)
model.operation_time = pyo.Param(model.operations)

model.assigned = pyo.Var(model.machines, model.operations, model.time_indices, domain=pyo.Binary)


### Objective
def objective_function(model):
    return sum(t * model.assigned[m, o, t] for m in model.machines for o in model.operations for t in model.time_indices)  
model.objective = pyo.Objective(rule=objective_function)


### Constraints
# placed once only
def duplicate_const(model, operation):
    return sum(model.assigned[m,operation,t] for m in model.machines for t in model.time_indices) == 1
model.duplicate_const = pyo.Constraint(model.operations, rule=duplicate_const)

# executed on a valid machine
def machine_type_const(model, machine, operation):
    return sum(model.assigned[machine, operation, t] for t in model.time_indices) <= model.operation_valid_machine_types[operation,machine]
model.machine_type_const = pyo.Constraint(model.machines, model.operations, rule=machine_type_const)

# no overlap
def overlap_const(model, machine, operation, time_index):
    return sum(model.assigned[machine, o, t] for o in model.operations for t in (range(min(time_index,3), min(model.operation_time[operation] + time_index, 4)))) <= 1
model.overlap_const = pyo.Constraint(model.machines, model.operations, model.time_indices, rule=overlap_const)