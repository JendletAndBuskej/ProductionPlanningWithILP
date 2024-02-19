import pyomo.environ as pyo

model = pyo.AbstractModel()

model.num_machines = pyo.Param(within=pyo.NonNegativeIntegers)
model.num_operations = pyo.Param(within=pyo.NonNegativeIntegers)
model.num_time_indices = pyo.Param(within=pyo.NonNegativeIntegers)

model.machines = pyo.RangeSet(1, model.num_machines)
model.operations = pyo.RangeSet(1, model.num_operations)
model.time_indices = pyo.RangeSet(1, model.num_time_indices)

model.operation_valid_machine_types = pyo.Param(model.operations, model.machines)
model.operation_time = pyo.Param(model.operations)

# model.assigned = pyo.Var(model.machines, model.operations, model.time_indices, within=pyo.Binary)
model.assigned = pyo.Var(model.machines, model.operations, model.time_indices, domain=pyo.Binary)


# def obj_function(model):
    # return pyo.Expression(expr=max(model.assigned[m,o,t] * t for m in model.machines for o in model.operations for t in model.time_indices))
    # return pyo.max(model.assigned[m,o,t] * t for m in model.machines for o in model.operations for t in model.time_indices)
# 
# model.obj = pyo.Objective(rule=obj_function)

def objective_rule(model):
    #return sum(model.time_indices[t] * model.assigned[m, o, t] for (m, o, t) in model.assigned) 
    return sum(t * model.assigned[m, o, t] for m in model.machines for o in model.operations for t in model.time_indices)  

model.objective = pyo.Objective(rule=objective_rule)

# placed at once only
def no_duplicate_const(model, operation):
    return sum(model.assigned[m,operation,t] for m in model.machines for t in model.time_indices) == 1

model.no_duplicate = pyo.Constraint(model.operations, rule=no_duplicate_const)


"""
model.m = pyo.Param(within=pyo.NonNegativeIntegers)
model.n = pyo.Param(within=pyo.NonNegativeIntegers)

model.I = pyo.RangeSet(1, model.m)
model.J = pyo.RangeSet(1, model.n)

model.a = pyo.Param(model.I, model.J)
model.b = pyo.Param(model.I)
model.c = pyo.Param(model.J)

# the next line declares a variable indexed by the set J
model.x = pyo.Var(model.J, domain=pyo.NonNegativeReals)

def obj_expression(m):
    return pyo.summation(m.c, m.x)

model.OBJ = pyo.Objective(rule=obj_expression)

def ax_constraint_rule(m, i):
    # return the expression for the constraint for i
    return sum(m.a[i,j] * m.x[j] for j in m.J) >= m.b[i]

# the next line creates one constraint for each member of the set model.I
model.AxbConstraint = pyo.Constraint(model.I, rule=ax_constraint_rule)

"""