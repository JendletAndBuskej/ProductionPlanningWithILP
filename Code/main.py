import pyomo.environ as pyo

model = pyo.AbstractModel()

model.num_M = pyo.Param(within=pyo.NonNegativeIntegers)
model.num_O = pyo.Param(within=pyo.NonNegativeIntegers)
model.num_T = pyo.Param(within=pyo.NonNegativeIntegers)

model.M = pyo.RangeSet(1, model.num_M)
model.O = pyo.RangeSet(1, model.num_O)
model.T = pyo.RangeSet(1, model.num_T)

model.operation_valid_machine_types = pyo.Param(model.O, model.M)
model.operation_time = pyo.Param(model.O)

model.assigned = pyo.Var(model.M, model.O, model.T, domain=pyo.Binary)

model.max = max(model.assigned[m, o, t] * t for m in model.M for o in model.O for t in model.T)


def obj_function(model):
    return pyo.Expression(expr=max(model.assigned[m,o,t] * t for m in model.M for o in model.O for t in model.T))
    #return pyo.max(model.assigned[m,o,t] * t for m in model.M for o in model.O for t in model.T)

model.obj = pyo.Objective(rule=obj_function)


# placed at once only
def no_duplicate_const(model, operation):
    return sum(model.assigned(m,operation,t) for m in model.M for t in model.T) <= 1 



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