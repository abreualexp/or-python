from mip import *



# Instância

numProduto = 2
numPeriodo = 3

conjuntoProduto = range(1, numProduto+1) # {1, ..., numProduto}
conjuntoPeriodo = range(0, numPeriodo+1) # {1, ..., numPeriodo}

demanda = {
  1: {1: 900, 2: 1800, 3: 1800},
  2: {1: 400, 2: 600, 3: 800},
}

custoProducao = {
  1: {1: 1.0, 2: 1.5, 3: 2.0},
  2: {1: 0.5, 2: 0.5, 3: 0.9}
}

custoEstoque = {
  1: {1: 0.5, 2: 0.25, 3: 0},
  2: {1: 0.25, 2: 0.25, 3: 0}
}

custoPreparacao = {
  1: {1: 2.0, 2: 4.0, 3: 4.0},
  2: {1: 8.0, 2: 8.0, 3: 8.0}
}

tempoProcessamento = {
  1: 0.1,
  2: 0.08
}

tempoPreparacao = {
  1: 12,
  2: 8
}

capacidade = {1: 240, 2: 320, 3: 200}

bigM = 10000



model_lotes = Model()

# Variables
# x[i, t]: amount produced of product i in period t
x = {(i, t): model_lotes.add_var(var_type=CONTINUOUS, lb=0, name=f"x_{i}_{t}") for i in conjuntoProduto for t in conjuntoPeriodo}

# s[i, t]: stock of product i at the end of period t
I = {(i, t): model_lotes.add_var(var_type=CONTINUOUS, lb=0, name=f"s_{i}_{t}") for i in conjuntoProduto for t in conjuntoPeriodo}

# y[i, t]: binary variable, 1 if product i is produced in period t
# y = {(i, t): model_lotes.add_var(var_type=BINARY, name=f"y_{i}_{t}") for i in conjuntoProduto for t in periods}

# Objective Function: Minimize (Production + Inventory + Setup costs)
model_lotes.objective = minimize(
    xsum(custoProducao[i][t] * x[i, t] for i in conjuntoProduto for t in conjuntoPeriodo[1:]) +
    xsum(custoEstoque[i][t] * I[i, t] for i in conjuntoProduto for t in conjuntoPeriodo[1:])
)

for i in conjuntoProduto:
    model_lotes += I[i, 0] == 0

for i in conjuntoProduto:
    for t in conjuntoPeriodo[1:]:
        model_lotes += I[i,t] == I[i,t-1] + x[i,t] - demanda[i][t], f"Estoque_{i}_{t}"

for t in conjuntoPeriodo:
  if t > 0:
    model_lotes += ( xsum(tempoProcessamento[i] * x[i,t] for i in conjuntoProduto) <= capacidade[t] )

status = model_lotes.optimize()

print(f"Status: {status}")
if status == OptimizationStatus.OPTIMAL:
    print(f"Custo Mínimo Total: {model_lotes.objective_value:.2f}\n")

    for t in conjuntoPeriodo:
        print(f"--- Período {t} ---")
        for i in conjuntoProduto:
            if x[i, t].x > 0.001:
                print(f"Produto {i}: Produzido={x[i, t].x}, Estoque Final={I[i, t].x}")
            else:
                print(f"Produto {i}: Sem produção, Estoque Final={I[i, t].x}")