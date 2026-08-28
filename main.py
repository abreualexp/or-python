from mip import *

class Instance:
    def __init__(self, conjuntoProduto, conjuntoPeriodo, demanda, custoProducao, custoEstoque, custoPreparacao, tempoProcessamento, tempoPreparacao, capacidade, bigM):
        self.conjuntoProduto = conjuntoProduto
        self.conjuntoPeriodo = conjuntoPeriodo
        self.demanda = demanda
        self.custoProducao = custoProducao
        self.custoEstoque = custoEstoque
        self.custoPreparacao = custoPreparacao
        self.tempoProcessamento = tempoProcessamento
        self.tempoPreparacao = tempoPreparacao
        self.capacidade = capacidade
        self.bigM = bigM


def readInstance():
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

    return Instance(
        conjuntoProduto, 
        conjuntoPeriodo, 
        demanda, 
        custoProducao, 
        custoEstoque, 
        custoPreparacao, 
        tempoProcessamento, 
        tempoPreparacao, 
        capacidade, 
        bigM)


def solveModel(instance):
    # model = Model(solver_name="HiGHS")
    model = Model()

    x = {(i, t): model.add_var(var_type=CONTINUOUS, lb=0, name=f"x_{i}_{t}") for i in instance.conjuntoProduto for t in instance.conjuntoPeriodo}
    I = {(i, t): model.add_var(var_type=CONTINUOUS, lb=0, name=f"s_{i}_{t}") for i in instance.conjuntoProduto for t in instance.conjuntoPeriodo}
    # y = {(i, t): model.add_var(var_type=BINARY, name=f"y_{i}_{t}") for i in instance.conjuntoProduto for t in instance.conjuntoPeriodo}

    model.objective = minimize(
        xsum(instance.custoProducao[i][t] * x[i, t] for i in instance.conjuntoProduto for t in instance.conjuntoPeriodo[1:]) +
        xsum(instance.custoEstoque[i][t] * I[i, t] for i in instance.conjuntoProduto for t in instance.conjuntoPeriodo[1:])
    )

    for i in instance.conjuntoProduto:
        model += I[i, 0] == 0

    for i in instance.conjuntoProduto:
        for t in instance.conjuntoPeriodo[1:]:
            model += I[i,t] == I[i,t-1] + x[i,t] - instance.demanda[i][t], f"Estoque_{i}_{t}"

    for t in instance.conjuntoPeriodo:
        if t > 0:
            model += ( xsum(instance.tempoProcessamento[i] * x[i,t] for i in instance.conjuntoProduto) <= instance.capacidade[t] )

    model.optimize()

    xVal = {(i, t): x[i, t].x for i in instance.conjuntoProduto for t in instance.conjuntoPeriodo}
    IVal = {(i, t): I[i, t].x for i in instance.conjuntoProduto for t in instance.conjuntoPeriodo}

    return model, xVal, IVal

def printSolution(instance, model, x, I):
    print(f"Status: {model.status}")
    if model.status == OptimizationStatus.OPTIMAL:
        print(f"Custo Mínimo Total: {model.objective_value:.2f}\n")

        for t in instance.conjuntoPeriodo:
            print(f"--- Período {t} ---")
            for i in instance.conjuntoProduto:
                if x[i, t] > 0.001:
                    print(f"Produto {i}: Produzido={x[i, t]}, Estoque Final={I[i, t]}")
                else:
                    print(f"Produto {i}: Sem produção, Estoque Final={I[i, t]}")


def main():
    instance = readInstance()
    model, xVal, IVal = solveModel(instance)
    printSolution(instance, model, xVal, IVal)


if __name__ == "__main__":
    main()
