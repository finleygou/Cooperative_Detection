import random
from deap import base, creator, tools, algorithms
import numpy as np

# 定义目标函数
def objective(individual):
    # 目标函数：f(x) = sum(x[i] * (x[i] - 1)**2) 对于 0-1 变量
    return -np.sum(np.array(individual) * (np.array(individual) - 1)**2),  # 负值因为我们要最大化

# 定义约束检查函数
def is_valid(individual, max_sum=1):
    return np.sum(individual) <= max_sum

# 初始化种群
def init_individual(n):
    return [random.randint(0, 1) for _ in range(n)]

def main():
    # 问题规模
    n = 5
    max_sum = 1
    
    # 创建遗传算法的环境
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, lambda: init_individual(n))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", objective)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # 创建初始种群
    population = toolbox.population(n=100)
    
    # 遗传算法参数
    ngen = 50
    cxpb = 0.5
    mutpb = 0.2
    
    for gen in range(ngen):
        # 评估种群
        fitnesses = map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # 选择
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        # 交叉
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        # 变异
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # 更新种群
        population[:] = offspring
        
        # 打印当前代的信息
        fits = [ind.fitness.values[0] for ind in population]
        length = len(population)
        mean = np.mean(fits)
        best = np.max(fits)
        print(f"Generation {gen}: Max fitness = {best}, Avg fitness = {mean}")
    
    # 找到最优解
    best_individual = tools.selBest(population, 1)[0]
    print(f"Best individual: {best_individual}")
    print(f"Best fitness: {best_individual.fitness.values[0]}")

if __name__ == "__main__":
    main()
