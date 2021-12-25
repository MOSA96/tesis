from deap import tools, creator, base
from utils.utils import test, function, ind
import matplotlib.pyplot as plt
import numpy as np
import utils.elitism as elitism
import seaborn as sns

#Params
SIZE = 1
BOUND_LOW, BOUND_UP =  -2,8
POPULATION_SIZE = 100
P_CROSSOVER = 1
P_MUTATION = 1 
MAX_GENERATIONS = 50
HALL_OF_FAME_SIZE = 20
CROWDING_FACTOR = 20.0 


toolbox = base.Toolbox()
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox.register("randomInd", ind, BOUND_LOW, BOUND_UP)
toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.randomInd)
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)
toolbox.register("evaluate", function)
toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("mate", tools.cxSimulatedBinaryBounded,low=BOUND_LOW, up=BOUND_UP, eta=CROWDING_FACTOR)
toolbox.register("mutate", tools.mutPolynomialBounded,low=BOUND_LOW, up=BOUND_UP, eta=CROWDING_FACTOR, indpb=1.0)

def main():
    
    # create initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    # define the hall-of-fame object:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # perform the Genetic Algorithm flow with elitism:
    population, logbook, history = elitism.eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=False,
                                              plot = True)

    # print info for best solution found:
    best = hof.items[0]
    best_individual =  np.round(best[0],2)
    best_fitness = np.round(best.fitness.values[0],2)
    print("-- Best Individual = ", best_individual)
    print("-- Best Fitness = ", best_fitness)

    
    # extract statistics:
    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")

    # plot statistics:
    sns.set_style("whitegrid")
    plt.figure(1)
    plt.plot(minFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Min / Average Fitness')
    plt.title('Min and Average fitness over Generations')

    plt.figure(2)
    x=np.linspace(BOUND_LOW, BOUND_UP, 250)
    plt.plot(x, test(x))
    plt.axvline(x=best_individual, ymin=best_fitness, ymax=1, linestyle="--", color="r")
    plt.title("Best Individual")

    plt.show()

if __name__ == "__main__":
    main()