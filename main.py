import json
import random
import matplotlib.pyplot as plt
import numpy
from deap import base, creator, tools, algorithms

CONFIG_FILE_NAME = "config.json"
ALLOWED_FUNCTIONS = ['x', '*', '+', '-', '.', ' ']

def load_config():
    def _validate_config(config_file):
        config_keys = ["num_generations", "population_size", "crossover_rate", "mutation_rate", "lower_bound", "upper_bound", "function"]
        keys_counter = 0
        for key in config_file.keys():
            if key not in config_keys:
                raise ValueError(f"Invalid key in config file: {key}")
            keys_counter += 1
        if keys_counter != len(config_keys):
            raise ValueError("Missing key in config file")
        for key in config_file.keys():
            if key not in config_file:
                raise ValueError(f"Missing key in config file: {key}")
        for key, value in config_file.items():
            if key == "function":
                if not isinstance(value, str):
                    raise ValueError("Function string must be a string")
            elif key == "num_generations" or key == "population_size" or key == "lower_bound" or key == "upper_bound":
                if not isinstance(value, int):
                    raise ValueError(f"{key} must be an integer")
            elif key == "crossover_rate" or key == "mutation_rate":
                if not isinstance(value, float):
                    raise ValueError(f"{key} must be a float")
            else:
                raise ValueError(f"Invalid key in config file: {key}")  
    with open(CONFIG_FILE_NAME) as config_file:
        json_config = json.load(config_file)
        _validate_config(json_config)
        return json_config

def validate_function_string(function_string):
    for char in function_string:
        if char not in ALLOWED_FUNCTIONS and not char.isdigit():
            raise ValueError("Invalid character in function string:" + char)

def evaluate(individual):
    x_value = individual[0]
    return fitness_function(x_value), 

def setup_deap():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

def non_proportional_roulette(population, k):
    chosen = []
    for _ in range(k):
        chosen.append(random.choice(population))
    return chosen

def configure_toolbox(mutation_probability):
    _toolbox = base.Toolbox()
    _toolbox.register("attr_generator", random.randint, lower_bound, upper_bound)
    _toolbox.register("individual", tools.initRepeat, creator.Individual, _toolbox.attr_generator, 2)
    _toolbox.register("population", tools.initRepeat, list, _toolbox.individual)
    _toolbox.register("evaluate", evaluate)
    _toolbox.register("mate", tools.cxTwoPoint)
    _toolbox.register("mutate", tools.mutUniformInt, low=lower_bound, up=upper_bound, indpb=mutation_probability)
    _toolbox.register("select", non_proportional_roulette)
    return _toolbox
    return toolbox

def display_fitness_function(start, end, max_x, max_fitness_value):
    x_range = range(start, end)
    y_values = [fitness_function(x) for x in x_range]

    plt.plot(x_range, y_values, label=f"Fitness function: {function_str}")
    plt.scatter(max_x, max_fitness_value, color="red", zorder=5, label=f"Max result: f({max_x}) = {max_fitness_value:.2f}")
    plt.annotate(f"Max result\nf({max_x}) = {max_fitness_value:.2f}", 
                 (max_x, max_fitness_value),
                 xytext=(max_x + 1, max_fitness_value - 10),
                 arrowprops=dict(facecolor='red', arrowstyle="->"))

    plt.title("Fitness Function Plot")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()


def display_stats(log_data):
    generations = [entry["gen"] for entry in log_data]
    avg_fitness = [entry["avg"] for entry in log_data]
    std_dev = [entry["std"] for entry in log_data]
    min_fitness = [entry["min"] for entry in log_data]
    max_fitness = [entry["max"] for entry in log_data]

    plt.figure(figsize=(20, 10))
    plt.plot(generations, avg_fitness, label="Average Fitness")
    plt.plot(generations, min_fitness, label="Minimum Fitness")
    plt.plot(generations, max_fitness, label="Maximum Fitness")
    plt.fill_between(generations,
                     [avg_fitness[i] - std_dev[i] for i in range(len(log_data))],
                     [avg_fitness[i] + std_dev[i] for i in range(len(log_data))],
                     color="gray", alpha=0.2, label="Standard Deviation")
    plt.title("Fitness Progress Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    config = load_config()
    num_generations, pop_size, crossover_rate, mutation_rate, lower_bound, upper_bound, function_str = config.values()
    validate_function_string(function_str)
    fitness_function = eval(f"lambda x: {function_str}")

    setup_deap()
    toolbox = configure_toolbox(mutation_rate)
    population = toolbox.population(n=pop_size)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    _, logbook = algorithms.eaSimple(
        population, toolbox,
        cxpb=crossover_rate,
        mutpb=mutation_rate,
        ngen=num_generations,
        stats=stats,
        verbose=True
    )

    best_individual = tools.selBest(population, 1)[0]
    max_x = best_individual[0]
    max_fitness_value = fitness_function(max_x)

    display_stats(logbook)
    display_fitness_function(lower_bound, upper_bound, max_x, max_fitness_value)
