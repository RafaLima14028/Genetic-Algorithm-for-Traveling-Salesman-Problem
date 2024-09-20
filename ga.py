import random
import numpy as np


def calculate_cost(route, distance_matrix):
    """
    Calculates the total cost (distance) of a route.

    Args:
    route: List representing the order of the cities visited.
    distance_matrix: Matrix of distances between cities.

    Returns: The total cost of the route.
    """
    cost = 0

    for i in range(len(route) - 1):
        cost += distance_matrix[route[i]][route[i + 1]]

    cost += distance_matrix[route[-1]][route[0]]  # Return to the starting city

    return cost


def tournament_selection(population, distance_matrix):
    """
    Selects 50% of the population using tournament.

    Args:
    population: List of routes (individuals).
    distance_matrix: Matrix of distances.

    Returns: A list of selected individuals.
    """
    num_selected = len(population) // 2
    selected_population = []

    for _ in range(num_selected):
        # Choose the best individual in the tournament
        tournament_participants = random.sample(population, num_selected)

        # Choose the best individual in the tournament
        winner = min(
            tournament_participants,
            key=lambda route: calculate_cost(route, distance_matrix)
        )

        selected_population.append(winner)

    return selected_population


def crossover_order(parent1, parent2):
    """
    Applies Order Crossover (OX) to generate two children.

    Args:
    parent1, parent2: Parent routes.

    Returns: Two children resulting from the crossover.
    """

    def create_child(p1, p2):
        size = len(p1)
        start, end = sorted(random.sample(range(size), 2))

        child = [None] * size
        child[start:end] = p1[start:end]

        pointer = 0

        for city in p2:
            if city not in child:
                while child[pointer] is not None:
                    pointer += 1

                child[pointer] = city

        return child

    # Generate two children: one from parent1 -> parent2 and another from parent2 -> parent1
    child1 = create_child(parent1, parent2)
    child2 = create_child(parent2, parent1)

    return child1, child2


def two_opt_mutation(route):
    """
    Applies the 2-opt mutation to improve the route.

    Args:
    route: The route to be mutated.

    Returns: The mutated route.
    """
    new_route = route.copy()

    i, j = sorted(random.sample(range(len(route)), 2))
    new_route[i:j + 1] = reversed(new_route[i:j + 1])

    return new_route


def generate_population(N, distance_matrix):
    """
    Generates a population of N individuals using the nearest neighbor heuristic.

    Args:
    N: number of individuals
    distance_matrix: matrix of distances between cities

    Returns: A list of individuals (each is a solution represented by a list of cities).
    """
    num_cities = len(distance_matrix)
    cities = list(range(num_cities))
    population = []

    for _ in range(N):
        start_city = random.choice(cities)
        individual = nearest_neighbor(cities, start_city, distance_matrix)

        population.append(individual)

    return population


def nearest_neighbor(cities, start_city, distance_matrix):
    """
    Applies the nearest neighbor heuristic from a starting city.

    Args:
    cities: list of cities
    start_city: starting city
    distance_matrix: matrix of distances between cities

    Returns: A list representing the path of a solution.
    """
    unvisited = cities.copy()
    unvisited.remove(start_city)

    tour = [start_city]
    current_city = start_city

    while unvisited:
        next_city = min(unvisited, key=lambda city: distance_matrix[current_city][city])

        tour.append(next_city)
        unvisited.remove(next_city)

        current_city = next_city

    return tour


def generate_random_individual(distance_matrix):
    """
    Generates a random individual (route) for the TSP.

    Args:
    distance_matrix: Matrix of distances between cities.

    Returns: An individual, represented as a list of cities (permutation of cities).
    """
    while True:
        num_cities = len(distance_matrix)
        individual = list(range(num_cities))
        random.shuffle(individual)

        # Checks whether the generated route respects the capacity restriction
        current_load = 0

        for city in individual:
            current_load += demands[city]

            if current_load > capacity:
                break  # Invalid route, exits inner loop

        # If you have reached the end of the inner loop without encountering any violations, the route is valid
        else:
            return individual


def genetic_algorithm(distances, population_size, crossover_rate, mutation_rate, generations,
                      adjust_rate=False, max_no_improvement=50, elitism=False, elite_size=1):
    """
    Implements a genetic algorithm to solve the traveling salesman problem.

    Args:
    distances (list): Array of distances between cities.
    population_size (int): Population size.
    crossover_rate (float): Probability of crossover.
    mutation_rate (float): Probability of mutation.
    generations (int): Number of generations.
    adjust_rate (boolean): If after 20 iterations without improvements the algorithm
    should start changing the crossover_rate and mutation_rate.
    Also, if after 30 iterations without improvements, the population size
    divided by 10 should be created random individuals.
    max_no_improvement (int): Number of generations without improvement to stop.
    elitism (boobean): Elitism should be used to always maintain the best individual in the population.
    elite_size (int): The number of best individuals that should be maintained every generation.

    Returns:
    tuple: The best route found and its total cost.
    """
    # Generate initial population using nearest neighbor heuristic
    population = generate_population(population_size, distances)
    best_solution = min(population, key=lambda route: calculate_cost(route, distances))
    best_cost = calculate_cost(best_solution, distances)

    no_improvement_count = 0

    for generation in range(generations):
        if no_improvement_count >= max_no_improvement:
            break

        if adjust_rate:
            if no_improvement_count > 20:
                crossover_rate = min(1.0, crossover_rate + 0.2)
                mutation_rate = min(1.0, mutation_rate + 0.2)
            else:
                crossover_rate = max(0.1, crossover_rate - 0.2)
                mutation_rate = max(0.1, mutation_rate - 0.2)

            if no_improvement_count > 30:
                num_random_individuals = population_size // 10

                for _ in range(num_random_individuals):
                    random_individual = generate_random_individual(distances)
                    population[random.randint(0, len(population) - 1)] = random_individual

        if elitism:
            elite = sorted(
                population,
                key=lambda route: calculate_cost(route, distances)
            )[:elite_size]

        new_population = []

        # Select new population using tournament
        selected_population = tournament_selection(population, distances)

        for i in range(0, len(selected_population), 2):
            parent1 = selected_population[i]

            if i + 1 < len(selected_population):
                parent2 = selected_population[i + 1]
            else:
                parent2 = random.choice(selected_population)  # Select a second random parent if needed

            # Crossover
            if random.random() < crossover_rate:
                child1, child2 = crossover_order(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Apply mutation
            if random.random() < mutation_rate:
                child1 = two_opt_mutation(child1)
            if random.random() < mutation_rate:
                child2 = two_opt_mutation(child2)

            new_population.extend([child1, child2])

        if elitism:
            new_population[:elite_size] = elite

        population = new_population
        current_best = min(
            population,
            key=lambda route: calculate_cost(route, distances)
        )
        current_best_cost = calculate_cost(current_best, distances)

        # Check if there has been improvement
        if current_best_cost <= best_cost:
            best_solution = current_best
            best_cost = current_best_cost

            no_improvement_count = 0  # Reset the upgrade counter
        else:
            no_improvement_count += 1

    return best_solution, best_cost
