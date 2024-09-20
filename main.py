import time
from statistics import mean

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from python_tsp.distances import tsplib_distance_matrix

from ga import genetic_algorithm

if __name__ == '__main__':
    list_files_graph = ['brazil58', 'pr76', 'ch150', 'bayg29', 'pr124', 'pr107',
                        'pr136', 'pr144', 'eil101', 'lin105', 'eil76', 'eil51',
                        'berlin52', 'st70', 'bier127']

    best_known_values = {
        'brazil58': 25395,
        'pr76': 108159,
        'ch150': 6528,
        'bayg29': 1610,
        'pr124': 59030,
        'pr107': 44303,
        'pr136': 96772,
        'pr144': 58537,
        'eil101': 629,
        'lin105': 14379,
        'eil76': 538,
        'eil51': 426,
        'berlin52': 7542,
        'st70': 675,
        'bier127': 118282
    }

    # Genetic algorithm parameters
    population_size = 200
    crossover_rate = 0.8
    mutation_rate = 0.9
    generations = 5_000
    adjust_rate = False
    elitism = True
    elitism_size = 1
    max_no_improvement = 500

    num_executions = 20

    results = []

    for file in list_files_graph:
        print(f'Processing {file}...')
        distances = tsplib_distance_matrix(f'tsplib-data/{file}.tsp')

        best_costs = []
        times = []

        for _ in range(num_executions):
            start_time = time.time()

            best_route, best_cost = genetic_algorithm(
                distances,
                population_size,
                crossover_rate,
                mutation_rate,
                generations,
                adjust_rate,
                max_no_improvement,
                elitism,
                elitism_size
            )

            end_time = time.time()

            best_costs.append(best_cost)
            times.append(end_time - start_time)

        best_tour_cost = min(best_costs)
        average_cost = mean(best_costs)
        average_time = mean(times)

        percentage_deviation = average_cost / best_known_values[file]

        results.append(
            [file, best_known_values[file], best_tour_cost, average_cost, percentage_deviation, average_time]
        )

        print(f"Best Route Cost for {file}: {best_tour_cost}")
        print(f"Average Cost over {num_executions} runs: {average_cost}")
        print(f"Average Time over {num_executions} runs: {average_time:.2f} seconds")
        print(f"Percentage Deviation from Best Known Value: {percentage_deviation:.2f}")

        print('#' * 50)

    # Create a DataFrame from the results
    df = pd.DataFrame(results, columns=['TSPLIB', 'MTC', 'MT', 'MDT', 'Desv', 'TP'])
    df['Desv'] = df['Desv'].round(2)  # Round the 'Dev' column to 2 decimal places
    df['TP'] = df['TP'].round(2)  # Round column 'TP' to 2 decimal places

    # Creation of the figure and axes
    fig, ax = plt.subplots(figsize=(10, len(list_files_graph) * 0.6))

    # Remove edges from axes
    ax.axis('tight')
    ax.axis('off')

    # Configure colors for alternating rows
    colors = ['#f1f1f2', '#e8e8e8']  # Light gray tones

    # Create the table with styling
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center',
                     rowColours=[colors[i % 2] for i in range(len(df))],  # Switch background colors
                     colColours=['#4f81bd'] * len(df.columns),  # Header background color
                     )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)  # Increase cell size

    # Header style
    for j, col in enumerate(df.columns):
        cell = table[(0, j)]  # Access cell in row 0, column j (header)
        cell.set_text_props(weight='bold', color='white')  # Make text bold and white
        cell.set_fontsize(12)

    # Add borders to cells
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(weight='bold')
            cell.set_linewidth(0.5)
        else:
            cell.set_linewidth(0.25)
            cell.set_edgecolor(mcolors.to_rgba('black', 0.3))

    # Título da tabela
    plt.title('Genetic Algorithm Results for TSPLIB Instances', pad=20, fontsize=14, fontweight='bold')

    # Save the table as a PNG image
    plt.savefig('genetic_algorithm_results.png', bbox_inches='tight', dpi=1000)
    plt.show()
