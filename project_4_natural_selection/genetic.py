import PySimpleGUI as sg
import tsplib95 as tsp
import matplotlib.pyplot as plt
import matplotlib
import random
import time
import cv2
import io
import base64
import numpy as np
from datetime import datetime
from pathlib import Path
matplotlib.use('Agg')

population_size = 500
generations = 750
# Create the first dataset
crossover_rate = 0.70
mutation_rate = 0.05

# Initialize the GA variables
def initilize_population(cities):
    return[random.sample(cities, len(cities)) for _ in range(population_size)]

fitness_cache = {}
# Determine fitnesss
def fitness(route, tsp_instance):
    # Check the cache first
    route_tuple = tuple(route)
    if route_tuple in fitness_cache:
        return fitness_cache[route_tuple]
    
    # If not in the cache, compute the fitness
    fitness_value = tsp_instance.trace_tours([route])[0] + tsp_instance.get_weight(route[-1], route[0])
    
    # Store the computed fitness in the cache
    fitness_cache[route_tuple] = fitness_value
    
    return fitness_value

# Selection (Tournament selection)
def tournament_select(population, tsp_instance):
    tournament_size = 3
    selected = []
    for _ in range(2):
        participants = random.sample(population, tournament_size)
        selected.append(min(participants, key=lambda route: fitness(route, tsp_instance)))
    return selected

# Selection (Roulette wheel selection)
def roulette_wheel_select(population, tsp_instance):
    # Compute the total fitness of the population (using inverse for speed reasons)
    total_fitness = sum(1 / fitness(route, tsp_instance) for route in population) 

    # Select a random value between 0 and the total fitness
    wheel_spin = random.uniform(0, total_fitness)
    
    running_total = 0
    for route in population:
        running_total += 1 / fitness(route, tsp_instance)
        if running_total > wheel_spin:
            return route
        
# Crossover (Order crossover)
def crossover(parent1, parent2, cities):
    if random.random() > crossover_rate:
        return parent1
    
    start, end = sorted(random.sample(range(len(cities)), 2))
    child = [-1] * len(cities)
    child[start:end+1] = parent1[start:end+1]
    
    pointer = (end + 1) % len(cities)
    for city in parent2:
        if city not in child:
            child[pointer] = city
            pointer = (pointer + 1) % len(cities)

    return child

# Mutation (Inversion mutation)
def inversion_mutation(route):
    if random.random() > mutation_rate:
        return route
    
    i, j = sorted(random.sample(range(len(route)), 2))
    route[i:j+1] = reversed(route[i:j+1])
    return route

# Mutation (Swap mutation)
def swap_mutate(route):
    if random.random() > mutation_rate:
        return route
    
    i, j = random.sample(range(len(route)), 2)
    route[i], route[j] = route[j], route[i]
    return route

def GA(tsp_instance):
    fitness_distances = []
    generations_plotting = []
    image_buffers = []
    cities = list(tsp_instance.get_nodes())
    population = initilize_population(cities)
    start_time = time.time()
    best_distance = float('inf')
    elite_count = int(population_size * 0.01)

    for generation in range(generations):
        print("Generation: ", generation)
        elites = sorted(population, key=lambda route: fitness(route, tsp_instance))[:elite_count]
        new_population = []
        for _ in range(population_size - elite_count // 2):
            parent1, parent2 = tournament_select(population, tsp_instance)
            #parent1, parent2 = roulette_wheel_select(population, tsp_instance)
            child1 = crossover(parent1, parent2, cities)
            child2 = crossover(parent2, parent1, cities)

            #new_population.extend([swap_mutate(child1), swap_mutate(child2)])
            new_population.extend([inversion_mutation(child1), inversion_mutation(child2)])

        # Append the generation numbers to list for plotting
        generations_plotting.append(generation)

        new_population.extend(elites)
        
        population = new_population

        current_best_route = min(population, key=lambda route: fitness(route, tsp_instance))
        current_best_distance = fitness(current_best_route, tsp_instance)

        # Append the current best distance for each generation to list for plotting
        fitness_distances.append(current_best_distance)

        # Check if the current generation is one of the ones we want to display
        if generation % 5 == 0 or generation == generations - 1:  

            buffer, base64_image = plot_tsp(current_best_route, tsp_instance, generation)
            image_buffers.append(buffer)
            print(current_best_distance)
            # Update the GUI image with the base64 string
            window['-IMAGE-'].update(data=base64_image)
            window.refresh()

        if current_best_distance < best_distance:
            best_route = current_best_route
            best_distance = current_best_distance

    create_video(image_buffers, "tsp_evolution.avi")
    end_time = time.time()
    execution_time = end_time - start_time
    return best_route, best_distance, execution_time, fitness_distances, generations_plotting

def create_video(image_buffers, video_name):
    # Read the first image to get dimensions
    img = cv2.imdecode(np.frombuffer(image_buffers[0].getvalue(), np.uint8), cv2.IMREAD_COLOR)
    height, width, layers = img.shape
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"DIVX"), 2, (width, height))
    
    for buf in image_buffers:
        img = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        video.write(img)

    video.release()

def plot_tsp(best_route, tsp_instance, generation):
    coords = tsp_instance.node_coords

    best_route_cycle = best_route + [best_route[0]]  # Append first city to end of list to complete cycle
    x = [coords[node][0] for node in best_route_cycle]
    y = [coords[node][1] for node in best_route_cycle]

    # Plot the path
    plt.plot(x, y, marker='o', linestyle='-')

    # Number all nodes in the best_route order
   # for i, node in enumerate(best_route):
      #  plt.annotate(str(i+1), (coords[node][0], coords[node][1]), fontsize=12, ha='center', va='bottom', color='blue')
    for node in best_route:
        plt.annotate(str(node), (coords[node][0], coords[node][1]), fontsize=12, ha='center', va='bottom', color='blue')

    plt.title('Optimal Tour for Generation')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()

    # Convert buffer to base64 for GUI
    base64_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    # Reset buffer position so it can be read from elsewhere
    buf.seek(0)
    
    return buf, base64_image

# Function to plot a linear graph examining the fitness of current best route for each generation
def plot_fitness(generational_plotting, fitness_distances):
    plt.plot(generational_plotting, fitness_distances, marker='o', linestyle='-')
    plt.title('Fitness of Best Route for Each Generation')
    plt.xlabel('Generation')
    plt.ylabel('Fitness (Distance)')
    plt.grid(True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    plot_filename = f"fitness_plot_{timestamp}.png"
    plt.savefig(plot_filename)
    plt.close()

    return plot_filename

# This is simply the popup extension for if you want to preview a file
def popup_text(filename, text): 

    layout = [
        [sg.Multiline(text, size=(80, 25)),],
    ]
    win = sg.Window(filename, layout, modal=True, finalize=True)

    while True:
        event, values = win.read()
        if event == sg.WINDOW_CLOSED:
            break
    win.close()

sg.theme("DarkBlack")
sg.set_options(font=("Microsoft JhengHei", 16))

layout = [
    [
        sg.Input(key='-INPUT-'),
        sg.FileBrowse(file_types=(("ALL Files", "*.*"), ("TXT Files", "*.txt"))),
        sg.Button("Open", size= (8,2), font=("Microsoft JhengHei", 10)),
        sg.Button("Clear", size=(8,2), font=("Microsoft JhengHei", 10)),
        sg.Button("Close Application", size=(8,2), font=("Microsoft JhengHei", 10)),
        sg.Button("Download", size=(8,2), font=("Microsoft JhengHei", 10)),
        sg.Column([
            [sg.Button("Run Genetic Algorithm", size=(12,2), font=("Microsoft JhengHei", 10))],
        ]),
    ],
    [
        # Add in text
        sg.Column([
            [sg.Text("Best Path: "), sg.Multiline("", key='-BEST-PATH-', size=(200,2), font=("Microsoft JhengHei", 14))],
            [sg.Text("Optimal Distance:"), sg.Text("", key='-MINIMUM-DISTANCE-')],
            [sg.Text("Execution Time: "), sg.Text("", key='-EXECUTION-TIME-')],
        ])
    ],
    [
        # Add in picture of graph
        sg.Column([
        [sg.Image(key='-IMAGE-', size=(300, 300))]
        ]),
        sg.Column([
        [sg.Image(key='-FITNESS-PLOT-', size=(300, 300))],  # Fitness plot
        ]),
    ]
]

window = sg.Window('Traveling Sales Merchant', layout, size = (800, 600), margins=(100,50), finalize=True)

window.Maximize()

# While application is running
while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED or event == "Close Application":
        break
    elif event == 'Open': # This is for simply previewing the file
        filename = values['-INPUT-']
        if Path(filename).is_file():
            try:
                with open(filename, "rt", encoding='utf-8') as f:
                    text = f.read()                 
                popup_text(filename, text)
            except Exception as e:
                print("Error: ", e)
    # Button event for Closest Edge 
    
    elif event == "Run Genetic Algorithm":
        filename = values['-INPUT-']
        if Path(filename).is_file():
            tsp_instance = tsp.load(filename)
            best_route, shortest_distance, execution_time, fitness_distances, generations_plotting = GA(tsp_instance)

            fitness_plot_filename = plot_fitness(generations_plotting, fitness_distances)

            window['-BEST-PATH-'].update(" -> ".join(map(str, best_route)))
            window['-MINIMUM-DISTANCE-'].update(f"{shortest_distance:.2f}")
            window['-EXECUTION-TIME-'].update(f"{execution_time:.4f} seconds")
            window['-FITNESS-PLOT-'].update(filename=fitness_plot_filename)
            

    elif event == "Clear": #For clearing the entire GUI
        window['-BEST-PATH-'].update('')
        window["-INPUT-"].update('')
        window['-IMAGE-'].update(filename=None)
        window['-MINIMUM-DISTANCE-'].update("")
        window['-EXECUTION-TIME-'].update("")
        window['-INSERTION-ORDER-'].update("")

    elif event == "Download":
        save_path = sg.popup_get_file("Save Video As", save_as=True, default_extension=".avi", file_types=(("AVI Files", "*.avi"),))
        if save_path:
            import shutil
            shutil.copy("tsp_evolution.avi", save_path)


window.close()
