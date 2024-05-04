import PySimpleGUI as sg
import tsplib95 as tsp
import matplotlib.pyplot as plt
import time
import queue
from datetime import datetime
from pathlib import Path

def dfs(tsp_instance):
    number_of_nodes = tsp_instance.dimension # get length of tsp file
    starting_node = 1  # Initialize the starting node as 1 (matches to connected edges)
    end_node = number_of_nodes  # Initialize the last node as the length of tsp_instance (11)
    visited_nodes = [False] * (number_of_nodes + 1)  # Initialize all nodes in tsp_instance as not visited yet
    transitions = [0]  # Initialize the transition counter

    # Recursive function for DFS
    def recursive_dfs(current_node, current_distance, current_path, optimal_distance, optimal_path):
        transitions[0] += 1  # Increment the transition counter
       # print(current_path, current_distance)

       # Check if the current node is the end or target and update distance if less than optimal currently
        if current_node == end_node:
            if current_distance < optimal_distance[0]:
                optimal_distance[0] = current_distance
                optimal_path[0] = current_path.copy()
            return

        # Mark node as has been visited
        visited_nodes[current_node] = True
        # Get the neighbors of the node
        neighbors = connected_edges.get(current_node, [])

        # Explore to unvisited neighbors
        for neighbor in neighbors:
            if not visited_nodes[neighbor]:
                distance = tsp_instance.get_weight(current_node, neighbor)
                # Append the neighbor to the path
                current_path.append(neighbor)
                current_distance += distance
                # Call recursive DFS for the neighbor of current node
                recursive_dfs(neighbor, current_distance, current_path, optimal_distance, optimal_path)

                # Backtrack removing the last node from the current path
                current_path.pop()
                current_distance -= distance

        # Mark the current node as being unvisited
        visited_nodes[current_node] = False

    optimal_distance = [float('inf')]
    optimal_path = [None]

    # Initial DFS recursion, where 0 is the current distance (havent started yet)
    recursive_dfs(starting_node, 0, [starting_node], optimal_distance, optimal_path)

   # print("Optimal Distance (DFS):", optimal_distance[0])
   # print("Optimal Path (DFS):", optimal_path[0])
   # print("Number of Transitions (DFS):",transitions[0])
    return optimal_distance[0], optimal_path[0], transitions[0]

def bfs(tsp_instance):
    number_of_nodes = tsp_instance.dimension # Get dimension of file which is the number of nodes
    starting_node = 1  # Initialize starting node as 1
    end_node = number_of_nodes  # Initialize last node which will be 11
    visited_nodes = [False] * (number_of_nodes + 1) # Initialize visited nodes to false for all nodes
    optimal_distance = float('inf') # Initialize optimal distance to inifinity
    optimal_path = None # Initialize the optimal path to be nothing
    transitions = 0 # Intialize transition counter
    traversal_queue = queue.Queue()
 

    for neighbor in connected_edges[starting_node]:
        distance = tsp_instance.get_weight(starting_node, neighbor)
        initial_path = [starting_node, neighbor]
        # Add neighbor, distance, and initial path to queue
        traversal_queue.put((neighbor, distance, initial_path))

    # Continue until all nodes in connected_edges have been visited
    while not traversal_queue.empty():
        current_node, current_distance, current_path = traversal_queue.get()
        transitions += 1  # Increment the transition counter
      #  print(current_path)

      # Check to see if we have reached the end node yet
        if current_node == end_node:
            if current_distance < optimal_distance:
                optimal_distance = current_distance
                optimal_path = current_path
            continue
        
        # Mark current node as visited
        visited_nodes[current_node] = True
        # Get neighbors of current node
        neighbors = connected_edges.get(current_node, [])

        # Explore unvisited neighbors
        for neighbor in neighbors:
            if not visited_nodes[neighbor]:
                distance = tsp_instance.get_weight(current_node, neighbor) 
                next_path = current_path + [neighbor]
                traversal_queue.put((neighbor, current_distance + distance, next_path))

  #  print("Best Distance:", optimal_distance)
  #  print("Best Path:", optimal_path)  
  #  print("Number of Transitions:", transitions)
    return optimal_distance, optimal_path, transitions

# Plots the path and the nodes 
def plot_tsp(optimal_path, tsp_instance):
    coords = tsp_instance.node_coords

    x = [coords[node][0] for node in optimal_path]
    y = [coords[node][1] for node in optimal_path]

    # Plot the path
    plt.plot(x, y, marker='o', linestyle='-')
    
    # Plot all nodes (Even the ones taht aren't in the path)
    all_nodes = list(coords.keys())
    for node in all_nodes:
        if node not in optimal_path:
            plt.plot(coords[node][0], coords[node][1], marker='o', markersize=8, color='blue')  # Change color and size as desired
            plt.annotate(str(node), (coords[node][0], coords[node][1]), fontsize=12, ha='center', va='bottom', color='black')

    # Number all nodes accordingly
    for i, node in enumerate(optimal_path):
        plt.annotate(str(node), (coords[node][0], coords[node][1]), fontsize=12, ha='center', va='bottom', color='blue')

    plt.title('Optimal Tour Found')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    plot_filename = f"hamiltonian_path_{timestamp}.png"
    plt.savefig(plot_filename)
    plt.close()

    return plot_filename

# Define the possible paths that can be taken as given to us in the Word Doc
connected_edges = {
    1: [2, 3, 4],
    2: [3],
    3: [4, 5],
    4: [5, 6, 7],
    5: [7, 8],
    6: [8],
    7: [9, 10],
    8: [9, 10, 11],
    9: [11],
    10: [11]
}

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
        sg.Button("Open"),
        sg.Button("Clear"),
        sg.Column([
            [sg.Button("Run DFS", size=(10,1))],
            [sg.Button("Run BFS", size=(10,1))],
        ]),
    ],
    [
        # Add in hamiltonian traced path and text (2 columns)
        sg.Image(key='-IMAGE-', size=(300, 300)),
        sg.Column([
            [sg.Text("Optimal Path:"), sg.Text("", key='-BEST-PATH-')],
            [sg.Text("Optimal Distance:"), sg.Text("", key='-MINIMUM-DISTANCE-')],
            [sg.Text("Execution Time: "), sg.Text("", key='-EXECUTION-TIME-')],
            [sg.Text("Transitions: "), sg.Text("", key='-TRANSITIONS-')]
        ])
    ],
    [
        [sg.Column([
            [sg.Button("Close Application")],
        ], justification="center")],
    ]
]

window = sg.Window('Traveling Sales Merchant', layout, margins=(150,75))

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

    # Button event to run BFS algorithm
    elif event == "Run BFS":
        # Initialize window as cleared
        window['-IMAGE-'].update(filename=None)
        window['-BEST-PATH-'].update("")
        window['-MINIMUM-DISTANCE-'].update("")
        window['-EXECUTION-TIME-'].update("")
        window['-TRANSITIONS-'].update("")

        filename = values['-INPUT-']
        if Path(filename).is_file():
            try:
                tsp_instance = tsp.load(filename)
                start_time = time.time() # Start run time
                optimal_distance, optimal_path, transitions = bfs(tsp_instance)  # Start BFS algorithm
                end_time = time.time() # End run time
                execution_time = end_time - start_time

                if optimal_path:
                    plot_filename = plot_tsp(optimal_path, tsp_instance)

                    window['-IMAGE-'].update(filename=plot_filename)
                    window['-BEST-PATH-'].update(" ".join(map(str, optimal_path)))  
                    window['-MINIMUM-DISTANCE-'].update(str(optimal_distance))
                    window['-EXECUTION-TIME-'].update(f"Execution Time: {execution_time:.5f}s")
                    window['-TRANSITIONS-'].update(f"{transitions}")
            except Exception as e:
                print("Error:", e)  
        else:
            sg.popup_error("File not found or invalid file path.")

    # Button event to run DFS algorithm
    elif event == "Run DFS":
        window['-IMAGE-'].update(filename=None)
        window['-BEST-PATH-'].update("")
        window['-MINIMUM-DISTANCE-'].update("")
        window['-EXECUTION-TIME-'].update("")
        window['-TRANSITIONS-'].update("")

        filename = values['-INPUT-']
        if Path(filename).is_file():
            try:
                tsp_instance = tsp.load(filename)
                start_time = time.time() # Start run time
                optimal_distance, optimal_path, transitions = dfs(tsp_instance) # Start DFS algorithm
                end_time = time.time() # End run time
                execution_time = end_time - start_time

                if optimal_path:
                    plot_filename = plot_tsp(optimal_path, tsp_instance)

                    window['-IMAGE-'].update(filename=plot_filename)
                    window['-BEST-PATH-'].update(" ".join(map(str, optimal_path)))
                    window['-MINIMUM-DISTANCE-'].update(str(optimal_distance))
                    window['-EXECUTION-TIME-'].update(f"Execution Time: {execution_time:.5f}s")
                    window['-TRANSITIONS-'].update(f"{transitions}")
            except Exception as e:
                pass
        else:
            sg.popup_error("File not found or invalid file path.")

    elif event == "Clear": #For clearing the entire GUI
        window["-INPUT-"].update('')
        window['-IMAGE-'].update(filename=None)
        window['-BEST-PATH-'].update("")
        window['-MINIMUM-DISTANCE-'].update("")
        window['-EXECUTION-TIME-'].update("")
        window['-TRANSITIONS-'].update("")

window.close()