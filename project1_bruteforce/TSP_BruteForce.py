import PySimpleGUI as sg
import tsplib95 as tsp
import matplotlib
import matplotlib.pyplot as plt
import threading as th
from datetime import datetime
from itertools import permutations
from pathlib import Path
matplotlib.use("Agg")

# Calculates the total distance of the path being taken
def calculate_total_distance(path, tsp_instance):
    total_distance = 0
    for i in range(len(path) - 1):
        # get_weight is part of tsplib95 library that calculates the distance between nodes (using standard distance formula)
        total_distance += tsp_instance.get_weight(path[i], path[i + 1])
    total_distance += tsp_instance.get_weight(path[-1], path[0])
    return total_distance

# Where brute forcing occurs for TSP problem
def brute_force(tsp_instance):
    nodes = list(tsp_instance.get_nodes())
    min_distance = float('inf') # First distance calculated always less than infinity
    best_path = None
    permutations_checked = 0

    for perm in permutations(nodes):
        permutations_checked += 1
        distance = calculate_total_distance(perm, tsp_instance) #Calculate Distance of current permutation
        if distance < min_distance:
            min_distance = distance
            best_path = perm
        
        #Display progress per 1000 permutations so user can know its still going
        if permutations_checked % 50000 == 0:
            print(f"Permuations checked: {permutations_checked}")

    return best_path, min_distance, permutations_checked

# Plots graph of hamiltonian path for problem
def plot_tsp(best_path, tsp_instance, plot_filename_list):
    coords = tsp_instance.node_coords
    x = [coords[node][0] for node in best_path]
    y = [coords[node][1] for node in best_path]
    x.append(coords[best_path[0]][0])
    y.append(coords[best_path[0]][1])

    plt.plot(x, y, marker= 'o')
    plt.title('Best Tour Found')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    plot_filename = (f"hamiltonian_path_{timestamp}.png")
    plt.savefig(plot_filename)
    plt.close()

    plot_filename_list.append(plot_filename)

    return plot_filename

# Calls plot_tsp to create a thread of instance so application doesn't crash
def plot_tsp_thread(best_path, tsp_instance, plot_list):
    #Plot hamiltonian path in a thread so it doesn't freeze the GUI
    plot_list.append(plot_tsp(best_path, tsp_instance))

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

sg.theme("DarkBlue3")
sg.set_options(font=("Microsoft JhengHei", 16))

layout = [
    [
        sg.Input(key='-INPUT-'),
        sg.FileBrowse(file_types=(("ALL Files", "*.*"), ("TXT Files", "*.txt"))),
        sg.Button("Open"),
        sg.Button("Clear"),
        sg.Button("Run Brute Force"),
        sg.Button("Close Application")
    ],
    [
        # Add in hamiltonian traced path and text (2 columns)
        sg.Image(key='-IMAGE-', size=(300, 300)),
        sg.Column([
            [sg.Text("Best Path:"), sg.Text("", key='-BEST-PATH-')],
            [sg.Text("Minimum Distance:"), sg.Text("", key='-MINIMUM-DISTANCE-')],
        ])
    ]
]

window = sg.Window('Traveling Sales Merchant Brute Force', layout, margins=(150,75))

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

    elif event == "Run Brute Force":
        #Clear all GUI Images and Text to ensure latest copy
        window['-IMAGE-'].update(filename=None)
        window['-BEST-PATH-'].update("")
        window['-MINIMUM-DISTANCE-'].update("")

        filename = values['-INPUT-']
        if Path(filename).is_file():
            try:
                # Load tsp file chosen for brute force
                tsp_instance = tsp.load(filename)

                # Prints file to console (for debugging initially)
                print(tsp_instance) 

                # Call brute force to start finding best path
                best_path, min_distance, permutations_checked = brute_force(tsp_instance)

                # Prints to console so you know application hasn't crashed (its just slow for 12 cities)
                print(f"Permutations checked: {permutations_checked}")
                print(f"Best path: {best_path}")
                print(f"Minimum distance: {min_distance}")

                #Initialize list for plots to be saved
                plot_filename_list = []

                # Plot hamiltonian thread
                plot_tsp_thread = th.Thread(target=plot_tsp, args=(best_path, tsp_instance, plot_filename_list))
                plot_tsp_thread.start()

                # Ensure plotting is finished before saving the image
                plot_tsp_thread.join()

                #Close the plot as it is already saved as an image
                plt.close()

                plot_filename = plot_filename_list[0]

                # Update the Image and text on GUI with latest run
                window['-IMAGE-'].update(filename=plot_filename)
                window['-BEST-PATH-'].update(" ".join(map(str, best_path)))
                window['-MINIMUM-DISTANCE-'].update(str(min_distance))

            except Exception as e:
                print("Error: ", e)
    
    elif event == "Clear": #For clearing the entire GUI
        window["-INPUT-"].update('')
        window['-IMAGE-'].update(filename=None)
        window['-BEST-PATH-'].update("")
        window['-MINIMUM-DISTANCE-'].update("")

window.close()