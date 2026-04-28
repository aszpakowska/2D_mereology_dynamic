import matplotlib as plt
import matplotlib.pyplot as pypl
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import math
import numpy as np
import pygame
from pygame.locals import *
import heapq
from typing import List, Tuple, Any, Dict
import sys

# --- Initialization and Helpers ---

def read_coordinates(file_name):
    with open(file_name) as file:
        coordinates = []
        for line in file:
            row = line.split()
            x = row[0]
            y = row[1]
            coordinates.append([float(x), float(y)])
    return coordinates

def d_eucl(x_temp, y_temp, x2_temp, y2_temp):
    return math.sqrt((x_temp - x2_temp)**2 + (y_temp - y2_temp)**2)

def check_obstacle(obstacles_coordinates, x_temp, y_temp, radius=55):
    for obs in obstacles_coordinates:
        if d_eucl(x_temp, y_temp, obs[0], obs[1]) < radius:
            return 1
    return 0

def find_minimum_of_the_sum(list_of_coords):
    minimum = 1000
    point = []
    for item in list_of_coords:
        sum_of_xy = item[0] + item[1]
        if sum_of_xy < minimum:
            minimum = sum_of_xy
            point = [[item[0], item[1]]]
    return point

def getImage(path, zoom=0.07):
    return OffsetImage(pypl.imread(path), zoom=zoom)

# --- A* Algorithm ---

def heuristic(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def a_star(start, goal, obstacles, x_max, y_max, step_size=10, surf=None):
    open_set = []
    # (f_score, count, current_node)
    count = 0
    heapq.heappush(open_set, (heuristic(start, goal), count, start))
    
    came_from = {}
    g_score = {start: 0.0}
    
    visited = set()

    while open_set:
        f, _, current = heapq.heappop(open_set)
        
        if current in visited:
            continue
        visited.add(current)

        # Check if reached goal (within step size)
        if heuristic(current, goal) < step_size * 1.5:
            # Reconstruct path
            path = [goal]
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return list(reversed(path))

        # Neighbors (8 directions)
        x, y = current
        neighbors = [
            (x + step_size, y), (x - step_size, y),
            (x, y + step_size), (x, y - step_size),
            (x + step_size, y + step_size), (x - step_size, y - step_size),
            (x + step_size, y - step_size), (x - step_size, y + step_size)
        ]

        for neighbor in neighbors:
            nx, ny = neighbor
            
            # Bounds check
            if not (0 <= nx <= x_max and 0 <= ny <= y_max):
                continue
                
            # Obstacle check
            if check_obstacle(obstacles, nx, ny, radius=30) == 1: # Adjusted radius for A*
                continue
            
            tentative_g = g_score[current] + heuristic(current, neighbor)
            
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                count += 1
                heapq.heappush(open_set, (f_score, count, neighbor))
                
                # Visual feedback for exploration
                if surf:
                    pygame.draw.circle(surf, (200, 200, 200), (int(nx), int(ny)), 1)

    return []

# --- Path Smoothing ---

def path_smoothing(path, iter_number):
    if len(path) < 3: return path
    x_list = [p[0] for p in path]
    y_list = [p[1] for p in path]
    x0, y0 = x_list.copy(), y_list.copy()
    
    alpha, beta = 0.3, 0.3
    
    for _ in range(iter_number):
        for i in range(1, len(path) - 1):
            x_list[i] += alpha * (x_list[i-1] + x_list[i+1] - 2 * x_list[i]) + beta * (x0[i] - x_list[i])
            y_list[i] += alpha * (y_list[i-1] + y_list[i+1] - 2 * y_list[i]) + beta * (y0[i] - y_list[i])
            
    return list(zip(x_list, y_list))

# --- Main Logic ---

# 1. Load Data
goal_coordianets = [(250, 70)] # Default values
center_list = read_coordinates('old_txt/txt_files5/boundaries_detector.txt')
obstacles_coordinates = [(250, 250), (250, 300)]
start_coordinates = [(250, 400)]

# 2. Normalize
mins = find_minimum_of_the_sum(center_list)
x_min, y_min = mins[0][0], mins[0][1]

# Width and Height from borders
x_raw = [x for x, y in center_list]
y_raw = [y for x, y in center_list]
padding = 50
width = int(max(x_raw) - x_min + 2 * padding)
height = int(max(y_raw) - y_min + 2 * padding)

# Transformation function: Normalize to (padding, padding) and flip Y for Pygame
def transform(p):
    return (int(p[0] - x_min + padding), int(height - (p[1] - y_min + padding)))

x_start_p, y_start_p = transform(start_coordinates[0])
x_goal_p, y_goal_p = transform(goal_coordianets[0])

obs_p = [transform(p) for p in obstacles_coordinates]
border_p = [transform(p) for p in center_list]

# Matplotlib plot generation - PIXEL PERFECT (No margins)
dpi = 100
fig = pypl.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
ax = fig.add_axes([0, 0, 1, 1]) # Fill entire figure, no margins

# Use paths for icons
marker_paths = ['marker_images/marker_1751.png', 'marker_images/marker_4076.png', 
                'marker_images/marker_1281.png', 'marker_images/marker_1184.png']
obs_paths = ['marker_images/marker_733.png', 'marker_images/marker_2165.png']
goal_path = 'marker_images/marker_497.png'
start_path = 'start_images/start_point.jpg'

for i, p in enumerate(border_p):
    if i < len(marker_paths): ax.add_artist(AnnotationBbox(getImage(marker_paths[i]), p, frameon=False))

for i, p in enumerate(obs_p):
    if i < len(obs_paths): ax.add_artist(AnnotationBbox(getImage(obs_paths[i]), p, frameon=False))

ax.add_artist(AnnotationBbox(getImage(goal_path), (x_goal_p, y_goal_p), frameon=False))
ax.add_artist(AnnotationBbox(getImage(start_path), (x_start_p, y_start_p), frameon=False))

ax.set_xlim(0, width)
ax.set_ylim(height, 0) # Invert Y axis: 0 is top
ax.axis('off')
pypl.savefig('plots/my_plot_final.png', dpi=dpi)
pypl.close(fig)

# 3. Pygame Rendering & Dynamic A* Interaction
pygame.init()
bg_surf = pygame.image.load('plots/my_plot_final.png')
size = bg_surf.get_size()
screen = pygame.display.set_mode(size)
pygame.display.set_caption("A* Dynamic Pathfinding")

start_node = (x_start_p, y_start_p)
goal_node = (x_goal_p, y_goal_p)

dynamic_obstacles = set(obs_p)

def get_path_a_star(current_obs):
    if check_obstacle(current_obs, start_node[0], start_node[1], radius=30) == 1: return "ERROR"
    if check_obstacle(current_obs, goal_node[0], goal_node[1], radius=30) == 1: return "ERROR"

    raw_path = a_star(start_node, goal_node, current_obs, size[0], size[1], step_size=8, surf=None)
    if raw_path:
        smooth_path = path_smoothing(raw_path, 20)
        smooth_path[0] = start_node
        smooth_path[-1] = goal_node
        return smooth_path
    return []

clock = pygame.time.Clock()
run = True
update_count = 0
pygame.font.init()
font = pygame.font.SysFont('Arial', 24)

print("Starting A* Pathfinding Dynamic UI...")
current_path = get_path_a_star(dynamic_obstacles)

while run:
    screen.blit(bg_surf, (0, 0))
    
    obst_surf = pygame.Surface(size, pygame.SRCALPHA)
    for obs in dynamic_obstacles:
        pygame.draw.circle(obst_surf, (200, 0, 0, 80), (int(obs[0]), int(obs[1])), 30)
    screen.blit(obst_surf, (0, 0))

    if current_path == "ERROR":
        warning_surf = font.render("Błąd: Start lub cel w obszarze przeszkody!", True, (255, 0, 0))
        screen.blit(warning_surf, (20, 50))
    elif current_path and isinstance(current_path, list) and len(current_path) > 1:
        points = [(int(x), int(y)) for x, y in current_path]
        pygame.draw.lines(screen, (0, 200, 0), False, points, 5)

    pygame.draw.circle(screen, (255, 0, 0), start_node, 5)
    pygame.draw.circle(screen, (255, 0, 0), goal_node, 5)

    text_surf = font.render(f"Liczba aktualizacji: {update_count}", True, (255, 255, 0))
    screen.blit(text_surf, (20, 20))

    for event in pygame.event.get():
        if event.type == QUIT:
            run = False
        if event.type == MOUSEBUTTONDOWN:
            m_pos = pygame.mouse.get_pos()
            changed = False
            
            if event.button == 1: # Left click adds
                new_obs = (m_pos[0], m_pos[1])
                dynamic_obstacles.add(new_obs)
                changed = True
            elif event.button == 3: # Right click removes
                to_remove = set()
                for obs in dynamic_obstacles:
                    if d_eucl(m_pos[0], m_pos[1], obs[0], obs[1]) < 30:
                        to_remove.add(obs)
                if to_remove:
                    dynamic_obstacles.difference_update(to_remove)
                    changed = True

            if changed:
                new_path = get_path_a_star(dynamic_obstacles)
                if current_path != new_path:
                    update_count += 1
                current_path = new_path

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
