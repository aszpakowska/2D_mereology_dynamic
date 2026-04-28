import matplotlib as plt
import matplotlib.pyplot as pypl
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import math
import numpy as np
import pygame
from pygame.locals import *
import heapq
from typing import List, Tuple, Any, Dict, Set
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

def d_eucl(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

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
    try:
        return OffsetImage(pypl.imread(path), zoom=zoom)
    except:
        # Fallback if image missing during dev
        return None

def path_smoothing(path, alpha=0.5, beta=0.1, tolerance=0.00001):
    if not path:
        return []
    new_path = [list(p) for p in path]
    change = tolerance
    while change >= tolerance:
        change = 0.0
        for i in range(1, len(path) - 1):
            for j in range(len(path[i])):
                old_val = new_path[i][j]
                new_path[i][j] += alpha * (path[i][j] - new_path[i][j]) + \
                                  beta * (new_path[i-1][j] + new_path[i+1][j] - 2.0 * new_path[i][j])
                change += abs(old_val - new_path[i][j])
    return [tuple(p) for p in new_path]

# --- D* Lite Algorithm Logic ---

class PriorityQueue:
    def __init__(self):
        self.vertices = []

    def top(self):
        if not self.vertices:
            return None, (float('inf'), float('inf'))
        return self.vertices[0][2], self.vertices[0][0]

    def top_key(self):
        if not self.vertices:
            return float('inf'), float('inf')
        return self.vertices[0][0]

    def pop(self):
        return heapq.heappop(self.vertices)[2]

    def insert(self, vertex, key):
        heapq.heappush(self.vertices, (key, id(vertex), vertex))

    def remove(self, vertex):
        self.vertices = [v for v in self.vertices if v[2] != vertex]
        heapq.heapify(self.vertices)

    def update(self, vertex, key):
        self.remove(vertex)
        self.insert(vertex, key)

    def contains(self, vertex):
        return any(v[2] == vertex for v in self.vertices)

class DStarLite:
    def __init__(self, start, goal, width, height, step):
        self.s_start = start
        self.s_goal = goal
        self.s_last = start
        self.k_m = 0
        self.step = step
        self.width = width
        self.height = height
        
        self.rhs = {}
        self.g = {}
        self.U = PriorityQueue()
        self.obstacles = set()
        
        # Initialize
        self.rhs[self.s_goal] = 0
        self.U.insert(self.s_goal, self.calculate_key(self.s_goal))

    def calculate_key(self, s):
        min_g_rhs = min(self.get_g(s), self.get_rhs(s))
        return (min_g_rhs + d_eucl(self.s_start, s) + self.k_m, min_g_rhs)

    def get_g(self, s):
        return self.g.get(s, float('inf'))

    def get_rhs(self, s):
        if s == self.s_goal:
            return 0
        return self.rhs.get(s, float('inf'))

    def get_neighbors(self, s):
        x, y = s
        results = []
        for dx in [-self.step, 0, self.step]:
            for dy in [-self.step, 0, self.step]:
                if dx == 0 and dy == 0: continue
                new_s = (x + dx, y + dy)
                if 0 <= new_s[0] <= self.width and 0 <= new_s[1] <= self.height:
                    results.append(new_s)
        return results

    def cost(self, s1, s2):
        if s1 in self.obstacles or s2 in self.obstacles:
            return float('inf')
        return d_eucl(s1, s2)

    def update_vertex(self, u):
        if u != self.s_goal:
            min_rhs = float('inf')
            for s_prime in self.get_neighbors(u):
                val = self.cost(u, s_prime) + self.get_g(s_prime)
                if val < min_rhs:
                    min_rhs = val
            self.rhs[u] = min_rhs
            
        if self.U.contains(u):
            self.U.remove(u)
            
        if self.get_g(u) != self.get_rhs(u):
            self.U.insert(u, self.calculate_key(u))

    def compute_shortest_path(self):
        while self.U.top_key() < self.calculate_key(self.s_start) or self.get_rhs(self.s_start) != self.get_g(self.s_start):
            k_old = self.U.top_key()
            u = self.U.pop()
            
            if k_old < self.calculate_key(u):
                self.U.insert(u, self.calculate_key(u))
            elif self.get_g(u) > self.get_rhs(u):
                self.g[u] = self.rhs[u]
                for s_prime in self.get_neighbors(u):
                    self.update_vertex(s_prime)
            else:
                self.g[u] = float('inf')
                for s_prime in [u] + self.get_neighbors(u):
                    self.update_vertex(s_prime)

    def update_obstacles(self, new_obstacles: Set[Tuple[int, int]]):
        changed_nodes = new_obstacles.symmetric_difference(self.obstacles)
        self.k_m += d_eucl(self.s_last, self.s_start)
        self.s_last = self.s_start
        
        # Zaktualizuj przeszkody PRZED aktualizacją wierzchołków,
        # by self.cost() brało pod uwagę nowe przeszkody
        self.obstacles = new_obstacles
        
        for node in changed_nodes:
            self.update_vertex(node)
            for neighbor in self.get_neighbors(node):
                self.update_vertex(neighbor)
        
        self.compute_shortest_path()

# --- Main Logic ---

# 1. Load Data
goal_coordianets = [(250, 70)] 
center_list = read_coordinates('old_txt/txt_files5/boundaries_detector.txt')
obstacles_coordinates = [(250, 250), (250, 300)]
start_coordinates = [(250, 400)]

# 2. Normalize
mins = find_minimum_of_the_sum(center_list)
x_min, y_min = mins[0][0], mins[0][1]

x_raw = [x for x, y in center_list]
y_raw = [y for x, y in center_list]
padding = 60 # Slightly larger padding for safety
width = int(max(x_raw) - x_min + 2 * padding)
height = int(max(y_raw) - y_min + 2 * padding)

step = 10

def transform(p):
    return (int(p[0] - x_min + padding), int(height - (p[1] - y_min + padding)))

def snap(p):
    return (int(p[0] // step * step), int(p[1] // step * step))

start_p = snap(transform(start_coordinates[0]))
goal_p = snap(transform(goal_coordianets[0]))
obs_p_initial = [transform(p) for p in obstacles_coordinates]
border_p = [transform(p) for p in center_list]

# Matplotlib plot generation (Background)
dpi = 100
fig = pypl.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
ax = fig.add_axes([0, 0, 1, 1])
marker_paths = ['marker_images/marker_1751.png', 'marker_images/marker_4076.png', 
                'marker_images/marker_1281.png', 'marker_images/marker_1184.png']
obs_paths = ['marker_images/marker_733.png', 'marker_images/marker_2165.png']
goal_path = 'marker_images/marker_497.png'
start_path = 'start_images/start_point.jpg'

# Borders
for i, p in enumerate(border_p):
    if i < len(marker_paths): 
        img = getImage(marker_paths[i])
        if img: ax.add_artist(AnnotationBbox(img, p, frameon=False))

# Static Obstacles (Icons)
for i, p in enumerate(obs_p_initial):
    if i < len(obs_paths):
        img = getImage(obs_paths[i])
        if img: ax.add_artist(AnnotationBbox(img, p, frameon=False))

ax.add_artist(AnnotationBbox(getImage(goal_path), transform(goal_coordianets[0]), frameon=False))
ax.add_artist(AnnotationBbox(getImage(start_path), transform(start_coordinates[0]), frameon=False))

ax.set_xlim(0, width)
ax.set_ylim(height, 0)
ax.axis('off')
pypl.savefig('plots/my_plot_dstar.png', dpi=dpi)
pypl.close(fig)

# 3. Pygame Rendering & D* Lite Interaction
pygame.init()
bg_surf = pygame.image.load('plots/my_plot_dstar.png')
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("D* Lite Dynamic Pathfinding")

# Create a transparent surface for obstacles
obst_surf = pygame.Surface((width, height), pygame.SRCALPHA)

# D* Lite setup
dstar = DStarLite(start_p, goal_p, width, height, step)

def add_obstacle_cluster(center_p, radius=35):
    new_obs = set()
    for dx in range(-radius, radius + 1, step):
        for dy in range(-radius, radius + 1, step):
            if math.sqrt(dx**2 + dy**2) <= radius:
                new_obs.add(snap((center_p[0] + dx, center_p[1] + dy)))
    return new_obs

# Initial obstacles
initial_obs_set = set()
for op in obs_p_initial:
    initial_obs_set.update(add_obstacle_cluster(op, radius=40))
dstar.update_obstacles(initial_obs_set)

def get_path():
    path = []
    curr = dstar.s_start
    visited = {curr}
    max_steps = 2000
    while curr != dstar.s_goal and max_steps > 0:
        path.append(curr)
        min_cost = float('inf')
        next_node = None
        for neighbor in dstar.get_neighbors(curr):
            cost = dstar.cost(curr, neighbor) + dstar.get_g(neighbor)
            if cost < min_cost:
                min_cost = cost
                next_node = neighbor
        
        if next_node is None or next_node in visited or min_cost == float('inf'):
            break
        curr = next_node
        visited.add(curr)
        max_steps -= 1
        
    if curr == dstar.s_goal:
        path.append(dstar.s_goal)
        
        # Apply smoothing
        if len(path) > 3:
             return path_smoothing(path)
        return path
    else:
        return []

# Main Loop
clock = pygame.time.Clock()
run = True
update_count = 0
pygame.font.init()
font = pygame.font.SysFont('Arial', 24)

while run:
    screen.blit(bg_surf, (0, 0))
    
    # Draw transparent buffers
    obst_surf.fill((0, 0, 0, 0)) # Clear
    for obs in dstar.obstacles:
        # Reddish transparent fill (Alpha = 80)
        pygame.draw.rect(obst_surf, (200, 0, 0, 80), (obs[0], obs[1], step, step))
    screen.blit(obst_surf, (0, 0))

    # Zapobieganie błędom ze smart/start i błąd - komunikaty
    start_in_obs = dstar.s_start in dstar.obstacles
    goal_in_obs = dstar.s_goal in dstar.obstacles

    if start_in_obs or goal_in_obs:
        warning_surf = font.render("Błąd: Start lub cel w obszarze przeszkody!", True, (255, 0, 0))
        screen.blit(warning_surf, (20, 50))
    else:
        # Pathfinding
        path = get_path()
        if len(path) > 1:
            pygame.draw.lines(screen, (0, 255, 0), False, path, 6)

    # Render Update Count
    text_surf = font.render(f"Liczba aktualizacji: {update_count}", True, (255, 255, 0)) # Yellow color
    screen.blit(text_surf, (20, 20))

    for event in pygame.event.get():
        if event.type == QUIT:
            run = False
        if event.type == MOUSEBUTTONDOWN:
            m_pos = pygame.mouse.get_pos()
            current_obs = set(dstar.obstacles)
            changed = False
            if event.button == 1: # Left click adds
                new_obs = add_obstacle_cluster(m_pos, radius=25)
                if not new_obs.issubset(current_obs):
                    current_obs.update(new_obs)
                    changed = True
            elif event.button == 3: # Right click removes
                to_remove = add_obstacle_cluster(m_pos, radius=25)
                if not current_obs.isdisjoint(to_remove):
                    current_obs.difference_update(to_remove)
                    changed = True
            
            if changed:
                old_path = get_path()
                dstar.update_obstacles(current_obs)
                new_path = get_path()
                
                # Zwiększ licznik tylko, jeśli trasa rzeczywiście uległa zmianie
                if old_path != new_path:
                    update_count += 1

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
