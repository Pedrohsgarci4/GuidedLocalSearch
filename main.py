import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

def bfs(G, source, target):
    visited = set()
    queue = [source]
    predecessors = {}

    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.add(vertex)
            if G.neighbors(vertex):
                neighbors = list(G.neighbors(vertex))
                random.shuffle(neighbors)
            for neighbor in neighbors:
                if neighbor not in visited:
                    predecessors[neighbor] = vertex
                    queue.append(neighbor)
                if neighbor == target:
                    queue = []
                    break

    path = [target]
    current = target

    while True:
        try:

            path.append(predecessors[current])
            current = predecessors[current]
        except KeyError:
            return list(reversed(path))

def calculate_route_cost(path, graph, penalty_factor):
    cost = 0
    for i in range(len(path) - 1):
        edge = (path[i], path[i + 1])
        weight = graph[edge[0]][edge[1]]['weight']
        penalty = graph[edge[0]][edge[1]]['penalty']
        cost += weight + penalty * penalty_factor
    return cost

def calculate_route_distance(path, graph):
    return sum(graph[path[i]][path[i + 1]]['weight'] for i in range(len(path) - 1))

def select_node_based_on_penalty(nodes, penalties):
    inv_penalties = [1 / (penalties.get(node, 1) + 1) for node in nodes.keys()]
    total = sum(inv_penalties)
    probabilities = [p / total for p in inv_penalties]
    return random.choices(list(nodes.keys()), weights=probabilities, k=1)[0]

def apply_penalties(path, graph):
    utilities = []
    for i in range(len(path) - 1):
        edge = (path[i], path[i + 1])
        utility = graph[edge[0]][edge[1]]['weight'] / (1 + graph[edge[0]][edge[1]]['penalty'])
        utilities.append((utility, edge))
    utilities.sort(reverse=True, key=lambda x: x[0])
    num_top_edges = max(1, int(len(utilities) * 0.2))
    top_edges = [edge for _, edge in utilities[:num_top_edges]]

    for edge in top_edges:
        u, v = edge
        graph[u][v]['penalty'] += 10

def initialize_solution(path, graph, target_nodes):
    if not target_nodes:
        return path
    if not path:
        path.append(target_nodes[0])
    remaining_nodes = set(target_nodes[1:])

    while remaining_nodes:
        current_node = path[-1]
        reachable_nodes = [n for n in remaining_nodes if nx.has_path(graph, current_node, n)]
        if not reachable_nodes:
            break
        next_node = reachable_nodes.pop()
        path.extend(bfs(graph, current_node, next_node)[1:])
        remaining_nodes.remove(next_node)
    return path

def complete_solution(graph, new_path, target, old_path, target_nodes):
    last_node = new_path[-1]
    path = bfs(graph, last_node, target=target)[1:]
    new_path = new_path + path + old_path[1:]

    i = 0
    while target_nodes:
        if new_path[i] in target_nodes:
            target_nodes.remove(new_path[i])
        i += 1
    return new_path[:i]

def local_search(graph, path, target_nodes, penalty_factor, current_solution):
    best_route = path[:]
    best_cost = calculate_route_cost(best_route, graph, penalty_factor)

    utilities = []
    for i in range(len(path) - 1):
        edge = (path[i], path[i + 1])
        utility = graph[edge[0]][edge[1]]['weight'] / (1 + graph[edge[0]][edge[1]]['penalty'])
        utilities.append(utility)

    max_utility_index = utilities.index(np.random.choice(sorted(utilities, reverse=True)[:int(len(utilities) * 0.5)]))
    node = path[max_utility_index]

    new_route = best_route[:max_utility_index + 1]
    node = select_node_based_on_penalty(graph[node], {n: graph[node][n]['penalty'] for n in graph[node]})

    index_target_node = 0
    v = [n for n in best_route if node not in new_route and node in target_nodes]

    for i in range(len(path)):
        if i > max_utility_index and v == path[i]:
            index_target_node = i

    new_route += [node]
    new_route = complete_solution(graph, new_route, best_route[index_target_node], best_route[index_target_node:], target_nodes.copy())

    new_cost = calculate_route_cost(new_route, graph, penalty_factor)

    if calculate_route_distance(current_solution, graph) > calculate_route_distance(new_route, graph) and len([n for n in target_nodes if n in new_route]) == len(target_nodes):
        current_solution = new_route.copy()

    if new_cost < best_cost:
        best_route = new_route
        best_cost = new_cost
    else:
        apply_penalties(best_route, graph)

    return best_route, best_cost

def guided_local_search(graph, target_nodes, max_iterations=500, penalty_factor=10):
    current_solution = initialize_solution([], graph, target_nodes)
    for edge in graph.edges():
        graph[edge[0]][edge[1]]['penalty'] = 0


    best_route = current_solution[:]
    best_cost = calculate_route_cost(best_route, graph, penalty_factor)

    
    i = 0
    for _ in range(max_iterations):
        new_route, new_cost = local_search(graph, best_route, target_nodes, penalty_factor, current_solution)
        print(new_route)
        if new_cost < best_cost:
            best_route = new_route
            best_cost = new_cost
        else:
            i += 1
        if i > 20:
            i = 0
            best_route = initialize_solution([], graph, target_nodes)
            best_cost = calculate_route_cost(best_route, graph, penalty_factor)

    return best_route, calculate_route_distance(best_route, graph)

# Geração do grafo aleatório
num_nos = 30
num_arestas = 4 * num_nos
g = nx.gnm_random_graph(num_nos, num_arestas)

# Adicionar pesos aleatórios às arestas
for (u, v) in g.edges():
    g[u][v]['weight'] = np.random.randint(1, 10)

# Configuração do layout com ajuste no parâmetro k
pos = nx.spring_layout(g, k=0.5, iterations=50)

# Aumente o tamanho da figura
plt.figure(figsize=(10, 8))

# Desenhe o gráfico
nx.draw(g, pos, with_labels=True, node_color="lightgreen", edge_color="gray", node_size=200)
labels = nx.get_edge_attributes(g, 'weight')
nx.draw_networkx_edge_labels(g, pos, edge_labels=labels)

# Salve a figura
plt.savefig('graph.png')

target_nodes = list(g.nodes())
random.shuffle(target_nodes)
target_nodes = target_nodes[:5]

print('Alvos: ', target_nodes)

best_route, best_cost = guided_local_search(g, target_nodes, max_iterations=200, penalty_factor=3)

print("Melhor rota encontrada:", best_route)
print("Custo da melhor rota:", best_cost)
