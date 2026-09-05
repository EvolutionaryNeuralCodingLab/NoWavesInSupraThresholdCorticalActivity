import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from numpy.linalg import norm


def count_nodes_by_connections_original(nodes_connections):
    connection_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    for node, connections in nodes_connections.items():
        if connections <= 4:
            connection_counts[connections] += 1

    return connection_counts


def nodes_connections_original(vector_field, theta, plot):
    M, N, _ = vector_field.shape
    G = nx.DiGraph()
    all_edges = nx.DiGraph()

    cos_theta = np.cos(np.radians(theta))
    nodes_connections = {}
    all_connections = {}

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    norms = np.linalg.norm(vector_field, axis=2)  # Precompute all norms

    for i in range(M):
        for j in range(N):
            current_node = (i, j)
            G.add_node(current_node)
            all_edges.add_node(current_node)
            connections, all_con = 0, 0

            v1 = vector_field[i, j]
            v1_norm = norms[i, j]

            if v1_norm == 0:
                nodes_connections[current_node] = connections
                all_connections[current_node] = all_con
                continue

            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < M and 0 <= nj < N:
                    neighbor_node = (ni, nj)
                    v2 = vector_field[ni, nj]
                    v2_norm = norms[ni, nj]
                    dot_product = np.dot(v1, v2)


                    if v2_norm > 0:
                        cos_angle = dot_product / (v1_norm * v2_norm)

                        if cos_angle >= cos_theta:
                            G.add_edge(current_node, neighbor_node, weight=dot_product)
                            connections += 1
                        all_edges.add_edge(current_node, neighbor_node, weight=dot_product)
                        all_con += 1

            nodes_connections[current_node] = connections
            all_connections[current_node] = all_con

    node_counts = count_nodes_by_connections(nodes_connections)
    theo_counts = count_nodes_by_connections(all_connections)



    if plot == 'yes':
        filtered_edges = [(u, v) for u, v, d in G.edges(data=True)]
        filtered_graph = G.edge_subgraph(filtered_edges)
        theoretical_edges = [(u, v) for u, v, d in all_edges.edges(data=True)]
        theoretical_graph = all_edges.edge_subgraph(theoretical_edges)

        fig, ax = plt.subplots(figsize=(6, 6))
        pos = {node: node for node in G.nodes()}
        X, Y = np.meshgrid(np.arange(N), np.arange(M))
        ax.set_title('Vector Field')
        ax.set_xlim(-0.25, M - 0.5)
        ax.set_ylim(-0.25, N - 0.5)

        nx.draw_networkx_edges(theoretical_graph, pos, node_size=50, edge_color='tomato', alpha=0.15, width=2, ax=ax, arrows=False)
        nx.draw_networkx_nodes(theoretical_graph, pos, node_size=30, ax=ax, node_color='black')
        nx.draw(filtered_graph.reverse(), pos, with_labels=False, node_size=30, node_color='black', edge_color='tomato', alpha=1, width=4, ax=ax, arrows=False)
        ax.quiver(X, M - Y - 1, vector_field[:, :, 0], -vector_field[:, :, 1], scale=30)
        plt.tight_layout()

    return node_counts, theo_counts



def count_nodes_by_connections(nodes_connections):
    connection_counts = {i: 0 for i in range(9)}  # Initialize counts for 0-8 connections

    for connections in nodes_connections.values():
        connection_counts[connections] += 1  # Count exact connections

    # Convert to cumulative count (nodes with at least k connections)
    cumulative_counts = {}
    total = 0
    for k in range(8, -1, -1):  # Start from 8 and go down to 0
        total += connection_counts[k]
        cumulative_counts[k] = total

    return cumulative_counts


def nodes_connections(vector_field, theta, plot):
    M, N, _ = vector_field.shape
    G = nx.DiGraph()
    all_edges = nx.DiGraph()

    cos_theta = np.cos(np.radians(theta))
    nodes_connections = {}
    all_connections = {}

    directions = [(-1, 1), (1, -1), (1, 1), (-1, -1),
                  (2,0) , (-2,0) , (0,2) , (0,-2)]
    norms = np.linalg.norm(vector_field, axis=2)  # Precompute all norms

    for i in range(2,M-2):
        for j in range(2,N-2):
            current_node = (i, j)
            G.add_node(current_node)
            all_edges.add_node(current_node)
            connections, all_con = 0, 0

            v1 = vector_field[i, j]
            v1_norm = norms[i, j]

            if v1_norm == 0:
                nodes_connections[current_node] = connections
                all_connections[current_node] = all_con
                continue

            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < M and 0 <= nj < N:
                    neighbor_node = (ni, nj)
                    v2 = vector_field[ni, nj]
                    v2_norm = norms[ni, nj]
                    dot_product = np.dot(v1, v2)


                    if v2_norm > 0:
                        cos_angle = dot_product / (v1_norm * v2_norm)

                        if cos_angle >= cos_theta:
                            G.add_edge(current_node, neighbor_node, weight=dot_product)
                            connections += 1
                        all_edges.add_edge(current_node, neighbor_node, weight=dot_product)
                        all_con += 1

            nodes_connections[current_node] = connections
            all_connections[current_node] = all_con

    node_counts = count_nodes_by_connections(nodes_connections)
    theo_counts = count_nodes_by_connections(all_connections)



    if plot == 'yes':
        filtered_edges = [(u, v) for u, v, d in G.edges(data=True)]
        filtered_graph = G.edge_subgraph(filtered_edges)
        theoretical_edges = [(u, v) for u, v, d in all_edges.edges(data=True)]
        theoretical_graph = all_edges.edge_subgraph(theoretical_edges)

        fig, ax = plt.subplots(figsize=(6, 6))
        pos = {node: node for node in G.nodes()}
        X, Y = np.meshgrid(np.arange(N), np.arange(M))
        ax.set_title('Vector Field')
        ax.set_xlim(-0.25, M - 0.5)
        ax.set_ylim(-0.25, N - 0.5)

        nx.draw_networkx_edges(theoretical_graph, pos, node_size=50, edge_color='tomato', alpha=0.15, width=2, ax=ax, arrows=False)
        nx.draw_networkx_nodes(theoretical_graph, pos, node_size=30, ax=ax, node_color='black')
        nx.draw(filtered_graph.reverse(), pos, with_labels=False, node_size=30, node_color='black', edge_color='tomato', alpha=1, width=4, ax=ax, arrows=False)
        ax.quiver(X, M - Y - 1, vector_field[:, :, 0], -vector_field[:, :, 1], scale=30)
        plt.tight_layout()

    return node_counts, theo_counts



# Example usage:
vector_field=(np.random.rand(4,4,2)-0.1)/2
vector_field[vector_field<0.4]=0
#vector_field = np.array([[[0, 0], [0, 0], [0.33142296, 0.52608871],[0, 0]],
#    [[0, 0], [0.43258202, 0.21398492], [0.33142296, 0.52608871],[0.33142296, 0.52608871]],
#    [[0.47284963, -0.20674897], [0.53551432, 0.23384063], [0.22213769, 0.35897794],[0.33142296, 0.52608871]],
#    [[0.56552441, -0.19352592], [-0.41081992, 0.4015286], [-0.45122815, -0.361123],[0.33142296, 0.52608871]]])

#nodes_connections(vector_field, theta=30)



#print('average cluster: ',x)

#find_longest_path_2d(vector_field, theta=10)