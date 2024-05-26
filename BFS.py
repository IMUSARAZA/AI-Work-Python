graph = {
    'A': ['B', 'C', 'D'],
    'B': ['E', 'F'],
    'C': ['G',  'H'],
    'D': [None, 'I'],  # No left edge, right edge is 'I'
    'E': ['J', 'K'],
    'F': [None, None],  # No left or right edge
    'G': ['L', None],   # No right edge
    'H': [None, None],  # No left or right edge
    'I': ['M', None],   # No right edge
    'J': [None, None],  # No left or right edge
    'K': ['N', None],   # No right edge
    'L': [None, None],  # No left or right edge
    'M': [None, None],  # No left or right edge
}


visited = []  # List for visited nodes.
queue = []  #Initialize a queue


def bfs(visited, graph, node):   # function for bfs
    visited.append(node)
    queue.append(node)

    while queue:
        m = queue.pop(0)
        print(m, end=" ")

        for neighbour in graph[m]:
            if neighbour is not None and neighbour not in visited and neighbour in graph:  # Check if neighbour is a valid node
                visited.append(neighbour)
                queue.append(neighbour)

 



                
print("Following is the Breath-First Search")
bfs(visited, graph, 'A')