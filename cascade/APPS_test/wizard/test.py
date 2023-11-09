import math
from typing import List, Tuple
import random
def solution(stdin: str) -> str:

    def find_parent(parent, u):
        if parent[u] != u:
            parent[u] = find_parent(parent, parent[u])
        return parent[u]

    def kruskal(edges, n):
        parent = list(range(n + 1))
        edges.sort(key=lambda x: x[2])
        mst = []
        for edge in edges:
            u, v, w = edge
            if find_parent(parent, u) != find_parent(parent, v):
                mst.append(edge)
                union(parent, u, v)
                if len(mst) == n - 1:
                    break
        return mst

    def union(parent, u, v):
        parent[find_parent(parent, u)] = find_parent(parent, v)

    def simple_paths(mst, a, b, c):
        paths = []
        for edge in mst:
            u, v, w = edge
            if a in (u, v) and b in (u, v) and c in (u, v):
                paths.append(edge)
        return paths

    def max_edges(paths):
        return len(paths)

    def find_vertices(n):
        a, b, c = random.sample(range(1, n + 1), 3)
        while a == b or a == c or b == c:
            a, b, c = random.sample(range(1, n + 1), 3)
        return a, b, c

    def main(n, edges):
        mst = kruskal(edges, n)
        a, b, c = find_vertices(n)
        paths = simple_paths(mst, a, b, c)
        res = max_edges(paths)
        return res, a, b, c

    def parse_input(stdin: str) -> tuple:
        lines = stdin.split('\n')
        n = int(lines[0])
        edges = [tuple(map(int, line.split())) for line in lines[1:]]
        return n, edges

    n, edges = parse_input(stdin)
    res, a, b, c = main(n, edges)
    return f"{res}\n{a} {b} {c}"

print(solution('3\n1 2\n1 3'))

def check(candidate):
    assert candidate('8\n1 2\n2 3\n3 4\n4 5\n4 6\n3 7\n3 8') == '5\n1 8 6'
    assert candidate('5\n3 2\n3 5\n2 4\n1 3') == '4\n5 1 4'
    assert candidate('4\n1 2\n2 3\n3 4') == '3 \n1 2 4'
    assert candidate('6\n1 2\n1 3\n2 4\n2 5\n3 6') == '5\n5 4 6'
    assert candidate('4\n4 3\n3 1\n1 2') == '3 \n2 1 4'
    assert candidate('3\n1 2\n1 3') == '2 \n2 1 3'
    assert candidate('3\n1 2\n1 3') == '2 \n2 1 3'
    assert candidate('3\n3 1\n1 2') == '2 \n2 1 3'
    assert candidate('3\n1 3\n1 2') == '2 \n2 1 3'
    assert candidate('3\n1 2\n1 3') == '2 \n2 1 3'
    assert candidate('3\n1 2\n1 3') == '2 \n2 1 3'
    assert candidate('3\n3 1\n3 2') == '2 \n1 3 2'
    assert candidate('3\n1 3\n2 1') == '2 \n2 1 3'
    assert candidate('3\n1 3\n1 2') == '2 \n2 1 3'
    assert candidate('3\n3 1\n2 1') == '2 \n2 1 3'
    assert candidate('3\n1 2\n1 3') == '2 \n2 1 3'
    assert candidate('3\n2 1\n3 1') == '2 \n2 1 3'
    assert candidate('3\n2 3\n3 1') == '2 \n1 3 2'
    assert candidate('3\n1 3\n3 2') == '2 \n1 3 2'
    assert candidate('3\n1 3\n1 2') == '2 \n2 1 3'
    assert candidate('3\n3 2\n1 3') == '2 \n1 3 2'
    assert candidate('3\n1 2\n3 1') == '2 \n2 1 3'
    assert candidate('3\n1 2\n1 3') == '2 \n2 1 3'
    assert candidate('3\n3 2\n2 1') == '2 \n1 2 3'
    assert candidate('3\n3 2\n3 1') == '2 \n1 3 2'
    assert candidate('3\n1 2\n2 3') == '2 \n1 2 3'
    assert candidate('3\n3 1\n3 2') == '2 \n1 3 2'
    assert candidate('3\n2 3\n1 3') == '2 \n1 3 2'
    assert candidate('3\n1 2\n3 2') == '2 \n1 2 3'
    assert candidate('3\n3 2\n3 1') == '2 \n1 3 2'
    assert candidate('3\n3 2\n1 2') == '2 \n1 2 3'
    assert candidate('3\n2 3\n2 1') == '2 \n1 2 3'
    assert candidate('3\n3 2\n1 3') == '2 \n1 3 2'
    assert candidate('3\n2 3\n3 1') == '2 \n1 3 2'
    assert candidate('3\n2 3\n2 1') == '2 \n1 2 3'
    assert candidate('3\n1 2\n2 3') == '2 \n1 2 3'
    assert candidate('3\n2 3\n3 1') == '2 \n1 3 2'
    assert candidate('3\n3 2\n3 1') == '2 \n1 3 2'
    assert candidate('3\n2 3\n2 1') == '2 \n1 2 3'
    assert candidate('3\n2 3\n3 1') == '2 \n1 3 2'
    assert candidate('4\n1 3\n1 2\n1 4') == '3\n3 2 4'
    assert candidate('4\n1 3\n3 4\n2 3') == '3\n2 1 4'
    assert candidate('4\n1 2\n1 3\n1 4') == '3\n3 2 4'
    assert candidate('4\n1 4\n1 3\n2 1') == '3\n3 2 4'
    assert candidate('4\n4 1\n1 2\n3 1') == '3\n3 2 4'
    assert candidate('4\n3 1\n4 1\n2 1') == '3\n3 2 4'
    assert candidate('4\n4 1\n1 2\n1 3') == '3\n3 2 4'
    assert candidate('4\n1 4\n3 1\n1 2') == '3\n3 2 4'
    assert candidate('4\n1 3\n4 1\n3 2') == '3 \n4 1 2'
    assert candidate('4\n1 4\n3 4\n1 2') == '3 \n2 1 3'
    assert candidate('4\n2 1\n1 4\n2 3') == '3 \n4 1 3'
    assert candidate('4\n4 1\n3 1\n1 2') == '3\n3 2 4'
    assert candidate('4\n2 1\n3 1\n4 1') == '3\n3 2 4'
    assert candidate('4\n1 3\n1 4\n3 2') == '3 \n4 1 2'
    assert candidate('4\n4 1\n1 2\n3 1') == '3\n3 2 4'
    assert candidate('4\n4 1\n4 2\n3 1') == '3 \n3 1 2'
    assert candidate('4\n1 4\n3 2\n3 4') == '3 \n1 4 2'
    assert candidate('4\n3 2\n1 2\n4 1') == '3 \n4 1 3'
    assert candidate('4\n2 1\n1 4\n2 3') == '3 \n4 1 3'
    assert candidate('4\n3 4\n2 4\n3 1') == '3 \n1 3 2'
    assert candidate('4\n1 4\n3 4\n2 4') == '3\n2 1 3'
    assert candidate('4\n3 1\n2 3\n2 4') == '3 \n1 3 4'
    assert candidate('4\n2 3\n1 2\n1 4') == '3 \n4 1 3'
    assert candidate('4\n4 1\n2 3\n1 3') == '3 \n4 1 2'
    assert candidate('4\n2 1\n2 3\n4 3') == '3 \n1 2 4'
    assert candidate('4\n4 1\n2 4\n1 3') == '3 \n3 1 2'
    assert candidate('4\n4 2\n1 3\n3 4') == '3 \n1 3 2'
    assert candidate('4\n2 3\n4 2\n1 3') == '3 \n1 3 4'
    assert candidate('4\n2 3\n1 4\n4 2') == '3 \n1 4 3'
    assert candidate('4\n3 4\n2 4\n1 4') == '3\n2 1 3'
    assert candidate('4\n2 3\n3 1\n2 4') == '3 \n1 3 4'
    assert candidate('4\n2 3\n4 1\n4 3') == '3 \n1 4 2'
    assert candidate('4\n2 3\n3 1\n4 2') == '3 \n1 3 4'
    assert candidate('4\n2 1\n4 3\n3 2') == '3 \n1 2 4'
    assert candidate('4\n4 1\n4 2\n3 4') == '3\n2 1 3'
    assert candidate('5\n1 3\n3 2\n5 1\n1 4') == '4\n5 4 2'
    assert candidate('5\n1 3\n1 2\n4 1\n1 5') == '3\n4 3 5'
    assert candidate('5\n1 5\n4 1\n1 2\n1 3') == '3\n4 3 5'
    assert candidate('5\n1 2\n1 4\n5 1\n1 3') == '3\n4 3 5'
    assert candidate('5\n2 1\n3 1\n4 1\n5 1') == '3\n4 3 5'
    assert candidate('5\n5 1\n4 2\n4 1\n1 3') == '4\n5 3 2'
    assert candidate('5\n3 1\n2 1\n1 5\n4 1') == '3\n4 3 5'
    assert candidate('5\n1 5\n3 1\n1 4\n2 1') == '3\n4 3 5'
    assert candidate('5\n2 3\n1 5\n4 1\n3 1') == '4\n5 4 2'
    assert candidate('5\n1 2\n4 1\n3 1\n1 5') == '3\n4 3 5'
    assert candidate('5\n5 4\n1 4\n1 2\n3 4') == '4\n2 3 5'
    assert candidate('5\n5 1\n5 4\n4 3\n1 2') == '4 \n2 1 3'
    assert candidate('5\n2 4\n4 1\n1 3\n1 5') == '4\n5 3 2'
    assert candidate('5\n3 1\n2 1\n1 5\n1 4') == '3\n4 3 5'
    assert candidate('5\n1 4\n2 1\n5 1\n5 3') == '4\n4 2 3'
    assert candidate('5\n1 5\n3 4\n2 3\n1 3') == '4\n5 2 4'
    assert candidate('5\n2 5\n5 3\n1 3\n2 4') == '4 \n1 3 4'
    assert candidate('5\n3 1\n5 1\n2 1\n4 2') == '4\n5 3 4'
    assert candidate('5\n1 3\n5 3\n2 1\n1 4') == '4\n4 2 5'
    assert candidate('5\n3 5\n3 1\n2 1\n4 3') == '4\n2 4 5'
    assert candidate('5\n2 3\n2 1\n1 5\n4 2') == '4\n5 3 4'
    assert candidate('5\n5 2\n4 5\n1 2\n3 5') == '4\n1 3 4'
    assert candidate('5\n5 3\n2 4\n1 2\n4 3') == '4 \n1 2 5'
    assert candidate('5\n4 1\n1 5\n3 5\n2 5') == '4\n4 2 3'
    assert candidate('5\n5 3\n4 2\n1 2\n2 5') == '4\n4 1 3'
    assert candidate('5\n5 1\n2 3\n5 4\n3 5') == '4\n4 1 2'
    assert candidate('5\n4 5\n1 5\n2 5\n2 3') == '4\n4 1 3'
    assert candidate('5\n5 1\n4 2\n5 4\n3 5') == '4\n3 1 2'
    assert candidate('5\n1 5\n2 3\n2 5\n3 4') == '4 \n1 5 4'
    assert candidate('5\n2 3\n4 1\n5 4\n5 2') == '4 \n1 4 3'
    assert candidate('5\n5 4\n2 3\n2 5\n1 3') == '4 \n1 3 4'
    assert candidate('5\n5 4\n5 2\n1 3\n4 3') == '4 \n1 3 2'
    assert candidate('5\n2 5\n2 1\n3 2\n4 3') == '4\n5 1 4'
    assert candidate('5\n4 5\n2 1\n3 2\n5 3') == '4 \n1 2 4'
    assert candidate('5\n5 2\n4 3\n1 2\n2 4') == '4\n5 1 3'
    assert candidate('6\n4 2\n5 1\n6 1\n2 1\n1 3') == '4\n6 5 4'
    assert candidate('6\n2 1\n1 4\n1 6\n1 3\n5 1') == '3\n5 4 6'
    assert candidate('6\n5 1\n1 4\n2 3\n1 6\n1 3') == '4\n6 5 2'
    assert candidate('6\n1 5\n1 3\n6 1\n4 1\n1 2') == '3\n5 4 6'
    assert candidate('6\n1 2\n5 3\n3 4\n1 3\n6 3') == '4\n2 5 6'
    assert candidate('6\n2 4\n4 1\n4 3\n5 4\n1 6') == '4\n6 3 5'
    assert candidate('6\n2 1\n5 1\n1 4\n6 1\n1 3') == '3\n5 4 6'
    assert candidate('6\n1 3\n5 1\n2 4\n1 2\n1 6') == '4\n6 5 4'
    assert candidate('6\n1 5\n1 6\n4 1\n1 2\n3 1') == '3\n5 4 6'
    assert candidate('6\n5 6\n4 1\n3 1\n6 1\n2 6') == '4\n4 3 5'
    assert candidate('6\n6 1\n5 4\n1 3\n1 5\n5 2') == '4\n6 3 4'
    assert candidate('6\n1 6\n6 3\n4 3\n5 6\n2 1') == '5\n2 5 4'
    assert candidate('6\n2 3\n1 6\n1 3\n1 4\n5 6') == '5\n2 4 5'
    assert candidate('6\n5 1\n3 1\n4 1\n1 6\n2 6') == '4\n5 4 2'
    assert candidate('6\n2 1\n1 6\n3 1\n4 2\n2 5') == '4\n6 4 5'
    assert candidate('6\n1 3\n1 6\n6 4\n5 1\n2 5') == '5\n2 3 4'
    assert candidate('6\n2 5\n4 1\n2 1\n1 6\n3 4') == '5\n3 6 5'
    assert candidate('6\n2 4\n6 1\n5 1\n6 2\n3 6') == '5\n5 3 4'
    assert candidate('6\n1 5\n1 6\n6 2\n1 4\n4 3') == '5\n2 5 3'
    assert candidate('6\n1 4\n5 3\n3 1\n2 5\n1 6') == '5\n6 4 2'
    assert candidate('6\n5 3\n5 4\n2 1\n2 4\n6 4') == '5\n1 6 3'
    assert candidate('6\n4 3\n5 6\n6 1\n5 2\n6 3') == '5\n2 1 4'
    assert candidate('6\n5 3\n3 1\n5 6\n4 5\n2 3') == '4\n2 4 6'
    assert candidate('6\n4 3\n4 2\n2 1\n2 6\n5 4') == '4\n6 3 5'
    assert candidate('6\n3 6\n6 2\n6 5\n2 1\n6 4') == '4\n1 4 5'
    assert candidate('6\n5 4\n5 3\n2 4\n5 6\n2 1') == '5\n1 3 6'
    assert candidate('6\n6 3\n4 2\n6 5\n4 1\n2 5') == '5 \n1 4 3'
    assert candidate('6\n5 3\n4 6\n6 2\n1 2\n2 3') == '5\n4 1 5'
    assert candidate('6\n1 4\n6 2\n2 5\n3 5\n2 4') == '5\n1 6 3'
    assert candidate('6\n2 4\n3 6\n5 1\n3 5\n6 2') == '5 \n1 5 4'
    assert candidate('6\n5 4\n6 2\n1 5\n4 6\n5 3') == '5\n3 1 2'
    assert candidate('6\n6 2\n3 5\n4 2\n5 4\n1 2') == '5\n6 1 3'
    assert candidate('6\n1 4\n2 3\n5 6\n5 3\n2 4') == '5 \n1 4 6'
    assert candidate('6\n6 4\n4 1\n5 6\n3 2\n4 2') == '5\n3 1 5'
    assert candidate('6\n6 1\n5 6\n5 3\n2 5\n2 4') == '5\n1 3 4'
    assert candidate('7\n1 7\n6 1\n5 1\n2 1\n3 2\n2 4') == '4\n7 6 4'
    assert candidate('7\n3 1\n6 1\n4 1\n1 5\n5 2\n1 7') == '4\n7 6 2'
    assert candidate('7\n7 1\n5 1\n1 3\n1 2\n6 1\n4 7') == '4\n6 5 4'
    assert candidate('7\n1 7\n2 1\n1 5\n4 1\n3 1\n5 6') == '4\n7 4 6'
    assert candidate('7\n3 1\n1 5\n1 7\n1 4\n6 1\n2 1') == '3\n6 5 7'
    assert candidate('7\n1 4\n7 4\n5 1\n3 2\n6 4\n3 1') == '5\n2 6 7'
    assert candidate('7\n1 3\n6 1\n1 7\n1 4\n5 4\n1 2') == '4\n7 6 5'
    assert candidate('7\n5 1\n6 1\n2 1\n1 3\n1 7\n1 4') == '3\n6 5 7'
    assert candidate('7\n5 1\n5 7\n1 2\n5 6\n3 1\n4 5') == '4\n3 6 7'
    assert candidate('7\n1 4\n6 1\n2 1\n7 5\n1 7\n1 3') == '4\n6 4 5'
    assert candidate('7\n1 2\n7 3\n1 6\n5 1\n2 7\n4 6') == '6\n4 5 3'
    assert candidate('7\n2 6\n5 6\n4 1\n1 7\n1 6\n3 5') == '5\n7 4 3'
    assert candidate('7\n3 2\n1 4\n3 1\n1 6\n7 1\n5 4') == '5\n2 7 5'
    assert candidate('7\n1 5\n4 1\n2 1\n2 3\n1 7\n6 2') == '4\n7 5 6'
    assert candidate('7\n7 2\n3 4\n4 2\n7 1\n6 7\n5 7') == '5\n6 5 3'
    assert candidate('7\n1 6\n3 1\n5 6\n1 4\n1 2\n1 7') == '4\n7 4 5'
    assert candidate('7\n2 7\n4 6\n7 1\n5 1\n3 1\n7 6') == '5\n5 3 4'
    assert candidate('7\n3 2\n5 1\n3 5\n2 7\n3 6\n4 2') == '5\n1 6 7'
    assert candidate('7\n4 1\n4 3\n7 5\n1 6\n7 4\n2 6') == '6\n2 3 5'
    assert candidate('7\n5 3\n2 4\n6 1\n4 7\n3 1\n2 3') == '6\n6 5 7'
    assert candidate('7\n1 7\n6 5\n4 3\n7 4\n2 5\n5 4') == '5\n1 3 6'
    assert candidate('7\n1 4\n3 6\n6 7\n3 5\n4 2\n2 7') == '6 \n1 4 5'
    assert candidate('7\n2 7\n2 4\n1 3\n5 6\n5 3\n3 2') == '5\n6 4 7'
    assert candidate('7\n4 2\n6 3\n5 1\n6 1\n7 4\n6 4') == '5\n5 3 7'
    assert candidate('7\n3 2\n1 7\n7 2\n6 5\n6 1\n4 5') == '6 \n3 2 4'
    assert candidate('7\n3 2\n3 5\n4 7\n3 6\n4 5\n5 1') == '5\n6 2 7'
    assert candidate('7\n7 6\n7 1\n4 2\n4 5\n7 3\n5 3') == '6\n6 1 2'
    assert candidate('7\n1 2\n3 6\n6 5\n4 3\n7 5\n2 6') == '6\n4 1 7'
    assert candidate('7\n2 7\n6 7\n5 4\n1 2\n5 3\n3 6') == '6 \n1 2 4'
    assert candidate('7\n1 5\n4 2\n5 3\n6 5\n7 2\n1 2') == '5\n6 4 7'
    assert candidate('7\n5 2\n3 6\n7 1\n3 7\n3 4\n2 6') == '6\n1 4 5'
    assert candidate('7\n1 5\n3 5\n7 2\n7 6\n3 7\n4 5') == '5\n4 2 6'
    assert candidate('7\n7 6\n2 1\n6 5\n3 2\n3 6\n7 4') == '6\n1 5 4'
    assert candidate('7\n7 6\n2 3\n3 6\n5 4\n4 2\n1 5') == '6 \n1 5 7'
    assert candidate('7\n2 4\n7 1\n6 5\n3 6\n2 7\n7 6') == '5\n4 3 5'
    assert candidate('8\n1 5\n1 8\n7 1\n1 6\n1 2\n4 2\n1 3') == '4\n8 7 4'
    assert candidate('8\n6 1\n2 1\n5 4\n8 1\n7 3\n7 1\n4 7') == '5\n8 6 5'
    assert candidate('8\n6 8\n1 8\n8 2\n1 7\n5 7\n1 3\n4 1') == '5\n5 4 6'
    assert candidate('8\n2 1\n6 1\n4 1\n7 1\n1 3\n1 5\n1 8') == '3\n7 6 8'
    assert candidate('8\n1 8\n2 1\n1 7\n1 5\n1 3\n4 8\n1 6') == '4\n7 6 4'
    assert candidate('8\n2 1\n2 8\n7 8\n5 1\n2 6\n3 1\n4 6') == '6\n5 4 7'
    assert candidate('8\n6 4\n7 6\n1 5\n1 3\n1 6\n8 1\n2 7') == '5\n8 5 2'
    assert candidate('8\n8 5\n6 4\n2 4\n1 5\n1 7\n1 4\n4 3') == '5\n6 7 8'
    assert candidate('8\n6 5\n7 4\n1 3\n8 7\n1 7\n2 1\n6 1') == '5\n5 4 8'
    assert candidate('8\n7 3\n2 8\n4 1\n1 3\n2 1\n6 3\n5 1') == '5\n7 6 8'
    assert candidate('8\n1 4\n6 5\n1 6\n7 1\n2 1\n3 6\n5 8') == '5\n7 4 8'
    assert candidate('8\n4 8\n4 6\n1 3\n7 4\n7 5\n1 4\n1 2') == '5\n3 8 5'
    assert candidate('8\n7 8\n5 4\n8 1\n8 6\n1 2\n4 3\n1 4') == '5\n5 6 7'
    assert candidate('8\n2 3\n6 5\n1 7\n2 1\n4 6\n8 4\n1 6') == '6\n3 7 8'
    assert candidate('8\n5 3\n4 2\n6 5\n8 1\n5 1\n7 8\n1 2') == '6\n6 4 7'
    assert candidate('8\n7 3\n5 6\n6 1\n7 4\n6 2\n2 8\n6 4') == '6\n8 5 3'
    assert candidate('8\n8 3\n4 2\n4 1\n1 7\n1 3\n6 2\n8 5') == '7\n5 7 6'
    assert candidate('8\n5 1\n1 6\n6 3\n4 3\n5 2\n1 7\n3 8') == '6\n2 7 8'
    assert candidate('8\n5 1\n7 2\n4 3\n8 5\n7 1\n4 6\n5 4') == '6\n2 8 6'
    assert candidate('8\n8 7\n3 2\n1 3\n6 4\n1 8\n1 4\n8 5') == '6\n6 2 7'
    assert candidate('8\n7 2\n5 4\n2 6\n1 3\n3 2\n4 2\n4 8') == '5\n1 7 8'
    assert candidate('8\n3 5\n3 6\n8 7\n2 7\n2 1\n2 6\n1 4') == '7\n8 4 5'
    assert candidate('8\n1 8\n5 4\n2 5\n6 3\n1 5\n1 7\n8 6') == '6\n4 7 3'
    assert candidate('8\n1 7\n6 7\n2 3\n8 4\n5 7\n2 1\n6 8') == '7\n3 5 4'
    assert candidate('8\n8 1\n3 4\n6 4\n3 5\n2 4\n8 4\n7 3') == '5\n1 6 7'
    assert candidate('8\n2 3\n4 7\n3 7\n2 6\n5 6\n4 1\n3 8') == '7\n1 8 5'
    assert candidate('8\n3 6\n8 2\n3 1\n8 4\n8 7\n6 4\n1 5') == '7\n5 2 7'
    assert candidate('8\n1 6\n2 7\n4 5\n6 5\n4 8\n2 5\n3 7') == '7\n8 1 3'
    assert candidate('8\n2 3\n6 5\n1 8\n4 5\n4 8\n7 6\n2 6') == '7\n1 7 3'
    assert candidate('8\n1 2\n5 8\n6 8\n4 5\n6 2\n3 7\n7 4') == '7 \n1 2 3'
    assert candidate('8\n1 7\n2 8\n7 8\n4 5\n3 4\n1 5\n6 3') == '7 \n2 8 6'
    assert candidate('8\n2 8\n8 1\n5 7\n6 4\n4 7\n7 2\n7 3') == '6\n1 5 6'
    assert candidate('8\n8 5\n6 3\n8 3\n7 2\n1 2\n5 4\n6 7') == '7 \n1 2 4'
    assert candidate('8\n6 7\n5 8\n4 1\n3 5\n3 6\n7 2\n4 2') == '7 \n1 4 8'
    assert candidate('8\n6 7\n6 8\n1 3\n2 3\n5 6\n8 4\n7 3') == '6\n2 5 4'
    assert candidate('9\n3 1\n7 4\n1 4\n1 8\n2 1\n2 6\n9 1\n1 5') == '5\n6 9 7'
    assert candidate('9\n8 9\n6 2\n1 6\n1 4\n3 1\n9 1\n1 5\n1 7') == '5\n2 7 8'
    assert candidate('9\n3 9\n5 1\n4 1\n7 6\n3 1\n3 2\n8 1\n7 1') == '5\n6 8 9'
    assert candidate('9\n1 3\n6 4\n4 1\n5 1\n7 5\n1 9\n8 5\n1 2') == '5\n6 9 8'
    assert candidate('9\n4 1\n8 2\n6 1\n1 5\n3 1\n6 7\n9 5\n1 2') == '6\n8 7 9'
    assert candidate('9\n1 9\n4 9\n7 1\n3 2\n1 2\n1 6\n1 8\n2 5') == '5\n4 8 5'
    assert candidate('9\n4 1\n2 9\n1 2\n8 1\n9 5\n3 2\n7 6\n7 1') == '6\n6 8 5'
    assert candidate('9\n8 4\n5 8\n3 1\n2 8\n1 7\n9 8\n1 6\n1 8') == '4\n7 6 9'
    assert candidate('9\n8 3\n6 8\n9 4\n1 8\n8 5\n9 3\n2 1\n1 7') == '6\n7 6 4'
    assert candidate('9\n1 8\n5 1\n4 3\n9 1\n2 1\n1 4\n7 1\n6 1') == '4\n9 8 3'
    assert candidate('9\n7 2\n2 4\n5 8\n8 3\n2 1\n1 6\n6 9\n1 8') == '6\n7 5 9'
    assert candidate('9\n5 1\n8 1\n2 1\n4 1\n3 4\n1 7\n7 6\n5 9') == '6\n6 3 9'
    assert candidate('9\n8 1\n9 6\n6 1\n1 3\n2 1\n7 2\n1 4\n4 5') == '6\n7 5 9'
    assert candidate('9\n1 2\n9 8\n5 8\n7 5\n6 2\n5 3\n1 4\n1 5') == '6\n6 7 9'
    assert candidate('9\n7 1\n6 5\n1 8\n4 9\n2 8\n4 1\n3 8\n8 6') == '6\n9 7 5'
    assert candidate('9\n3 7\n3 9\n1 5\n6 1\n1 2\n4 3\n8 2\n3 2') == '5\n6 8 9'
    assert candidate('9\n6 2\n3 7\n5 1\n6 5\n3 1\n9 4\n6 8\n4 3') == '7\n8 7 9'
    assert candidate('9\n4 3\n6 9\n1 9\n1 3\n5 1\n7 1\n8 7\n2 5') == '6\n6 4 8'
    assert candidate('9\n3 5\n9 1\n4 1\n7 4\n3 8\n2 6\n9 2\n3 2') == '7\n7 6 8'
    assert candidate('9\n4 2\n9 5\n1 4\n1 7\n4 9\n5 3\n3 6\n9 8') == '7\n7 8 6'
    assert candidate('9\n5 2\n3 9\n2 4\n7 8\n5 6\n9 8\n1 2\n9 4') == '7\n6 3 7'
    assert candidate('9\n5 8\n8 9\n2 6\n2 7\n3 6\n2 1\n8 1\n4 9') == '7\n3 7 4'
    assert candidate('9\n4 2\n2 6\n4 1\n8 5\n8 7\n6 5\n3 4\n9 2') == '7\n3 9 7'
    assert candidate('9\n1 4\n5 8\n8 7\n3 4\n6 3\n5 3\n2 4\n2 9') == '7\n9 6 7'
    assert candidate('9\n3 6\n9 7\n1 6\n2 8\n7 4\n9 5\n8 6\n3 9') == '7\n2 5 4'
    assert candidate('9\n7 9\n6 7\n4 6\n3 9\n9 8\n2 5\n2 8\n1 2') == '7\n5 3 4'
    assert candidate('9\n9 3\n5 8\n2 7\n2 3\n9 6\n1 7\n4 5\n4 2') == '8\n6 1 8'
    assert candidate('9\n2 7\n3 7\n8 2\n6 7\n1 3\n2 9\n5 2\n4 6') == '6\n4 1 9'
    assert candidate('9\n9 2\n6 8\n4 1\n2 5\n1 9\n8 7\n3 6\n5 7') == '8 \n4 1 3'
    assert candidate('9\n4 3\n6 9\n4 8\n6 5\n7 5\n1 6\n8 5\n7 2') == '7\n9 2 3'
    assert candidate('9\n6 7\n8 6\n1 4\n3 8\n4 7\n5 6\n9 3\n2 7') == '7\n1 5 9'
    assert candidate('9\n6 4\n9 5\n7 8\n1 8\n9 3\n4 2\n7 5\n7 4') == '7\n6 1 3'
    assert candidate('9\n9 2\n3 8\n4 6\n7 9\n2 5\n5 3\n1 6\n9 1') == '8\n4 7 8'
    assert candidate('9\n6 9\n1 3\n6 7\n2 8\n4 6\n2 6\n5 2\n3 7') == '6\n1 9 8'
    assert candidate('9\n1 8\n6 9\n6 7\n4 3\n3 5\n8 7\n2 6\n9 3') == '7\n1 4 5'

check(solution)