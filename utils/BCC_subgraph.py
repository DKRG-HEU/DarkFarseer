import torch
import torch.nn.functional as F
import numpy as np


import numpy as np
import json
# coding=utf-8

# from Tarjan import Graph as TGraph
import sys
sys.path.append("..")
from collections import defaultdict, OrderedDict
from copy import deepcopy


# This class represents an directed graph
# using adjacency list representation
class Graph:

    def __init__(self, vertices):
        # No. of vertices
        self.V = vertices

        # default dictionary to store graph
        self.graph = defaultdict(list)

        self.res = []

        # time is used to find discovery times
        self.Time = 0

        # Count is number of biconnected components
        self.count = 0

    # function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)
        self.graph[v].append(u)

    '''A recursive function that finds and prints strongly connected
    components using DFS traversal
    u --> The vertex to be visited next
    disc[] --> Stores discovery times of visited vertices
    low[] -- >> earliest visited vertex (the vertex with minimum
               discovery time) that can be reached from subtree
               rooted with current vertex
    st -- >> To store visited edges'''

    def BCCUtil(self, u, parent, low, disc, st):

        # Count of children in current node
        children = 0

        # Initialize discovery time and low value
        disc[u] = self.Time
        low[u] = self.Time
        self.Time += 1

        # Recur for all the vertices adjacent to this vertex
        for v in self.graph[u]:
            # If v is not visited yet, then make it a child of u
            # in DFS tree and recur for it
            if disc[v] == -1:
                parent[v] = u
                children += 1
                st.append((u, v))  # store the edge in stack
                # st.append([u, v])
                self.BCCUtil(v, parent, low, disc, st)

                # Check if the subtree rooted with v has a connection to
                # one of the ancestors of u
                # Case 1 -- per Strongly Connected Components Article
                low[u] = min(low[u], low[v])

                # If u is an articulation point, pop
                # all edges from stack till (u, v)
                if parent[u] == -1 and children > 1 or parent[u] != -1 and low[v] >= disc[u]:
                    self.count += 1  # increment count
                    w = -1
                    tmp = []
                    while w != (u, v):
                        w = st.pop()
                        tmp.append(w)
                        # print(w, end=" ")
                    # print()
                    self.res.append(tmp)


            elif v != parent[u] and low[u] > disc[v]:
                '''Update low value of 'u' only of 'v' is still in stack
                (i.e. it's a back edge, not cross edge).
                Case 2
                -- per Strongly Connected Components Article'''

                low[u] = min(low[u], disc[v])

                st.append((u, v))
                # st.append((u, v))

    # The function to do DFS traversal.
    # It uses recursive BCCUtil()
    def BCC(self):

        # Initialize disc and low, and parent arrays
        disc = [-1] * (self.V)
        low = [-1] * (self.V)
        parent = [-1] * (self.V)
        st = []

        # Call the recursive helper function to
        # find articulation points
        # in DFS tree rooted with vertex 'i'
        for i in range(self.V):
            if disc[i] == -1:
                self.BCCUtil(i, parent, low, disc, st)

            # If stack is not empty, pop all edges from stack
            if st:
                self.count = self.count + 1
                tmp=[]
                while st:
                    w = st.pop()
                    tmp.append(w)
                    # print(w, end=" ")
                self.res.append(tmp)
                # print()


def get_1hop_neighbors(adj):
    num_nodes = adj.shape[0]
    neighbors_in_degree = {}

    full_neighbors = {}
    for i in range(num_nodes):
        neighbors_in_degree[i] = list(np.where(adj[i] > 0)[0])

        full_neighbors[i] = list(set(list(np.where(adj[i] > 0)[0]) + list(np.where(adj[:, i] > 0)[0])))

    return full_neighbors


def get_BBC(adj, dataset, edge_threshhold=None):
    if not edge_threshhold:
        if dataset=="PEMS04" or dataset == "PEMS03":
            edge_threshhold = 0.0
        elif dataset=="AIR36":
            edge_threshhold = 0.5
        elif dataset == "METR-LA" or dataset == "PEMS-BAY":
            edge_threshhold = 0.7
        elif dataset == "NREL-PA" or dataset == "USHCN":
            edge_threshhold = 0.9
        else:
            raise NameError("")
        
    original = adj

    g1 = Graph(len(original))
    add_edge_num = 0
    for i in range(len(original)):
        for j in range(len(original)):
            if original[i][j] > edge_threshhold: #threshhold
                g1.addEdge(i, j)
                add_edge_num += 1

    # print("add_edge_num: ", add_edge_num)

    g1.BCC()

    map=[]

    for lists in g1.res:
        tmp = ()
        for sets in lists:
            # print(type(set))
            tmp += sets
        map.append(list(set(tmp)))

    indices = ()
    for i in map:
        indices += tuple(i)

    base = [i for i in range(len(original[0]))]

    assigned = set(indices)
    notMap=set(base)-assigned
    notMap = list(notMap)

    BBC_dict = { key:None for key in notMap }
    for bbc_list in map:
        for element in bbc_list:
            if not BBC_dict.get(element):
                BBC_dict[element] = list()
                BBC_dict[element] = BBC_dict[element] + bbc_list
            else:
                BBC_dict[element] = BBC_dict[element] + bbc_list

    BBC_list = list()
    for key in sorted(BBC_dict):
        BBC_list.append(BBC_dict[key])

    # print(BBC_list)

    BBC_node_index = np.zeros((len(original), len(original)))
    for i in range(len(original)):
        if not BBC_list[i]:
            continue
        for j in BBC_list[i]:
            if i!=j:  # remove self-loop
                BBC_node_index[i][j] = 1

    # key: node_index; value: bbc_subgraph[List]
    positive_select = dict()
    negative_select = dict()

    negative_articulation_point = dict()

    node2_1hop_list = get_1hop_neighbors(adj)

    for query_node_index in range(adj.shape[0]):
        # print(node2_1hop_list[query_node_index])
        if query_node_index in notMap:
            ## dict.get(query_node_index) == None
            one_hop_list = node2_1hop_list[query_node_index]
            negative_articulation_point[query_node_index] = [node_idx for node_idx in notMap \
                                                             if node_idx not in one_hop_list and node_idx != query_node_index]
            
            positive_select[query_node_index] = deepcopy(one_hop_list)+[query_node_index]
            # print(one_hop_list)
            continue
        one_hop_list = node2_1hop_list[query_node_index]
        negative_articulation_point[query_node_index] = [node_idx for node_idx in notMap \
                                                             if node_idx not in one_hop_list and node_idx != query_node_index]

        memory_set = set()

        for prototype in map:
            if query_node_index in prototype:


                if not positive_select.get(query_node_index):
                    positive_select[query_node_index] = list()

                positive_select[query_node_index].append(prototype)

                memory_set = memory_set.union(set(prototype))


        memory_set = memory_set - set([query_node_index])

        if_used = {prototype_id:0 for prototype_id in range(len(map))}
        
        for node_idx in memory_set:
            for prototype_id, prototype in enumerate(map):
                if node_idx in prototype:
                    if_used[prototype_id] += 1

        for prototype_id, used in if_used.items():
            if used == 0:
                if not negative_select.get(query_node_index):
                    negative_select[query_node_index] = []
                negative_select[query_node_index].append(map[prototype_id])


    node2proto = dict() # order_dict

    for spatial_prototype_id, node_indices in enumerate(map):
        for node_index in node_indices:
            # if not isinstance(node2proto.get(node_index), list):
            if not node2proto.get(node_index):
                node2proto[node_index] = [spatial_prototype_id]
            else:
                node2proto[node_index].append(spatial_prototype_id)


    node2proto_node = dict()  # order_dict

    for node_index, spatial_prototype_id in node2proto.items():
        if len(spatial_prototype_id) == 1:
            # print(spatial_prototype_id)
            node2proto_node[node_index] = set(map[spatial_prototype_id[0]]) - set([node_index])
        else:
            result = []
            for j in spatial_prototype_id:
                result = result + map[j]
            
            node2proto_node[node_index] = set(result)-set([node_index])
            # node2proto_node[node_index] = set(result)


    need_delete = dict()
    for ind1, (k1, v1) in enumerate(node2proto_node.items()):
        for ind2, (k2, v2) in enumerate(node2proto_node.items()):
            if ind1 == ind2:
                continue
            if v1 == v2:
                if need_delete.get(k1) is None:
                    need_delete[k1] = [ind2]  # ind2: prototype_id
                else:
                    need_delete[k1].append(ind2)

    return BBC_node_index, notMap, positive_select, negative_select, negative_articulation_point

  
