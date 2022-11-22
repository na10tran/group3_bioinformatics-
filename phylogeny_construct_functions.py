import copy
import pandas as pd
import numpy as np
import networkx as nx

def show(T):
    T = copy.deepcopy(T)
    labels = nx.get_edge_attributes(T,'weight')
    max_value = 0
    for n1,n2 in T.edges():
        if T[n1][n2]['weight'] > max_value:
            max_value = T[n1][n2]['weight']
    for n1,n2 in T.edges():
        T[n1][n2]['weight']=max_value - T[n1][n2]['weight'] + 3
    pos=nx.spring_layout(T)
    nx.draw(T,pos,with_labels=True)
    nx.draw_networkx_edge_labels(T,pos,edge_labels=labels);
    
def show_adj(T):
    if len(T.nodes()) == 0:
        return pd.DataFrame()
    return pd.DataFrame(nx.adjacency_matrix(T).todense(),index=T.nodes(),columns=T.nodes())


def dfs_path(G,nodei,nodej):
    currNode = None
    stack = [nodei]
    visited = []
    prevDict = {}
    while currNode != nodej:
        currNode = stack.pop(0)
        visited.append(currNode)
        edgeList = [edge for edge in G.edges() if (edge[0] == currNode) or (edge[1] == currNode)]
        for edge in edgeList:
            if edge[0] == currNode:
                newNode = edge[1]
            else:
                newNode = edge[0]
            if newNode not in visited:
                stack.append(newNode)
                prevDict[newNode] = currNode
    path = [currNode]
    while currNode != nodei:
        currNode = prevDict[currNode]
        path.append(currNode)
    path.reverse()
    return path


def limb(D,j):
    min_length = np.Inf
    nodes = D.drop(j).index
    for ix,i in enumerate(nodes):
        for kx in range(ix+1,len(nodes)):
            k = nodes[kx]
            min_length = min(min_length, (D[j][i] + D[j][k] - D[i][k])/2)
    return min_length

def find(D,n):
    nodes = D.drop(n).index
    for ix,i in enumerate(nodes):
        for kx in range(ix+1,len(nodes)):
            k = nodes[kx]
            if D[i][k] == D[i][n] + D[n][k]:
                break
        if D[i][k] == D[i][n] + D[n][k]:
            break       
    return i,k

def base_case(D):
    T = nx.Graph()
    ind = D.index
    T.add_edge(ind[0],ind[1],weight=D[ind[0]][ind[1]])
    return T

def additive_phylogeny(D,new_number):
    D = copy.deepcopy(D)
    if len(D) == 2:
        return base_case(D)
    n = D.index[-1]
    limbLength = limb(D,n) 
    Dtrimmed = D.drop(n).drop(n,axis=1)
    for j in Dtrimmed.index:
        D.loc[j,n] = D.loc[j,n] - limbLength
        D.loc[n,j] = D.loc[j,n
    Dtrimmed = D.drop(n).drop(n,axis=1)
    T = additive_phylogeny(Dtrimmed,new_number+1)
    i,k = find(D,n)
    if D.loc[j,n] < D.loc[i,n]:
        i,k = k,i
    v = "v%s"%new_number
    path = dfs_path(T,i,k)
    edgeList = T.edges.data()
    for j in range(len(path)-1):
        tempLeft = path[:j+1]
        tempRight = path[j+1:]
        leftEdgeSum = 0
        rightEdgeSum = 0
        if len(tempLeft) != 1:
            for l in range(len(tempLeft)-1):
                leftEdgeSum += [e[2]["weight"] for e in edgeList if (tempLeft[l] == e[0] and tempLeft[l+1] == e[1]) or (tempLeft[l+1] == e[0] and  tempLeft[l] == e[1])][0]
        if len(tempRight) != 1:
            for l in range(len(tempRight)-1):
                rightEdgeSum += [e[2]["weight"] for e in edgeList if (tempRight[l] == e[0] and tempRight[l+1] == e[1]) or (tempRight[l+1] == e[0] and  tempRight[l] == e[1])][0]
        w1 = D[i][n] - leftEdgeSum
        w2 = D[k][n] - rightEdgeSum
        if w1 > 0 and w2 > 0:
            i = path[j]
            k = path[j+1]
            break
    T.remove_edge(i,k)
    T.add_edge(i,v,weight=w1)
    T.add_edge(n,v,weight=limbLength)
    T.add_edge(k,v,weight=w2)
    return T