import numpy as np
import matplotlib.pyplot as plt
import heapq
import random

def creat_city(num, scale):
    """
    input:
        num: 城市数量
        scale: 城市坐标范围x,y in (0, scale)
    return:
        V：城市的坐标集合
        E：城市的邻接矩阵
    """
    x = np.random.choice(scale, num)
    y = np.random.choice(scale, num)

    V = np.stack((x, y), axis=1)
    inner = -2 * V.dot(V.T)
    xx = np.sum(V ** 2, axis=1, keepdims=True)
    E = xx + inner + xx.T
    E = E ** 0.5
    index = [i for i in range(num)]
    # 为了防止蚂蚁出现自旋，邻接矩阵上的对角线取值尽量大一点。
    E[index, index] = 9999999
    return V, E

def a_res(samples, m):
    """
    :samples: [(item, weight), ...]
    :k: number of selected items
    :returns: [(item, weight), ...]
    """
    #根据概率选择下一个要去的城市

    heap = [] # [(new_weight, item), ...]
    for sample in samples:
        wi = sample[1]
        if wi==0:
            continue
        ui = random.uniform(0, 1)
        ki = ui ** (1/wi)

        if len(heap) < m:
            heapq.heappush(heap, (ki, sample))
        elif ki > heap[0][0]:
            heapq.heappush(heap, (ki, sample))

            if len(heap) > m:
                heapq.heappop(heap)

    return [item[1] for item in heap]


def possibility(eta, gamma, other_city, cur_city):
    """
    返回候选城市集合中，从start到各候选城市的概率，只返回有路径的
    """
    alpha = 1
    beta = 5
    start_city = cur_city[-1]

    t_i = gamma[start_city]#从startcity到各点的信息素浓度
    n_i = eta[start_city]#从startcity到各点的启发值

    temp = t_i ** alpha * n_i ** beta
    temp[cur_city] = 0
    add = temp.sum()
    p_ij = temp / add

    return p_ij

def rotate(l, n):
    '''
    旋转列表。
    '''
    return l[n:] + l[:n]

def get_path_dis(root, E):
    """
    获取该路径距离。
    """
    dis = E[root[:-1], root[1:]].sum()
    return dis + E[root[0],root[-1]]

def ACS(V, E, M, num):
    """
    Ant system
    V : 点集
    E: 邻接矩阵，点之间的连接性，
    M: 蚂蚁数量
    num：迭代次数
    """
    # 相关参数
    global_best_path = None  # 当前最优路径
    global_best_dis = 99999999
    cur_city = None
    other_city = [i for i in range(len(V))]
    lo = 0.5  # 信息素挥发率

    # 信息素启发值
    eta = 1 / E
    eta[np.isinf(eta)] = 0

    # 信息素浓度
    E_mean = E[E > 0].mean()
    gamma = np.full(E.shape, 1 / len(V))

    V_index = [i for i in range(len(V))]

    for i in range(num):
        epoch_gamma = np.zeros_like(gamma)  # 保存每一轮的各路径信息素累积量
        local_best_path = None  # 每一次迭代当前最优路径
        local_best_dis = 99999999
        for j in range(M):
            cur_city = [j % len(V)]#顺序分布蚂蚁
            other_city = [i for i in range(len(V))]
            other_city.remove(cur_city[-1])
            while other_city:
                p_ij = possibility(eta, gamma, other_city, cur_city)

                next_city = int(a_res(np.stack((V_index, p_ij), axis=1), 1)[0][0])

                epoch_gamma[cur_city[-1], next_city] += gamma[cur_city[-1], next_city]
                cur_city.append(next_city)
                other_city.remove(next_city)
            epoch_dis = get_path_dis(cur_city, E)
            if epoch_dis < local_best_dis:
                local_best_dis = epoch_dis
                local_best_path = cur_city

        if local_best_dis < global_best_dis:
            global_best_dis = local_best_dis
            global_best_path = local_best_path

        gamma = (1 - lo) * gamma + epoch_gamma

    best_path = rotate(global_best_path, global_best_path.index(0))

    return best_path