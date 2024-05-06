from db_raed import ScadaRead_WP
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay
from sklearn.metrics import davies_bouldin_score
from matplotlib import rcParams

config = {
    "font.family": 'serif',
    "font.size": 18,  # 相当于小四大小
    "mathtext.fontset": 'stix',  # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
    "font.serif": ['Times New Roman'],  # 宋体
    'axes.unicode_minus': False  # 处理负号，即-号
}
rcParams.update(config)

def Dij(data):
    # 计算距离矩阵
    distance_matrix = cdist(data, data)

    # 获取上三角部分的索引
    indices = np.triu_indices(len(data), k=1)
    # 根据索引从距离矩阵中提取上三角部分的元素
    dij_array = distance_matrix[indices]
    ascending_order = np.sort(dij_array)
    ind = np.int64(len(ascending_order) * 0.02)
    return distance_matrix, ascending_order[ind]


def get_local_density(distance_matrix, dc, method=None):
    n = np.shape(distance_matrix)[0]
    rhos = np.zeros(n)
    for i in range(n):
        if method is None:
            rhos[i] = np.where(distance_matrix[i, :] < dc)[0].shape[0] - 1
        else:
            pass
    return rhos


def get_deltas(distance_matrix, rhos):
    n = np.shape(distance_matrix)[0]
    deltas = np.zeros(n)
    rhos_index = np.argsort(-rhos)
    for i, index in enumerate(rhos_index):
        if i == 0:
            continue
        higher_rhos_index = rhos_index[:i]
        deltas[index] = np.min(distance_matrix[index, higher_rhos_index])
    deltas[rhos_index[0]] = np.max(deltas)
    return deltas


def find_k_centers(rhos, deltas, k):
    rho_and_delta = rhos * deltas
    centers = np.argsort(-rho_and_delta)
    return centers[:k]


def davies_bouldin_index(X):
    # 尝试不同的 k 值，并计算每个 k 对应的 Davies-Bouldin index
    k_values = range(2, 15)
    db_scores = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        labels = kmeans.labels_
        db_score = davies_bouldin_score(X, labels)
        db_scores.append(db_score)
    # 绘制 Davies-Bouldin index 随 k 变化的曲线图

    return np.argmin(np.array(db_scores)) + 2


def calculate_limits(data):
    # 计算第一个特征的四分位数
    quartiles = np.percentile(data[:, 0], [25, 50, 75], axis=0)

    # 计算四分位距（IQR）
    iqr = quartiles[2] - quartiles[0]

    # 计算上下限
    upper_limit = quartiles[2] + 1.5 * iqr
    lower_limit = quartiles[0] - 1.5 * iqr

    return upper_limit, lower_limit


def clustering_partition(wp):
    # k = davies_bouldin_index(wp)
    k = 15
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(wp)
    labels = kmeans.labels_
    clusters = []
    for label in np.unique(labels):
        cluster_points = wp[labels == label]
        clusters.append(cluster_points)

    return clusters


def t_boundary(dls, k=2):
    # 初始化KMeans模型
    kmeans = KMeans(n_clusters=k)
    # 拟合模型
    kmeans.fit(dls.reshape(-1, 1))
    # 获取聚类中心
    centroids = kmeans.cluster_centers_
    label_select = np.argmax(centroids)

    # 获取每个样本所属的簇
    labels = kmeans.labels_
    return np.min(dls[labels == label_select])


def boundary_point_get(points):
    # 进行三角剖分
    tri = Delaunay(points)

    # 获取每个点的邻居
    neighbors = [set() for _ in range(len(points))]
    l_neighbors = [set() for _ in range(len(points))]

    for simplex in tri.simplices:
        for i, j in zip(simplex, np.roll(simplex, -1)):
            neighbors[i].add(j)
            l_neighbors[i].add(np.linalg.norm(points[i] - points[j]))
            neighbors[j].add(i)
            l_neighbors[j].add(np.linalg.norm(points[i] - points[j]))

    Dl_list = []
    l_min = []
    for i, l_neighbor in enumerate(l_neighbors):
        if len(l_neighbor) > 0:
            DL_p = max(l_neighbor) - min(l_neighbor)
            Dl_list.append(DL_p)
            l_min.append(min(l_neighbor))
        else:
            Dl_list.append(0)
            l_min.append(0)

    Dl_array = np.array(Dl_list)
    Bf_list = []
    for _, n in enumerate(neighbors):
        if len(n) > 0:
            id = np.int64(np.array(list(n)))
            mean_DL_p = np.sum(Dl_array[id] / len(n))
            Bf_p = np.sqrt(np.sum((Dl_array[id] - mean_DL_p) ** 2) / len(n)) / mean_DL_p
            Bf_list.append(Bf_p)
        else:
            Bf_list.append(0)

    # 阈值
    T = t_boundary(np.array(Bf_list))
    l_min_array = np.array(np.array(l_min))
    avm = np.mean(l_min_array[np.array(Bf_list) > T])
    candidate_boundary = points[np.logical_and(np.array(Bf_list) > T, l_min_array < avm * (1 + T / 2))]
    interior = points[np.logical_and(np.array(Bf_list) < T, l_min_array < avm * (1 + T / 2))]

    return candidate_boundary, np.median(points[:, 1]), interior


def no_empty_sigma(bu, id):
    if len(id) > 0:
        sigma3 = np.std(bu[id, 0])
    else:
        sigma3 = -1
    return sigma3


def B_get(bu, w_list, upper=True):
    bu = bu[np.argsort(bu[:, 1])]
    bu_power = bu[:, 1]
    preserve_boundaries = []
    for i, win in enumerate(w_list):
        bu_id = np.where((w_list[i][0] < bu_power) & (bu_power <= w_list[i][1]))[0]
        if len(bu_id) > 0:
            if upper:
                w_max = np.max(bu[bu_id][:, 0])
            else:
                w_max = np.min(bu[bu_id][:, 0])

            if 3 <= i <= len(w_list) - 3 - 1:
                bu_1 = np.where((w_list[i + 1][0] < bu_power) & (bu_power <= w_list[i + 1][1]))[0]
                bu_2 = np.where((w_list[i + 2][0] < bu_power) & (bu_power <= w_list[i + 2][1]))[0]
                bu_3 = np.where((w_list[i + 3][0] < bu_power) & (bu_power <= w_list[i + 3][1]))[0]

                bu_1_n = np.where((w_list[i - 1][0] < bu_power) & (bu_power <= w_list[i - 1][1]))[0]
                bu_2_n = np.where((w_list[i - 2][0] < bu_power) & (bu_power <= w_list[i - 2][1]))[0]
                bu_3_n = np.where((w_list[i - 3][0] < bu_power) & (bu_power <= w_list[i - 3][1]))[0]

                sigma1 = no_empty_sigma(bu, bu_1)
                sigma2 = no_empty_sigma(bu, bu_2)
                sigma3 = no_empty_sigma(bu, bu_3)
                sigma1_n = no_empty_sigma(bu, bu_1_n)
                sigma2_n = no_empty_sigma(bu, bu_2_n)
                sigma3_n = no_empty_sigma(bu, bu_3_n)
                sigma = min([sigma1, sigma2, sigma3, sigma1_n, sigma2_n, sigma3_n])
                if sigma == -1:
                    preserve_boundaries.append(bu[bu_id])
                else:
                    if upper:
                        t = w_max - 3 * sigma
                        now_data = bu[bu_id]
                        preserve_boundaries.append(now_data[now_data[:, 0] >= t])
                    else:
                        t = w_max + 3 * sigma
                        now_data = bu[bu_id]
                        preserve_boundaries.append(now_data[now_data[:, 0] <= t])
            else:
                preserve_boundaries.append(bu[bu_id])

    return np.unique(np.vstack(preserve_boundaries), axis=0)


def boundary_regularization(boundary_point, center_power):
    # Bu = boundary_point[boundary_point[:, 1] >= center_power]
    # Bl = boundary_point[boundary_point[:, 1] < center_power]
    Bu_correct, Bl_correct = boundary_point, boundary_point
    # # 窗口
    # start = 0
    # window_width = 0.01
    # sliding_step = 0.002
    # end = window_width
    # window_list = []
    # while end <= 1:
    #     window_list.append(np.array([start, end]))
    #     start = start + sliding_step
    #     end = start + window_width
    #
    # Bu_correct = B_get(Bu, window_list)
    # Bl_correct = B_get(Bl, window_list, False)

    return Bu_correct, Bl_correct


def boundary_extraction(wp):
    wp_b, power_center, interior_point = boundary_point_get(wp)
    # a, b = boundary_regularization(wp_b, power_center)

    # plt.scatter(wp[:, 0], wp[:, 1])
    # plt.scatter(interior_point[:, 0], interior_point[:, 1], c='black')
    # # plt.scatter(b[:, 0], b[:, 1], c='blue')
    # plt.show()


def M1(x, c, max_delta):
    vector_modulus = np.linalg.norm(x - c, axis=1)
    return 1 / (vector_modulus * max_delta)


def M2(rhos, rho_k, max_rho):
    vector_modulus = np.abs(rhos - rho_k)
    return 1 / (vector_modulus * max_rho)


def membership_function(wp_data, deltas, rhos, centers_up, centers_wp_up, centers_wp_low, a=0.6):
    max_rho = np.max(rhos)
    max_delta = np.max(deltas)

    id_up = np.argmin(np.linalg.norm((centers_wp_up - wp_data), axis=1))
    rhos_up = rhos[id_up]
    id_low = np.argmin(np.linalg.norm((centers_wp_low - wp_data), axis=1))
    rhos_low = rhos[id_low]
    id_centers = np.argmin(np.linalg.norm((centers_up - wp_data), axis=1))
    rhos_centers = rhos[id_centers]

    # 中心位置
    m1_k0 = M1(wp_data, centers_up, max_delta)
    m2_k0 = M1(wp_data, rhos_centers, max_rho)

    # 上边位置
    m1_k1 = M1(wp_data, centers_wp_up, max_delta)
    m2_k1 = M1(wp_data, rhos_up, max_rho)

    # 下边位置
    m1_k2 = M1(wp_data, centers_wp_low, max_delta)
    m2_k2 = M1(wp_data, rhos_low, max_rho)

    M_k0 = a ** rhos * (m1_k0 + m2_k0)
    M_k1 = a ** rhos * (m1_k1 + m2_k1)
    M_k2 = a ** rhos * (m1_k2 + m2_k2)
    M_k = np.column_stack([M_k0, M_k1, M_k2])

    idd = np.argmax(M_k, axis=1)
    return wp_data[idd == 0]


def density_clustering(wp_p):
    data = []
    for wp in wp_p:
        # 计算距离dij
        dij_a, dc = Dij(wp)
        rhos = get_local_density(dij_a, dc)
        deltas = get_deltas(dij_a, rhos)

        # 高密度索引
        centers_up = wp[find_k_centers(rhos, deltas, 1)]
        centers_wp_up, centers_wp_low = calculate_limits(wp)

        wwp = membership_function(wp, deltas, rhos, centers_up, centers_wp_up, centers_wp_low)
        data.append(wwp)
    return data


def density_clustering_boundary_extraction(wp):
    scaler = MinMaxScaler()
    scaler.fit_transform(wp)
    wp_stand = scaler.transform(wp)
    data_p = clustering_partition(wp_stand)
    data_list = density_clustering(data_p)
    boundary_extraction(np.vstack(data_list))
    return scaler.inverse_transform(np.vstack(data_list))


if __name__ == '__main__':
    data_name = "B13_201706"  # 替换为你需要的变量值

    # data_preprocess
    query0 = "SELECT 30秒平均风速, 有功功率 FROM " + data_name
    wp_get = ScadaRead_WP()
    wind_speed1, _ = wp_get.scada_data(query0, data_name)

    wind_speed1 = np.delete(wind_speed1, np.where(wind_speed1[:, 0] < 0), axis=0)
    wind_speed1 = np.delete(wind_speed1, np.where(wind_speed1[:, 1] < 0), axis=0)
    data_clear_after = density_clustering_boundary_extraction(wind_speed1)

    # 绘图
    fig = plt.figure(figsize=(8, 6))  # 设置画布大小
    plt.scatter(wind_speed1[:, 0], wind_speed1[:, 1], label='Abnormal data', color=[246 / 255, 1 / 255, 1 / 255], s=2,
                zorder=1)
    plt.scatter(data_clear_after[:, 0], data_clear_after[:, 1], label='Normal data', color=[0 / 255, 52 / 255, 245 / 255], s=2,
                zorder=2)
    # 设置图例
    plt.legend(loc='best', fontsize=14)  # 放置图例在最合适的位置

    # 设置坐标轴标签和标题
    plt.xlabel('Wind Speed(m/s)')
    plt.ylabel('Wind Power(kW)')

    # 设置坐标轴范围
    plt.xlim([0, 25])
    plt.ylim([-100, 2300])

    plt.show()
    fig.savefig(r"C:\Users\admin\Desktop\density_b_fig.png", format='png', dpi=600)
