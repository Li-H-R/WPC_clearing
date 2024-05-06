from db_raed import ScadaRead_WP
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
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


def rho(d_m, D_c):
    rho_list = []
    for i, a in enumerate(d_m):
        # 计算a[i:]中大于3的元素个数
        count = sum(1 for x in a[i:] if x < D_c)
        rho_list.append(count)
    return np.array(rho_list), 0.1 * max(rho_list)


def partition_process(data, boundaries):
    add_new_par = []
    segments = np.digitize(data[:, 0], boundaries)
    for i in range(1, len(boundaries)):
        segment_data = data[segments == i]
        if len(segment_data) > 0:
            add_new_par.append(segment_data)

    return add_new_par


def partition_wind_power(data1):
    v1 = np.max(data1[:, 0])
    v0 = np.min(data1[:, 0])
    boundaries2 = np.linspace(v0 - 1, v1 + 1, 20)
    condition2 = partition_process(data1, boundaries2)

    return condition2


def CFSFDP(wpd):
    """
    风速、功率、风向
    :param wpd:
    :return:
    """
    scaler = StandardScaler()
    data_z = scaler.fit_transform(wpd)  # 标准化
    data_after = []
    # 分段
    d = partition_wind_power(data_z)
    for o in d:
        # 计算距离dij
        if len(o) > 1:
            dij_a, d_c = Dij(o)
            # 计算密度
            rho_i, rho_t = rho(dij_a, d_c)
            data_after.append(o[rho_i > rho_t])

    return scaler.inverse_transform(np.vstack(data_after))


if __name__ == '__main__':
    data_name = "B10_201704"  # 替换为你需要的变量值

    # data_preprocess
    query0 = "SELECT 30秒平均风速, 有功功率, 风向角 FROM " + data_name
    wp_get = ScadaRead_WP()
    wind_speed1, _ = wp_get.scada_data(query0, data_name)

    wind_speed1 = np.delete(wind_speed1, np.where(wind_speed1[:, 0] < 0), axis=0)
    wind_speed1 = np.delete(wind_speed1, np.where(wind_speed1[:, 1] < 0), axis=0)

    data = CFSFDP(wind_speed1)

    fig = plt.figure(figsize=(8, 6))  # 设置画布大小
    plt.scatter(wind_speed1[:, 0], wind_speed1[:, 1], label='Abnormal data', color=[246 / 255, 1 / 255, 1 / 255], s=2,
                zorder=1)
    plt.scatter(data[:, 0], data[:, 1], label='Normal data', color=[0 / 255, 52 / 255, 245 / 255], s=2,
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
    fig.savefig(r"C:\Users\admin\Desktop\CFSFDP_fig.png", format='png', dpi=600)
