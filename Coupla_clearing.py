import numpy as np
import matplotlib.pyplot as plt
from db_raed import ScadaRead_WP
from scipy.optimize import minimize
from sympy import symbols, diff
from sklearn.neighbors import KernelDensity
import multiprocessing

def partial_derivatives(function, variables, point=None):
    """
    计算函数的偏导数

    参数：
    function (sympy expression): 要求偏导数的函数
    variables (list of sympy symbols): 要对其求偏导数的变量
    point (dict, optional): 要计算偏导数的点的坐标

    返回：
    dict: 包含偏导数表达式和值的字典
    """
    derivatives = {}
    for var in variables:
        derivative = diff(function, var)
        derivatives[var] = derivative
        if point:
            value = derivative.subs(point)
            # derivatives[f'{var}_value'] = value
    return value


def perpendicular_distance(points):
    """
    计算点到直线的垂直距离
    :param point: 数据点，形如 [x, y, z]
    :param line_point1: 直线上的第一个点，形如 [x1, y1, z1]
    :param line_point2: 直线上的第二个点，形如 [x2, y2, z2]
    :return: 垂直距离
    """
    # 计算直线的方向向量
    line_point2 = [1, 1, 1]
    line_direction = np.array(line_point2)

    # 计算点到直线的垂直距离
    distances = np.linalg.norm(np.cross(np.array(points), line_direction),
                               axis=1) / np.linalg.norm(line_direction)

    return distances


def cdf(d):
    # 计算概率密度函数 (PDF)
    hist, edges = np.histogramdd(d, bins=50)

    # 计算累积分布函数 (CDF)
    cdf = np.cumsum(hist / len(d))

    # 找到每个数据元素的CDF值
    bin_indices = np.digitize(d, edges[0])
    bin_indices[bin_indices > len(cdf)] = len(cdf)  # 处理位于边界外的情况
    return cdf[bin_indices - 1]


def confidence_interval_indices(data, confidence_level=90):
    """
    对一组数据构造置信区间，并保留区间中的数据索引值
    :param data: 一维数组，表示数据集
    :param confidence_level: 置信水平，取值范围 [0, 100]
    :return: 置信区间内的数据索引值
    """
    # 计算置信区间的上下界
    lower_bound = np.percentile(data, (100 - confidence_level) / 2)
    upper_bound = np.percentile(data, 100 - (100 - confidence_level) / 2)

    # 使用布尔索引获取置信区间内的数据索引值
    indices = np.where((data >= lower_bound) & (data <= upper_bound))[0]

    return indices


def pre_copula(data):
    # 构造Copula模型
    # copula = GaussianMultivariate()
    # copula.fit(data)
    # # 获取边缘分布
    # marginal_distributions = copula.probability_density(data)
    # 计算概率累计函数
    u_cdf = cdf(data[:, 0])
    v_cdf = cdf(data[:, 1])
    w_cdf = cdf(data[:, 2])
    # copula_y = marginal_distributions

    uvw_data = np.stack([u_cdf, v_cdf, w_cdf]).T

    dd = perpendicular_distance(uvw_data)
    ind = confidence_interval_indices(dd)

    # # # 绘制散点图
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # scatter = ax.scatter(uvw_data[:, 0], uvw_data[:, 1], uvw_data[:, 2], c=copula_y, cmap='viridis')
    # # 添加颜色条
    # cbar = fig.colorbar(scatter)
    # cbar.set_label('Fourth Dimension')
    # plt.show()
    return data[ind]


def split_array_by_first_column(data, num_segments=5):
    # 根据第一列数据排序
    sorted_indices = np.argsort(data[:, 0])
    sorted_data = data[sorted_indices]

    # 计算每个分段的大小
    segment_size = len(sorted_data) // num_segments

    # 初始化分段列表
    segments = []

    # 划分数据
    for i in range(num_segments):
        if i < num_segments - 1:
            segment = sorted_data[i * segment_size: (i + 1) * segment_size]
        else:
            # 最后一个段包含剩余的所有数据
            segment = sorted_data[i * segment_size:]
        segments.append(segment)

    return segments


def segment_clear(w_p):
    data_after = []
    data_s = split_array_by_first_column(w_p)
    for s in data_s:
        if len(s) != 0:
            data_after.append(pre_copula(s))
    return np.vstack(data_after)


def preliminarily_data_cleaning(wpw, v_cut_in=3, v_w_rated=8.87, v_rated=10.8, w_min=9, w_rated=15.3):
    # 第一部分清洗
    condition = np.logical_and(wpw[:, 0] < v_cut_in, wpw[:, 1] != 0)
    wpw_I = wpw[~condition]

    # 第二部分清洗
    condition = np.logical_and(wpw_I[:, 0] >= v_cut_in, wpw_I[:, 0] < v_w_rated)
    wpw_II = wpw_I[condition]
    wpw_II_after = segment_clear(wpw_II)

    # 第三部分清洗
    condition = np.logical_and(wpw_I[:, 0] >= v_w_rated, wpw_I[:, 0] < v_rated)
    wpw_III = wpw_I[condition]
    wpw_III_after = segment_clear(wpw_III)

    # 第四部分清洗
    condition = np.where(wpw_I[:, 0] >= v_rated)
    wpw_IV = wpw_I[condition]
    wpw_IV_after = segment_clear(wpw_IV)

    return wpw_II_after[:, :2], wpw_III_after[:, :2], wpw_IV_after[:, :2]


def C_G(u, v, theta_G):
    return np.exp(-((-np.log(u)) ** theta_G + (-np.log(v)) ** theta_G) ** (1 / theta_G))


# Define the Frank Copula function
def C_F(u, v, theta_F):
    return -1 / theta_F * np.log(1 + ((np.exp(-theta_F * u) - 1) * (np.exp(-theta_F * v) - 1)) / (np.exp(-theta_F) - 1))


# Define the mixed Archimedes Copula function
def mixed_archimedes_copula(u, v, phi_G, phi_F, theta_G, theta_F):
    return (phi_G * C_G(u, v, theta_G) + phi_F * C_F(u, v, theta_F)) / (phi_G + phi_F)


# Define the negative log-likelihood function for optimization
def negative_log_likelihood(params, data):
    phi_G, phi_F, theta_G, theta_F = params
    u, v = data
    return -np.sum(np.log(mixed_archimedes_copula(u, v, phi_G, phi_F, theta_G, theta_F)))


# EM algorithm with gradient descent optimization
def fit_mixed_archimedes_copula(data):
    # Initial parameter values
    initial_params = np.random.rand(4)  # phi_G, phi_C, phi_F, theta_G, theta_C, theta_F

    # Minimize negative log-likelihood using gradient descent
    result = minimize(negative_log_likelihood, initial_params, args=(data,), method='BFGS')

    return result.x


def condition_cfd(F_v, F_p, org_wp):
    # 风速-功率概率密度
    kde1 = KernelDensity(bandwidth=0.5, kernel='gaussian')
    kde1.fit(org_wp)
    max_p = np.max(org_wp[:, 1])
    y_grid_power = np.linspace(0, max_p, 100)

    data = np.vstack((F_v, F_p)).T
    # 使用Scikit-learn的KernelDensity估计器
    kde = KernelDensity(bandwidth=0.5, kernel='gaussian')
    kde.fit(data)

    y_grid_all = np.linspace(0, 1, 100)
    low = 0.045
    up = 0.995
    grid_low = np.int64(np.floor(100 * low))
    grid_up = np.int64(np.floor(100 * up))

    y_grid_low = np.linspace(0, np.int64(low), grid_low)
    y_grid_up = np.linspace(0, np.int64(up), grid_up)
    power_limit = []
    for i, d in enumerate(data):
        # 原始数据
        x_grid_wind = np.full(100, org_wp[i, 0])  # 生成一个包含100个值为5的数组
        xy_grid_wp = np.column_stack([x_grid_wind.ravel(), y_grid_power.ravel()])
        log_dens_wp = kde1.score_samples(xy_grid_wp)
        p_w = np.exp(log_dens_wp) / np.sum(np.exp(log_dens_wp))
        cdf_p_w = np.cumsum(p_w)

        # 定义网格，用于绘制概率分布图
        x_grid_low = np.full(grid_low, d[0])  # 生成一个包含100个值为5的数组
        x_grid_up = np.full(grid_up, d[0])  # 生成一个包含100个值为5的数组
        x_grid_x0 = np.full(100, d[0])  # 生成一个包含100个值为5的数组

        xy_grid_low = np.column_stack([x_grid_low.ravel(), y_grid_low.ravel()])
        xy_grid_up = np.column_stack([x_grid_up.ravel(), y_grid_up.ravel()])
        xy_grid_x0 = np.column_stack([x_grid_x0.ravel(), y_grid_all.ravel()])

        # 计算概率密度估计
        log_dens = kde.score_samples(xy_grid_x0)
        sum_x0 = np.sum(np.exp(log_dens))

        # 计算概率密度估计
        log_dens = kde.score_samples(xy_grid_low)
        gamma_low = np.sum(np.exp(log_dens) / sum_x0)

        log_dens = kde.score_samples(xy_grid_up)
        gamma_up = np.sum(np.exp(log_dens) / sum_x0)

        ind_low = np.argmin(np.abs(cdf_p_w - gamma_low))
        ind_up = np.argmin(np.abs(cdf_p_w - gamma_up))
        power_low = ind_low / 100 * max_p
        power_up = ind_up / 100 * max_p
        power_limit.append(np.array([power_low, power_up]))

    return np.array(power_limit)


def boundary_Copula(x):
    F_v = cdf(x[:, 0])
    F_p = cdf(x[:, 1])
    # 最小化目标函数
    # 执行EM算法
    # Fit mixed Archimedes Copula to data
    # phi_G, phi_F, theta_G, theta_F = fit_mixed_archimedes_copula((F_v, F_p))
    # 条件累积分布函数(求导)

    limit_b = condition_cfd(F_v, F_p, x[:, :2])
    clear_data = []
    for i, (low, up) in enumerate(limit_b):
        if low <= x[i, 1] <= up:
            clear_data.append(x[i, :])
    return np.array(clear_data)


def confidence_boundary_coupla(wp):
    wp_data_II, wp_data_III, wp_data_IV = preliminarily_data_cleaning(wp)

    # the mixed Archimedes Copula
    data0 = boundary_Copula(wp_data_II[:5000])
    data1 = boundary_Copula(wp_data_III[:5000])
    data2 = boundary_Copula(wp_data_IV[:5000])

    data_clear = np.concatenate([data0, data1, data2], axis=0)
    return data_clear

def parallel_confidence_boundary_coupla(wp):
    wp_data_II, wp_data_III, wp_data_IV = preliminarily_data_cleaning(wp)

    with multiprocessing.Pool(processes=3) as pool:
        results = pool.map(boundary_Copula, [wp_data_II, wp_data_III, wp_data_IV])

    data_clear = np.concatenate(results, axis=0)
    return data_clear


if __name__ == '__main__':
    data_name = "B13_201704"  # 替换为你需要的变量值

    # data_preprocess
    query0 = "SELECT 30秒平均风速, 有功功率, 风轮转速 FROM " + data_name
    wp_get = ScadaRead_WP()
    wind_speed1, _ = wp_get.scada_data(query0, data_name)

    wind_speed1 = np.delete(wind_speed1, np.where(wind_speed1[:, 0] < 0), axis=0)
    wind_speed1 = np.delete(wind_speed1, np.where(wind_speed1[:, 1] < 0), axis=0)
    data_clear_after = parallel_confidence_boundary_coupla(wind_speed1)

    plt.scatter(wind_speed1[:, 0], wind_speed1[:, 1])
    plt.scatter(data_clear_after[:, 0], data_clear_after[:, 1])
    plt.show()
