from db_raed import ScadaRead_WP
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from matplotlib import rcParams

config = {
    "font.family": 'serif',
    "font.size": 18,  # 相当于小四大小
    "mathtext.fontset": 'stix',  # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
    "font.serif": ['Times New Roman'],  # 宋体
    'axes.unicode_minus': False  # 处理负号，即-号
}
rcParams.update(config)

def longitudinal_quartile_algorithm(wind_speed, wind_power):
    # Step 1: Longitudinal quartile algorithm

    # Define the interval for wind speed bins
    wind_speed_bins = np.arange(np.min(wind_speed), np.max(wind_speed)+0.25, 0.25)
    # Initialize lists to store abnormal data indices
    abnormal_indices = []

    for i in range(len(wind_speed_bins) - 1):
        # Find wind power data within the current wind speed interval
        mask = (wind_speed >= wind_speed_bins[i]) & (wind_speed < wind_speed_bins[i + 1])

        wind_power_interval = wind_power[mask]

        if len(wind_power_interval) != 0:
            # Calculate quartiles and quartile distance
            Q1_1i = np.quantile(wind_power_interval, 0.25)
            Q1_3i = np.quantile(wind_power_interval, 0.75)
            I1_QRi = Q1_3i - Q1_1i

            # Calculate lower and upper limits of abnormal inner limit of wind power
            Pli = Q1_1i - 1.5 * I1_QRi
            Pui = Q1_3i + 1.5 * I1_QRi

            id = np.where((wind_power_interval < Pli) | (wind_power_interval > Pui))[0]
            abnormal_indices.extend(np.where(mask)[0][id])
    return abnormal_indices


def lateral_quartile_algorithm(wind_speed, wind_power):
    # Step 2: Lateral quartile algorithm

    # Define the interval for wind power sections
    wind_power_sections = np.arange(np.min(wind_power), np.max(wind_power) + 25, 25)

    # Initialize lists to store abnormal data indices
    abnormal_indices = []

    for i in range(len(wind_power_sections) - 1):
        # Find wind speed data within the current wind power interval
        mask = (wind_power >= wind_power_sections[i]) & (wind_power < wind_power_sections[i + 1])
        wind_speed_interval = wind_speed[mask]

        if len(wind_speed_interval) != 0:
            # Calculate quartiles and quartile distance
            Q2_1i = np.quantile(wind_speed_interval, 0.25)
            Q2_3i = np.quantile(wind_speed_interval, 0.75)
            I2_QRi = Q2_3i - Q2_1i

            # Calculate lower and upper limits of abnormal inner limit of wind speed
            Vli = Q2_1i - 1.5 * I2_QRi
            Vui = Q2_3i + 1.5 * I2_QRi

            # Find abnormal data indices and append to list
            id = np.where((wind_speed_interval < Vli) | (wind_speed_interval > Vui))[0]
            abnormal_indices.extend(np.where(mask)[0][id])

    return abnormal_indices


def basic_trend_transition_points(i_p):
    """
    基础转折点
    :param i_p: 功率曲线
    :return:
    """
    basic_power = []
    basic_power.append(i_p[0])

    for i, p_i in enumerate(i_p):
        if i != 0 and i != len(i_p) - 1:
            delta0 = i_p[i, 1] - i_p[i - 1, 1]
            delta1 = i_p[i + 1, 1] - i_p[i, 1]
            delta_mul = delta0 * delta1
            delta_plu = delta0 + delta1
            if delta_mul <= 0:
                if delta_plu != 0:
                    basic_power.append(i_p[i])

    basic_power.append(i_p[-1])

    return np.array(basic_power)


def relative_vertical_distance(p_it, p_j_bt, p_j1_bt):
    a = 6 * np.abs(p_it[1] + (p_j1_bt[1] - p_it[1]) *
                   (p_j_bt[0] - p_it[0]) / (p_j1_bt[0] - p_it[0]) - p_j_bt[1])
    b = p_it[1] + p_j_bt[1] + p_j1_bt[1]

    return a / b


def important_trend_transition_points(i_p):
    # 重要转折点
    important_power = []
    important_power.append(i_p[0])
    for i, p_i in enumerate(i_p):
        if i != 0 and i != len(i_p) - 1:
            p_IT = important_power[-1]
            p_j_BT = p_i
            p_j1_BT = i_p[i + 1]
            RD = relative_vertical_distance(p_IT, p_j_BT, p_j1_BT)
            delta_p = np.abs((p_j1_BT[1] - p_j_BT[1]) / (p_IT[1] - p_j_BT[1] + 1e-8))

            if RD >= 0.18:
                if np.abs((p_IT[1] - p_j_BT[1])) != 0:
                    if delta_p >= 0.01:
                        important_power.append(p_j_BT)
                else:
                    if (p_j1_BT[1] - p_j_BT[1]) != 0:
                        important_power.append(p_j_BT)

    important_power.append(i_p[-1])
    return np.array(important_power)


def suspected_power_limit_data(p_i):
    ind = p_i[:, 0]
    ind_left_list, ind_right_list = [], []
    gammar = 10
    for i, id in enumerate(ind):
        if i != 0:
            left = ind[i - 1]
            right = id
            L = right - left
            if L >= gammar:
                ind_left_list.append(left)
                ind_right_list.append(right)

    ind_left_numpy = np.array(ind_left_list)
    ind_right_numpy = np.array(ind_right_list)
    return ind_left_numpy, ind_right_numpy


def power_limit_get(left, right, w_p):
    scaler = StandardScaler()
    scaler.fit_transform(w_p)
    # 记录横向的索引值
    limit_list = []
    for _, (a, b) in enumerate(zip(left, right)):
        left_ind = np.int64(a)
        right_ind = np.int64(b)
        delta = 0.2
        wind = w_p[left_ind:right_ind + 1, 0]
        power = w_p[left_ind:right_ind + 1, 1]
        wp_segment = np.concatenate([wind.reshape(-1, 1), power.reshape(-1, 1)], axis=1)
        wp_zscore = scaler.transform(wp_segment)

        # 对数据进行线性拟合
        coefficients = np.polyfit(wp_zscore[:, 0], wp_zscore[:, 1], 1)

        # 获取拟合的斜率和截距
        slope = coefficients[0]
        if slope < delta:
            limit_list.append([left_ind, right_ind])

    limit_numpy = np.array(limit_list)

    return limit_numpy


def removal_rated_power(w_p, id_numpy, rated_power=2000):
    phi = 0.9

    # 限功率索引
    index_limited = []
    for _, (a, b) in enumerate(id_numpy):
        power = w_p[a:b, 1]
        index = a + np.where(power <= rated_power * phi)
        index_limited.append(index[0])

    return np.concatenate(index_limited)


def bidirectional_quartile(w_p):
    w = w_p[:, 0]
    p = w_p[:, 1]
    abnormal_indices_longitudinal = longitudinal_quartile_algorithm(w, p)
    abnormal_indices_lateral = lateral_quartile_algorithm(w, p)

    # Combine abnormal indices from both algorithms
    abnormal_indices = list(set(abnormal_indices_longitudinal) | set(abnormal_indices_lateral))
    # 创建布尔索引，选择不在排除列表中的索引
    keep_indices = np.logical_not(np.isin(np.arange(len(w_p)), abnormal_indices))

    # 使用布尔索引从数组中选择值
    result = w_p[keep_indices]
    return result


def timing_mismatching_bi_quartile(wp):
    """
    找到功率曲线中的转折点
    :param wp: 风功率数据
    :return: 重要趋势转折点
    """
    power = wp[:, 1]
    index_list = np.array(range(len(power)))
    # 带有索引的功率曲线
    index_power = np.concatenate([index_list.reshape(-1, 1), power.reshape(-1, 1)], axis=1)

    basic_point = basic_trend_transition_points(index_power)
    important_point = important_trend_transition_points(basic_point)
    l_array, r_array = suspected_power_limit_data(important_point)
    limit_index = power_limit_get(l_array, r_array, wp)
    ind_limited = removal_rated_power(wp, limit_index)

    # 创建布尔索引，选择不在排除列表中的索引
    keep_indices = np.logical_not(np.isin(np.arange(len(wp)), ind_limited))
    # 限功率数据剔除
    result_remaining = wp[keep_indices]

    # 双边四分位法
    data_clear_after = bidirectional_quartile(result_remaining)
    return data_clear_after


if __name__ == '__main__':
    data_name = "gh_62"  # 替换为你需要的变量值

    # data_preprocess
    query0 = "SELECT 1秒平均风速, 有功功率 FROM " + data_name
    wp_get = ScadaRead_WP()
    wind_speed1, _ = wp_get.scada_data(query0, data_name)

    wind_speed1 = np.delete(wind_speed1, np.where(wind_speed1[:, 0] < 0), axis=0)
    wind_speed1 = np.delete(wind_speed1, np.where(wind_speed1[:, 1] < 0), axis=0)
    wind_speed1 = np.array(wind_speed1).astype(float)
    data = timing_mismatching_bi_quartile(wind_speed1)

    plt.scatter(data[:, 0], data[:, 1], label='Abnormal data',
                color=[246 / 255, 1 / 255, 1 / 255], s=2,
                zorder=2)
    plt.show()

    # 将每个数组的每行视为一个元素，然后转换为元组集合
    set_A = set(map(tuple, wind_speed1))
    set_B = set(map(tuple, data))
    # 从集合中找到A中独有的元素
    unique_elements_in_A = set_A - set_B
    # 将结果转换为NumPy数组
    result = np.array(list(unique_elements_in_A))

    # 绘图
    fig = plt.figure(figsize=(8, 6))  # 设置画布大小
    plt.scatter(wind_speed1[:, 0], wind_speed1[:, 1], label='Normal data', color=[0 / 255, 52 / 255, 245 / 255], s=2,
                zorder=1)
    plt.scatter(result[:, 0], result[:, 1], label='Abnormal data',
                color=[246 / 255, 1 / 255, 1 / 255], s=2,
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
    # fig.savefig(r"C:\Users\admin\Desktop\time.png", format='png', dpi=600)


