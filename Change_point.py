from db_raed import ScadaRead_WP
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

config = {
    "font.family": 'serif',
    "font.size": 18,  # 相当于小四大小
    "mathtext.fontset": 'stix',  # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
    "font.serif": ['Times New Roman'],  # 宋体
    'axes.unicode_minus': False  # 处理负号，即-号
}
rcParams.update(config)

def calculate_variance(data):
    # 初始化一个列表来存储每个y的方差
    variances = []
    # 遍历每个y，计算方差
    for i in range(len(data)):
        # 计算方差公式中的每项
        sum_of_squares = np.sum((data[:i] - np.mean(data[:i])) ** 2)
        variance = sum_of_squares / (i + 1)
        variances.append(variance)
    return variances


def fit_and_calculate_errors(x1, y1):
    # 使用polyfit进行多项式拟合
    p = np.polyfit(x1, y1, deg=1)  # 这里选择1次多项式拟合
    y_pred = np.polyval(p, x1)

    # 计算预测值与实际值之间的误差
    errors = y1 - y_pred
    # plt.scatter(x1, y1)
    # plt.scatter(x1, y_pred)
    # plt.show()
    sum_error = np.sum(errors ** 2)

    return sum_error


# 定义误差函数
def error_function(j, before, x0, y0):
    error1 = fit_and_calculate_errors(x0[before:j], y0[before:j])
    error2 = fit_and_calculate_errors(x0[j:], y0[j:])
    return error1 + error2


def fit_segments(x1, y1):
    n = np.int64(len(x1) / 3)
    min_error = np.inf
    best_j = n

    for j in range(n + 1, len(x1)):
        result = error_function(j, n, x1, y1)
        # 如果找到更小的误差，则更新最优参数和分割点
        if result < min_error:
            min_error = result
            # best_params = result.x
            best_j = j
    # plt.plot(x1, y1)
    # plt.scatter(best_j, y1[best_j])
    # plt.scatter(n, y1[n])
    #
    # plt.show()
    return best_j


def partition(data, intervals=0.2):
    w_min = np.min(data[:, 0])
    w_max = np.max(data[:, 0])
    par_num = np.int64(np.floor((w_max - w_min) / intervals))

    boundaries = np.linspace(w_min, w_max, par_num)
    add_new_par = []
    segments = np.digitize(data[:, 0], boundaries)
    for i in range(1, len(boundaries)):
        segment_data = data[segments == i]
        if len(segment_data) > 0:
            add_new_par.append(segment_data)

    return add_new_par


def detect_outliers(data):
    # 计算第一和第三四分位数
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)

    # 计算四分位距
    iqr = q3 - q1

    # 定义异常值的上下限
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # 标记异常值
    outliers_mask = (data < lower_bound) | (data > upper_bound)

    return outliers_mask


def change_point_grouping_Q(wp):
    data_par = partition(wp, intervals=0.2)
    clear = []
    for _, d in enumerate(data_par):
        if len(d) > 5:
            x = d[:, 0]
            y = d[:, 1]
            # 将数据按照y的大小排序
            sort_index = np.argsort(y)[::-1]
            x = x[sort_index]
            y = y[sort_index]

            # 计算每一个y方差x
            s = calculate_variance(y)
            # 计算方差变化率
            k = np.abs(np.diff(s))
            # 分段拟合
            best_j = fit_segments(np.arange(0, len(s) - 1), k)

            result = np.column_stack((x[:best_j], y[:best_j]))

        else:
            result = d

        # 检测异常值
        outliers_mask = detect_outliers(result[:, 1])
        # 保留正常值
        normal_values = result[~outliers_mask]
        # print(normal_values.shape)
        clear.append(normal_values)
    return np.vstack(clear)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_name = "gh_67"  # 替换为你需要的变量值

    # data_preprocess
    query0 = "SELECT 1秒平均风速, 有功功率 FROM " + data_name
    wp_get = ScadaRead_WP()
    wind_speed1, _ = wp_get.scada_data(query0, data_name)

    wind_speed1 = np.delete(wind_speed1, np.where(wind_speed1[:, 0] < 0), axis=0)
    wind_speed1 = np.delete(wind_speed1, np.where(wind_speed1[:, 1] < 0), axis=0)

    # 将Decimal对象转换为float类型并转换为NumPy数组
    wind_speed1 = np.array(wind_speed1).astype(float)
    data_after = change_point_grouping_Q(wind_speed1)

    # 绘图
    fig = plt.figure(figsize=(8, 6))  # 设置画布大小
    plt.scatter(wind_speed1[:, 0], wind_speed1[:, 1], label='Abnormal data', color=[246 / 255, 1 / 255, 1 / 255], s=2,
                zorder=1)
    plt.scatter(data_after[:, 0], data_after[:, 1], label='Normal data', color=[0 / 255, 52 / 255, 245 / 255], s=2,
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
    # fig.savefig(r"C:\Users\admin\Desktop\Chang_point_fig.png", format='png', dpi=600)
