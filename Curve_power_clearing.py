from db_raed import ScadaRead_WP
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
from matplotlib import rcParams

config = {
    "font.family": 'serif',
    "font.size": 18,  # 相当于小四大小
    "mathtext.fontset": 'stix',  # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
    "font.serif": ['Times New Roman'],  # 宋体
    'axes.unicode_minus': False  # 处理负号，即-号
}
rcParams.update(config)

def power_curve(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d


# 定义功率上限函数
def power_curve_upper(x, a_upper, b_upper, c_upper, d_upper):
    return a_upper * x ** 3 + b_upper * x ** 2 + c_upper * x + d_upper


# 定义功率下限函数
def power_curve_lower(x, a_lower, b_lower, c_lower, d_lower):
    return a_lower * x ** 3 + b_lower * x ** 2 + c_lower * x + d_lower


# 使用DBSCAN算法对数据进行清洗，去除低密度的异常值
def dbscan_clear(df):
    # 1.数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(df)
    # 2.构建DBSCAN模型
    # # 使用DBSCAN算法进行聚类，其中的参数使用遍历进行确定
    # for eps in [0.06, 0.07, 0.08, 0.09, 0.1]:
    #     for min_samples in [25, 30, 35]:
    #         dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    #         dbscan.fit(X)
    #         # 将聚类结果可视化
    #         plt.figure(dpi=500)
    #         plt.scatter(X[:, 0], X[:, 1], s=1, c=dbscan.labels_)
    #         plt.xlabel('风速(m/s)')
    #         plt.ylabel('功率(kW)')
    #         plt.title('DBSCAN聚类结果-eps=' + str(eps) + ',min_samples=' + str(min_samples))
    #         plt.show()
    # 选取参数为eps=0.06,min_samples=35对数据进行聚类清洗
    dbscan = DBSCAN(eps=0.06, min_samples=35)
    dbscan.fit(X)
    # 将其中数量最多的类别作为正常数据点
    clear_01 = df[dbscan.labels_ == 0]
    # # 对清洗后的数据进行可视化
    # plt.figure(dpi=500)
    # plt.scatter(clear_01[0], clear_01[1], s=1)
    # plt.xlabel('风速(m/s)')
    # plt.ylabel('功率(kW)')
    # plt.xticks(range(0, 35, 5))
    # plt.title('DBSCAN清洗01-风功率曲线')
    # plt.show()
    return clear_01


# 划分区间，对功率曲线进行拟合，得到功率上下限
def curve_power_01(data_02, power_max=2000):
    # 根据功率对数据进行区间划分
    # 1.将数据按有功功率列划分成区间
    power_min = np.min(data_02[:, 1])
    power_interval = (power_max - power_min) / 35  # 将功率划分成35个区间

    data_02_new = np.floor((data_02[:, 1] - power_min) / power_interval).astype(int)  # 将功率划分到对应的区间

    power_density = []
    wind_density = []
    # 根据区间对风速功率数据进行分组
    grouped = [data_02[data_02_new == i] for i in range(np.max(data_02_new)+1)]

    for group in grouped:
        if len(group) > 0:
            power_density.append(np.mean(group[:, 1]))
            wind_density.append(np.mean(group[:, 0]))

    # 将功率密度为0的第一个数据点替换为0
    power_density[0] = 0

    # 利用最小二乘法对密度中点进行三次多项式拟合
    # 1.输入数据
    x = wind_density
    y = power_density

    # 设置初始值
    p0 = [1, 1, 1, 1]
    # 2.拟合数据
    popt, p_cov = curve_fit(power_curve, x[:-2], y[:-2], p0)

    # 拟合后的参数
    a, b, c, d = popt
    # 生成拟合曲线上的数据点
    x_fit = np.linspace(min(x), max(x), 50)
    y_fit = power_curve(x_fit, a, b, c, d)
    # # 使拟合出的曲线在第5个数据点和倒数第7个点处进行截断
    x_fit = x_fit[4:-7]
    y_fit = y_fit[4:-7]

    # 计算出每个区间的风速标准差和功率标准差
    wind_std, power_std = [], []
    # 1.计算风速标准差和功率标准差
    for group in grouped:
        if len(group) > 1:  # 至少有两个元素才能计算标准差
            wind_std.append(np.std(group[:, 0]))
            power_std.append(np.std(group[:, 1]))

    # 将含有标记值的列表转换为NumPy数组
    wind_std = np.array(wind_std)
    power_std = np.array(power_std)
    # 求出所有风速标准差和功率标准差的均值
    wind_std_mean = np.mean(wind_std)
    power_std_mean = np.mean(power_std)
    # 2.根据3σ原则计算出每个区间的风速和功率的上下限
    wind_upper = wind_density + 3 * wind_std_mean
    wind_lower = wind_density - 3 * wind_std_mean
    power_upper = power_density + 3 * power_std_mean
    power_lower = power_density - 3 * power_std_mean
    # 将下限中的负值替换为0
    power_lower[power_lower < 0] = 0
    # 将上限中的大于额定功率的值替换为额定功率
    power_upper[power_upper > power_max] = 2150
    # 将上限中的第一个功率值替换为下限中的第一个功率值
    power_upper[0] = power_lower[0]
    # 对功率上下限进行拟合
    # 1.输入数据
    x_upper = wind_upper
    y_upper = power_upper
    x_lower = wind_lower
    y_lower = power_lower
    # 2.拟合数据
    popt_upper, p_cov_upper = curve_fit(power_curve, x_upper[:-3], y_upper[:-3], p0)
    popt_lower, p_cov_lower = curve_fit(power_curve, x_lower[:-3], y_lower[:-3], p0)
    # 拟合后的参数
    a_upper, b_upper, c_upper, d_upper = popt_upper
    a_lower, b_lower, c_lower, d_lower = popt_lower
    # 生成拟合曲线上的数据点
    x_fit_upper = np.linspace(min(x_upper), max(x_upper), 50)
    y_fit_upper = power_curve(x_fit_upper, a_upper, b_upper, c_upper, d_upper)
    x_fit_lower = np.linspace(min(x_lower), max(x_lower), 50)
    y_fit_lower = power_curve(x_fit_lower, a_lower, b_lower, c_lower, d_lower)
    # 使拟合出的曲线在适当数据点处进行截断
    x_fit_upper_01 = x_fit_upper[:-8]
    y_fit_upper_01 = y_fit_upper[:-8]
    x_fit_lower_01 = x_fit_lower[:-7]
    y_fit_lower_01 = y_fit_lower[:-7]

    return x_fit_upper_01, y_fit_upper_01, x_fit_lower_01, y_fit_lower_01, wind_density, power_density, data_02, grouped

# 在拟合出的上下两条曲线中找到每个区间的功率密度值对应的风速值
def last_clear(df, x_fit_upper, y_fit_upper, x_fit_lower, y_fit_lower, wind_density, power_density, grouped):
    # 1.将拟合曲线数据转化为插值函数
    f_upper = np.polyfit(y_fit_upper, x_fit_upper, 3)  # 3表示三次多项式拟合
    f_lower = np.polyfit(y_fit_lower, x_fit_lower, 3)  # 3表示三次多项式拟合

    # 2.计算每个区间的功率密度中点对应的风速值
    wind_upper_values = []
    wind_lower_values = []
    for i in range(len(power_density)):
        if i < len(power_density) - 1:

            wind_speed_upper = np.polyval(f_upper, power_density[i + 1])
            wind_speed_lower = np.polyval(f_lower, power_density[i + 1])
            wind_upper_values.append(wind_speed_upper)
            wind_lower_values.append(wind_speed_lower)
    # # 打印出每个区间的风速上下限和风速功率中点
    # for i in range(len(power_density) - 1):
    #     print('第{}个区间的风速上限为{}，风速下限为{}，风速功率中点为{}'.format(i, wind_upper_values[i],
    #                                                                           wind_lower_values[i], wind_density[i]))
    # # 从第二个区间开始，将每个区间内的风速值与风速上下限进行比较，将不在范围内的数据点进行清洗
    clear_02 = []
    for name, group in enumerate(grouped):
        if len(group) > 0:
            name = int(name)
            if name == 0 or name >= len(grouped)-2:
                clear_02.append(group)
            else:
                clear_02.append(
                    group[(group[:, 0] >= wind_lower_values[name - 1]) & (group[:, 0] <= wind_upper_values[name - 1])])

    clear_02 = np.concatenate(clear_02, axis=0)
    return clear_02


def curve_ieee_access(wp):
    clear_01 = dbscan_clear(wp)
    x_fit_upper, \
        y_fit_upper, \
        x_fit_lower, \
        y_fit_lower, \
        wind_density, \
        power_density, \
        data_02, grouped = curve_power_01(clear_01)

    clear_data = last_clear(data_02,
                            x_fit_upper,
                            y_fit_upper,
                            x_fit_lower,
                            y_fit_lower,
                            wind_density,
                            power_density,
                            grouped)

    return clear_data
# 定义三次多项式函数
def cubic_function(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

if __name__ == '__main__':
    data_name = "gh_67" # 替换为你需要的变量值

    # data_preprocess
    query0 = "SELECT 1秒平均风速, 有功功率 FROM " + data_name
    wp_get = ScadaRead_WP()
    wind_speed1, _ = wp_get.scada_data(query0, data_name)

    wind_speed1 = np.delete(wind_speed1, np.where(wind_speed1[:, 0] < 0), axis=0)
    wind_speed1 = np.delete(wind_speed1, np.where(wind_speed1[:, 1] < 0), axis=0)
    wind_speed1 = np.array(wind_speed1).astype(float)
    data_clear = curve_ieee_access(wind_speed1)


    # 绘图
    # fig = plt.figure(figsize=(8, 6))  # 设置画布大小
    # plt.scatter(wind_speed1[:, 0], wind_speed1[:, 1], label='Abnormal data', color=[246 / 255, 1 / 255, 1 / 255], s=2,
    #             zorder=1)
    # plt.scatter(data_clear[:, 0], data_clear[:, 1], label='Normal data',
    #             color=[0 / 255, 52 / 255, 245 / 255], s=2,
    #             zorder=2)
    #
    # # 绘制原始数据
    # popt, pcov = curve_fit(cubic_function, w, p)
    # # 绘制拟合曲线
    # x_fit = np.linspace(min(w), max(w), 100)
    # y_fit = cubic_function(x_fit, *popt)
    # plt.plot(x_fit, y_fit, color=[254 / 255, 215 / 255, 7 / 255], linestyle='-', zorder=3, label='Fitting curve')
    # plt.scatter(w, p, label='Fit centers', color=[246 / 255, 1 / 255, 1 / 255],
    #             marker='s', s=40, edgecolors=[0 / 255, 24 / 255, 134 / 255], zorder=4)
    #
    # # 设置图例
    # plt.legend(loc='best', fontsize=14)  # 放置图例在最合适的位置
    #
    # # 设置坐标轴标签和标题
    # plt.xlabel('Wind Speed(m/s)')
    # plt.ylabel('Wind Power(kW)')
    #
    # # 设置坐标轴范围
    # plt.xlim([0, 25])
    # plt.ylim([-100, 2300])
    #
    # plt.show()
    # fig.savefig(r"C:\Users\admin\Desktop\Curve.png", format='png', dpi=600)