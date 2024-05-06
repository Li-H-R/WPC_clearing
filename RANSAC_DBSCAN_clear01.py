
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor, LinearRegression  # RANSAC算法
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline  # 用于构建管道
from db_raed import ScadaRead_WP
import matplotlib.pyplot as plt
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


# 设定规则对数据进行清洗
def rule_based_cleaning(df, rated_power=2000):

    # 1.将有功功率、风速和转速中数值小于零的数据删除
    df1 = df[(df[:, 0] >= 0) & (df[:,1] >= 0) & (df[:,2] >= 0)]  # 0:风速 1:有功功率 2:转速
    # 2.将风速中小于等于3时有功功率大于零的数据删除
    df2 = df1[(df1[:,0] > 3) | (df1[:,1] <= 0)]
    # 3.将瞬时风速大于25时有功功率大于零的数据进行删除
    df3 = df2[(df2[:,0] <= 25) | (df2[:,1] <= 0)]
    # 4.将功率大于额定功率2000的1.2倍的数据删除
    clear_01 = df3[(df3[:,1] <= 1.2 * rated_power)]
    # 5.将转速超过历史最大值的1.2倍的数据删除
    clear_01 = clear_01[(clear_01[:,2] <= 1.2 * 15)]  # 假定15是历史最大值
    # 6.将转速低于历史最小值的0.8倍的数据删除
    clear_01 = clear_01[(clear_01[:, 2] >= 0.8 * 0.05)]  # 假定0.05是历史最小值
    return clear_01


# 基于RANSAC算法的数据清洗
def ransac_model(data_02):

    poly = PolynomialFeatures(degree=3)  # 三次多项式
    X_poly = poly.fit_transform(data_02[:, [1, 0]])  # 需要将功率（第二列）作为第一个特征，风速（第一列）作为第二个特征
    # 2.构建RANSAC模型预测风速值
    ransac = RANSACRegressor(estimator=make_pipeline(PolynomialFeatures(degree=3), RANSACRegressor()),
                             min_samples=0.8)  # 0.8表示使用80%的数据进行拟合
    ransac.fit(X_poly, data_02[:, 0])

    # 3.预测风速值
    predict = ransac.predict(X_poly)
    # 4.计算预测风速值与真实风速值的残差绝对值
    T = np.abs(predict - data_02[:, 0])
    # 5.根据预先设定的阈值对残差进行筛选
    threshold = 2.8  # 假定阈值为2.8
    clear_02 = data_02[T <= threshold]
    return clear_02


# 使用DBSCAN算法对异常数据进行清洗
def dbscan_model(df):
    # 1.数据标准化
    data_03 = df.copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(data_03[:, :2])
    # 2.构建DBSCAN模型
    # 使用DBSCAN算法进行聚类，其中的参数使用遍历进行确定
    # for eps in [0.06, 0.07, 0.08, 0.09, 0.1]:
    #     for min_samples in [25, 30, 35, 40]:
    #         dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    #         dbscan.fit(X)
    #         # 将聚类结果可视化
    #         plt.figure(dpi=500)
    #         plt.scatter(X[:, 1], X[:, 0], s=1, c=dbscan.labels_)
    #         plt.xlabel('风速(m/s)')
    #         plt.ylabel('功率(kW)')
    #         plt.title('DBSCAN聚类结果-eps=' + str(eps) + ',min_samples=' + str(min_samples))
    #         plt.show()
    # 选取参数为eps=0.06,min_samples=40对数据进行聚类清洗
    dbscan = DBSCAN(eps=0.06, min_samples=40)
    dbscan.fit(X)
    # 将其中数量最多的类别作为正常数据点
    clear_03 = data_03[dbscan.labels_ == 0]
    return clear_03

def ransac_dbscan(wp):
    clear_01 = rule_based_cleaning(wp)
    clear_02 = ransac_model(clear_01)
    clear_03 = dbscan_model(clear_02)
    return clear_03

if __name__ == '__main__':
    data_name = "GH_60"  # 替换为你需要的变量值

    # data_preprocess
    query0 = "SELECT 30秒平均风速, 有功功率, 风轮转速 FROM " + data_name
    wp_get = ScadaRead_WP()
    wind_speed1, _ = wp_get.scada_data(query0, data_name)

    wind_speed1 = np.array(wind_speed1).astype(float)
    wind_speed1 = np.delete(wind_speed1, np.where(wind_speed1[:, 0] < 0), axis=0)
    wind_speed1 = np.delete(wind_speed1, np.where(wind_speed1[:, 1] < 0), axis=0)
    clear_data = ransac_dbscan(wind_speed1)

    # 绘图
    fig = plt.figure(figsize=(8, 6))  # 设置画布大小
    plt.scatter(wind_speed1[:, 0], wind_speed1[:, 1], label='Abnormal data', color=[246 / 255, 1 / 255, 1 / 255], s=2,
                zorder=1)
    plt.scatter(clear_data[:, 0], clear_data[:, 1], label='Normal data',
                color=[0 / 255, 52 / 255, 245 / 255], s=2,
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
    fig.savefig(r"C:\Users\admin\Desktop\RANSAC.png", format='png', dpi=600)