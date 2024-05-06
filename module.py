#编写一个名为TTLOF的文件, 该文件包含一个名为TTLOF的函数, 该函数接受一个形状为(n, 2)的数组wind, 并返回一个形状为(n, 2)的数组wind_safe, 该数组包含wind中的正常值.
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
from scipy.stats import t
def Ttlof(wind):
    #t-distribution筛除异常点
    step1=Tdist(wind)



    wind_safe =step1
    return wind_safe


def Tdist(wind,s = 100):
    #数据分组：根据风速将输出功率、发电机转速和桨叶角度分成s个部分。设
    #对于二维数组(n,2)加入时间序列保存为二维数组(n,3)，之后依据风速依顺序分为s段，s为未知量

    sorted_data = wind[wind[:, 0].argsort()]
    sections = np.array_split(sorted_data, s)
    for i in range(s):
        a = 1
        while(a):
            #m为第i条sections的长度
            m = len(sections[i][:,0])
            # 计算统计量：对每个部分计算功率的均值、标准差和。
            mean = np.mean(sections[i][:, 1])
            #print(mean)
            std = np.std(sections[i][:, 1])
            #计算单个功率与平均值的差的绝对值数组
            abs_diff = np.abs(sections[i][:, 1] - mean)
            #print(mean,std,abs_diff,'/n')
            #最大绝对偏差，并记录它的位置
            max_abs_deviation = np.max(abs_diff)
            max_abs_deviation_index = np.argmax(abs_diff)
            #print(max_abs_deviation)
            # 计算 t 分布
            degrees_of_freedom = m - 1
            alpha = 0.01
            t_distribution = t.ppf(1 - alpha / 2, degrees_of_freedom,loc = mean,scale = std)
            # 计算 τ
            tau = t_distribution- mean

            if(max_abs_deviation >= tau):
                #删除最大偏差所在异常点
                sections[i] = np.delete(sections[i], max_abs_deviation_index, axis=0)
            else:
                a=0
    #合并sections数据
    #LOF算法
        sections[i]=LOF(sections[i])
    t_dist = np.concatenate(sections, axis=0)
    return t_dist


def LOF(wind):
    #LOF算法
    # 拟合模型以进行异常值检测（默认）
    clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    # 使用fit_predict计算训练样本的预测标签
    # （当LOF用于异常值检测时，估计器没有预测，
    # Decision_function 和 Score_samples 方法）。
    y_pred = clf.fit_predict(wind[:,0:1])

    #根据y_pred对wind进行筛选，wind是【n，3】数组
    wind_safe = wind[y_pred == 1]

    return wind_safe