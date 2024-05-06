import numpy as np
import matplotlib.pyplot as plt
from Change_point import change_point_grouping_Q
from Timing_matching_biQuartile import timing_mismatching_bi_quartile
import sys
sys.path.append('../new_paper')  # 将A文件夹所在的路径添加到Python解释器的搜索路径中
from image import image_thresholding
from module import Ttlof
from Curve_power_clearing import curve_ieee_access
from Density_Boundray import density_clustering_boundary_extraction as DC
import new_paper.full_working_condition_boundary as ours
from db_raed import ScadaRead_WP
from CFSFDP_clearing import CFSFDP
import time
import pandas as pd
from RANSAC_DBSCAN_clear01 import ransac_dbscan

def metrics(file_path_p, file_path_n, data_after, T_0, T_1):
    """

    :param intersection_n_num: 未识别出的异常数量
    :param intersection_p_num: 识别出的正常数量
    :param P_num: 原始正常数量
    :param N_num: 原始异常数量
    :return: P, R, F1
    """
    # 加载数据到文件
    WP_normal = np.load(file_path_p, allow_pickle=True)
    WP_abnormal = np.load(file_path_n, allow_pickle=True)

    WP_normal = np.delete(WP_normal, np.where(WP_normal[:, 0] < 0), axis=0)
    WP_normal = np.delete(WP_normal, np.where(WP_normal[:, 1] < 0), axis=0)

    WP_abnormal = np.delete(WP_abnormal, np.where(WP_normal[:, 0] < 0), axis=0)
    WP_abnormal = np.delete(WP_abnormal, np.where(WP_normal[:, 1] < 0), axis=0)
    # 将每个数组的每行视为一个元素，然后转换为元组集合
    set_P = set(map(tuple, WP_normal))
    set_N = set(map(tuple, WP_abnormal))
    set_B = set(map(tuple, data_after))

    # 从集合中找到B中独有的元素
    # 取集合的交集
    intersection_n = set_N.intersection(set_B)
    intersection_p = set_P.intersection(set_B)

    intersection_n_num, intersection_p_num, P_num, N_num = len(intersection_n), \
        len(intersection_p), len(set_P), len(set_N)

    TP = intersection_p_num
    FP = intersection_n_num
    TN = N_num - intersection_n_num
    FN = P_num - TP

    # 异常值检测率
    P_n = TN / N_num
    P = TP / (TP + FP + 1e-8)
    R = TP / (TP + FN + 1e-8)
    F1 = 2 * (P * R) / (P + R + 1e-8)
    # 计算运行时间
    execution_time = T_1 - T_0
    return P, R, F1, P_n, execution_time


if __name__ == '__main__':
    # str_list = ['B10_', 'B11_', 'B12_', 'B13_', 'B14_', 'B15_', 'B16_', 'B17_']
    str_list = ['B13_']
    year = '2017'
    results = []
    try:
        for k in str_list:
            for i in range(1, 13):
                # 使用格式化字符串 '{:02d}' 来保证数字是两位的，并且在数字前面补零
                data_name = k + year + '{:02d}'.format(i)  # 替换为你需要的变量值
                print(data_name)
                file_path_norm = f"F:/pycharm_project/WP_normal_and_abnormal/{data_name + 'normal'}.npy"
                file_path_abn = f"F:/pycharm_project/WP_normal_and_abnormal/{data_name + 'abnormal'}.npy"

                # data_preprocess
                query0 = "SELECT 30秒平均风速, 有功功率 FROM " + data_name
                wp_get = ScadaRead_WP()
                wind_speed1, flag = wp_get.scada_data(query0, data_name)
                if flag:
                    wind_speed1 = np.delete(wind_speed1, np.where(wind_speed1[:, 0] < 0), axis=0)
                    wind_speed1 = np.delete(wind_speed1, np.where(wind_speed1[:, 1] < 0), axis=0)

                    # 开始计时
                    start_time = time.time()
                    data = timing_mismatching_bi_quartile(wind_speed1)
                    data = data[:, :2]
                    # 结束计时
                    end_time = time.time()

                    P, R, F1, P_n, execution_time = metrics(file_path_norm, file_path_abn, data, start_time, end_time)
                    data_store = [data_name, P, R, F1, P_n, execution_time]
                    results.append(data_store)
    except Exception as e:
        print(f"An error occurred: {e}")

    # Convert results to DataFrame
    df = pd.DataFrame(results,
                      columns=['Data Name', 'Precision', 'Recall', 'F1 Score', 'P_n', 'Execution Time'])

    # Save DataFrame to Excel
    df.to_excel(r"C:\Users\admin\Desktop\results.xlsx", index=False)
