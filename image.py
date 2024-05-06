import numpy as np
import matplotlib.pyplot as plt
from db_raed import ScadaRead_WP
from matplotlib import rcParams

config = {
    "font.family": 'serif',
    "font.size": 18,  # 相当于小四大小
    "mathtext.fontset": 'stix',  # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
    "font.serif": ['Times New Roman'],  # 宋体
    'axes.unicode_minus': False  # 处理负号，即-号
}
rcParams.update(config)

def rasterize_WPC(WPC_data, delta_x=0.2, delta_y=7):
    v_max, v_min = np.max(WPC_data[:, 0]), np.min(WPC_data[:, 0])
    p_max, p_min = np.max(WPC_data[:, 1]), np.min(WPC_data[:, 1])

    M = int(np.floor((v_max - v_min) / delta_x) + 1)
    N = int(np.floor((p_max - p_min) / delta_y) + 1)
    binary_image = np.zeros((M, N), dtype=np.uint8)
    for v, p in WPC_data:
        x_i = int(np.floor((v - v_min) / delta_x))
        y_i = int(np.floor((p - p_min) / delta_y))
        binary_image[x_i, y_i] = 1
        # Display binary image

    return binary_image


def generate_feature_image(binary_image, t=3):
    # Get dimensions of the binary image
    height, width = binary_image.shape

    # Initialize feature image with zeros
    feature_image = np.zeros_like(binary_image, dtype=np.uint8)

    # Define eight neighborhood
    D = [(di, dj) for di in [-1, 0, 1] for dj in [-1, 0, 1] if (di != 0 or dj != 0)]

    # Iterate over each pixel in the binary image
    for i in range(height):
        for j in range(width):
            if binary_image[i, j] == 1:
                distances = []
                # Iterate over eight neighbors
                for di, dj in D:
                    count = 1
                    new_i, new_j = i + di, j + dj
                    # Check if the new pixel is within the image boundaries
                    if not (0 <= new_i < height and 0 <= new_j < width):
                        distances.append(count)
                        continue
                    # Check if the new pixel belongs to Q
                    if binary_image[new_i, new_j] != 1:
                        distances.append(count)
                        continue
                    # Traverse in the current direction until encountering a pixel that doesn't belong to Q
                    while binary_image[new_i, new_j] == 1:
                        count += 1
                        new_i, new_j = new_i + di, new_j + dj
                        # Check if the new pixel is within the image boundaries
                        if not (0 <= new_i < height and 0 <= new_j < width):
                            break
                    distances.append(count)

                # Calculate the median of the eight distances
                intensity = np.median(distances)

                # Assign the intensity to the feature image
                feature_image[i, j] = intensity
    result = np.where(feature_image > 3, 1, 0)
    return result


def horizontal_linear_feature(w):
    row = np.sum(w, axis=1)
    columns = np.sum(w, axis=0)
    w = row / (np.sum(row) + 1e-8)
    return np.sum(w * row ** 2) - np.sum(columns ** 2) / len(w)


def sliding_window_clearing(data, window_size=9):
    stride = 1  # 步长与窗口宽度一致

    xi, yj = data.shape[0], data.shape[1]
    Q = []
    horizontal_detection_list = []
    for i in range(0, xi - stride + 1, stride):
        for j in range(0, yj - stride + 1, stride):
            window = data[i:i + stride, j:j + stride]
            q = horizontal_linear_feature(window)
            Q.append(q)
            horizontal_detection_list.append(np.array([i, j]))

    mean = np.mean(Q)
    indices = np.where(Q > mean)[0]  # 获取符合条件的索引数组
    id_abnormal = np.array(horizontal_detection_list)[indices]

    for ii, jj in id_abnormal:
        center = np.int64(stride / 2)
        data[ii:ii + center, jj:jj + center] = 0

    return data


def data_location(WPC_data, wp_image, delta_x=0.2, delta_y=7):
    clear_set = []
    v_max, v_min = np.max(WPC_data[:, 0]), np.min(WPC_data[:, 0])
    p_max, p_min = np.max(WPC_data[:, 1]), np.min(WPC_data[:, 1])
    for v, p in WPC_data:
        x_i = int(np.floor((v - v_min) / delta_x))
        y_i = int(np.floor((p - p_min) / delta_y))
        if wp_image[x_i, y_i] == 1:
            clear_set.append(np.array([v, p]))

    return np.vstack(clear_set)


def image_thresholding(wp):
    binary_image0 = rasterize_WPC(wp)
    feature_image = generate_feature_image(binary_image0)

    # 横向检测数据
    clear_image = sliding_window_clearing(feature_image)
    # plt.imshow(clear_image, cmap='gray')
    # plt.title('Binary WPC Image')
    # plt.show()

    return data_location(wp, clear_image)


if __name__ == '__main__':
    data_name = "B13_201706"  # 替换为你需要的变量值

    # data_preprocess
    query0 = "SELECT 30秒平均风速, 有功功率 FROM " + data_name
    wp_get = ScadaRead_WP()
    wind_speed1, _ = wp_get.scada_data(query0, data_name)

    wind_speed1 = np.delete(wind_speed1, np.where(wind_speed1[:, 0] < 0), axis=0)
    wind_speed1 = np.delete(wind_speed1, np.where(wind_speed1[:, 1] < 0), axis=0)
    data_clear_after = image_thresholding(wind_speed1)

    # 绘图
    fig = plt.figure(figsize=(8, 6))  # 设置画布大小
    plt.scatter(wind_speed1[:, 0], wind_speed1[:, 1], label='Abnormal data', color=[246 / 255, 1 / 255, 1 / 255], s=2,
                zorder=1)
    plt.scatter(data_clear_after[:, 0], data_clear_after[:, 1], label='Normal data',
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
    fig.savefig(r"C:\Users\admin\Desktop\Image_t.png", format='png', dpi=600)
