# import matplotlib.pyplot as plt
#
# # 数据
# hop = [1, 2, 3, 4]
# svm_accuracy = [0.529, 0.741, 0.897, 0.950]
# lightgbm_accuracy = [0.610, 0.735, 0.890, 0.939]
# xgboost_accuracy = [0.564, 0.721, 0.882, 0.922]
#
# # 创建折线图
# plt.plot(hop, svm_accuracy, marker='^', color='blue', label='SVM')
# plt.plot(hop, lightgbm_accuracy, marker='o', color='red', label='LightGBM')
# plt.plot(hop, xgboost_accuracy, marker='s', color='green', label='Random forest')
#
# # 添加数据标签
# # for i, acc in enumerate(svm_accuracy):
# #     plt.text(hop[i], acc, f'{acc:.3f}', ha='left', va='bottom', color='blue')
# #
# # for i, acc in enumerate(lightgbm_accuracy):
# #     plt.text(hop[i], acc, f'{acc:.3f}', ha='left', va='bottom', color='red')
# #
# # for i, acc in enumerate(xgboost_accuracy):
# #     plt.text(hop[i], acc, f'{acc:.3f}', ha='left', va='bottom', color='green')
#
# # 添加标签和标题
# plt.xlabel('hop')
# plt.ylabel('Accuracy')
# # plt.title('')
# plt.xticks(hop)
# plt.legend()
#
# # 显示图形
# plt.grid(True) # 可以选择是否显示网格
# plt.show()
#
#
# 柱状图
# import matplotlib.pyplot as plt
# import numpy as np
#
# # 数据
# hop = np.array([1, 2, 3, 4])
# svm_accuracy = np.array([0.711, 0.811, 0.873, 0.950])
# lightgbm_accuracy = np.array([0.690, 0.835, 0.890, 0.971])
# xgboost_accuracy = np.array([0.704, 0.861, 0.912, 0.958])
#
# bar_width = 0.2
# opacity = 0.8
#
# # 绘制柱状图
# plt.bar(hop - bar_width, svm_accuracy, bar_width,
#         alpha=opacity,
#         color='blue',
#         label='SVM')
#
# plt.bar(hop, lightgbm_accuracy, bar_width,
#         alpha=opacity,
#         color='red',
#         label='LightGBM')
#
# plt.bar(hop + bar_width, xgboost_accuracy, bar_width,
#         alpha=opacity,
#         color='green',
#         label='XGBoost')
#
# # 添加数据标签
# for i, acc in enumerate(svm_accuracy):
#     plt.text(hop[i] - bar_width, acc, f'{acc:.3f}', ha='center', va='bottom', color='blue')
#
# for i, acc in enumerate(lightgbm_accuracy):
#     plt.text(hop[i], acc, f'{acc:.3f}', ha='center', va='bottom', color='red')
#
# for i, acc in enumerate(xgboost_accuracy):
#     plt.text(hop[i] + bar_width, acc, f'{acc:.3f}', ha='center', va='bottom', color='green')
#
# # 添加标签和标题
# plt.xlabel('hop')
# plt.ylabel('AUC')
# # plt.title('Accuracy vs. hop for Different Models')
# plt.xticks(hop)
# plt.legend()
#
# # 显示图形
# # plt.grid(axis='y', linestyle='--') # 可以选择是否显示水平网格
# plt.tight_layout()
# plt.show()

# 混淆矩阵
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
#
# # 真实标签 (按图中顺序)
# true_labels = ['Avaddon', 'Babuk', 'Balaclava', 'BlackMatter', 'Cerber', 'CryLock',
#                'Crysis', 'DarkSide', 'DearCrypt', 'Egregor', 'GandCrab', 'LockBit',
#                'MedusaLocker', 'Mespinoza', 'Phobos', 'QNAPCrypt', 'RagnarLocker',
#                'Ranzy', 'Sage', 'SARA', 'Snatch', 'Sodinokibi']
#
# # 预测标签 (按图中顺序)
# predicted_labels = true_labels  # 在混淆矩阵中，预测标签通常与真实标签相同
#
# # 混淆矩阵数据 (从图片中提取)
# cm_data = np.array([
#     [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 7, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 1, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]
# ])
#
# # 设置字体以显示中文标签
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 或其他支持中文的字体
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
#
# # 绘制混淆矩阵
# plt.figure(figsize=(10, 8))
# sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', cbar=True,
#             xticklabels=predicted_labels, yticklabels=true_labels)
#
# # 添加标签和标题
# plt.xlabel('预测分类')
# plt.ylabel('真实分类')
# plt.title('混淆矩阵')
#
# # 调整刻度标签的旋转角度，防止重叠
# plt.xticks(rotation=90)
# plt.yticks(rotation=0)
#
# # 显示图形
# plt.tight_layout()
# plt.show()

# 散点图
# import matplotlib.pyplot as plt
# import numpy as np
#
# # 为了结果的可重复性
# np.random.seed(42)
#
# # 类别名称列表 (与混淆矩阵的顺序一致)
# categories = ['Avaddon', 'Babuk', 'Balaclava', 'BlackMatter', 'Cerber', 'CryLock',
#               'Crysis', 'DarkSide', 'DearCrypt', 'Egregor', 'GandCrab', 'LockBit',
#               'MedusaLocker', 'Mespinoza', 'Phobos', 'QNAPCrypt', 'RagnarLocker',
#               'Ranzy', 'Sage', 'SARA', 'Snatch', 'Sodinokibi']
#
# # 混淆矩阵数据
# cm_data = np.array([
#     [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 7, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 1, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]
# ])
#
# # 提取每个类别的点数 (混淆矩阵的对角线元素)
# n_points_per_category = np.diag(cm_data)
#
# # 为每个类别随机生成数据，并大致按照图中分布调整中心和形状
# data = {}
# for i, category in enumerate(categories):
#     n_points = n_points_per_category[i] * 5 # 乘以一个因子以增加可见性
#     if n_points > 0:
#         if category == 'Avaddon':
#             data[category] = np.random.normal(loc=[-60, -20], scale=[10, 10], size=(n_points, 2))
#         elif category == 'Babuk':
#             data[category] = np.random.normal(loc=[-30, 10], scale=[8, 8], size=(n_points, 2))
#         elif category == 'Balaclava':
#             data[category] = np.random.normal(loc=[-20, 20], scale=[7, 7], size=(n_points, 2))
#         elif category == 'BlackMatter':
#             data[category] = np.random.normal(loc=[0, 30], scale=[12, 12], size=(n_points, 2))
#         elif category == 'Cerber':
#             data[category] = np.random.normal(loc=[30, 60], scale=[5, 5], size=(n_points, 2))
#         elif category == 'CryLock':
#             data[category] = np.random.normal(loc=[65, 40], scale=[6, 6], size=(n_points, 2))
#         elif category == 'Crysis':
#             data[category] = np.random.normal(loc=[60, 30], scale=[8, 8], size=(n_points, 2))
#         elif category == 'DarkSide':
#             data[category] = np.random.normal(loc=[70, 10], scale=[4, 4], size=(n_points, 2))
#         elif category == 'DearCrypt':
#             data[category] = np.random.normal(loc=[75, 0], scale=[3, 3], size=(n_points, 2))
#         elif category == 'Egregor':
#             data[category] = np.random.normal(loc=[70, -10], scale=[5, 5], size=(n_points, 2))
#         elif category == 'GandCrab':
#             data[category] = np.random.normal(loc=[20, 60], scale=[6, 6], size=(n_points, 2))
#         elif category == 'LockBit':
#             data[category] = np.random.normal(loc=[-5, -60], scale=[10, 10], size=(n_points, 2))
#         elif category == 'MedusaLocker':
#             data[category] = np.random.normal(loc=[40, -50], scale=[8, 8], size=(n_points, 2))
#         elif category == 'Mespinoza':
#             data[category] = np.random.normal(loc=[20, -30], scale=[7, 7], size=(n_points, 2))
#         elif category == 'Phobos':
#             data[category] = np.random.normal(loc=[0, -70], scale=[9, 9], size=(n_points, 2))
#         elif category == 'QNAPCrypt':
#             data[category] = np.random.normal(loc=[-50, -30], scale=[12, 12], size=(n_points, 2))
#         elif category == 'RagnarLocker':
#             data[category] = np.random.normal(loc=[-65, 5], scale=[9, 9], size=(n_points, 2))
#         elif category == 'Ranzy':
#             data[category] = np.random.normal(loc=[-40, 40], scale=[7, 7], size=(n_points, 2))
#         elif category == 'Sage':
#             data[category] = np.random.normal(loc=[10, -65], scale=[6, 6], size=(n_points, 2))
#         elif category == 'SARA':
#             data[category] = np.random.normal(loc=[50, 15], scale=[5, 5], size=(n_points, 2))
#         elif category == 'Snatch':
#             data[category] = np.random.normal(loc=[-10, -45], scale=[8, 8], size=(n_points, 2))
#         elif category == 'Sodinokibi':
#             data[category] = np.random.normal(loc=[35, -20], scale=[7, 7], size=(n_points, 2))
#
# # 为每个类别选择不同的颜色
# colors = plt.cm.get_cmap('viridis', len(categories))
# color_map = {category: colors(i) for i, category in enumerate(categories)}
#
# plt.figure(figsize=(12, 8))
#
# for i, category in enumerate(categories):
#     if category in data:
#         points = data[category]
#         x = points[:, 0]
#         y = points[:, 1]
#         color = color_map[category]
#         plt.scatter(x, y, label=category, s=20, alpha=0.7)
#
# # 添加标题和图例
# # plt.title('RotatE 嵌入可视化 (按混淆矩阵对角线值确定点数)')
# plt.xlabel('')
# plt.ylabel('')
# plt.legend(loc='upper right', fontsize='small', bbox_to_anchor=(1.05, 1))
# plt.subplots_adjust(right=0.75) # 调整右边距以适应图例
#
# # 设置坐标轴范围 (大致与图中一致)
# plt.xlim([-80, 100])
# plt.ylim([-80, 100])
#
# plt.grid(False)
# plt.tight_layout()
# plt.show()