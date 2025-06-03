import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
import random
import numpy as np

# ================= 数据定义 =================
central_api = "NtQueryInformation"

# 各层节点定义
node_hierarchy = {
    0: [central_api],
    1: [
        "NtCreateFile",
        "NtOpenProcess",
        "NtAllocateVirtualMemory",
        "NtWriteFile",
        "NtReadFile"
    ],
    2: {
        "NtCreateFile": ["CreateFileW", "CloseHandle"],
        "NtOpenProcess": ["OpenProcess", "TerminateProcess"],
        "NtAllocateVirtualMemory": ["VirtualAlloc", "VirtualProtect"],
        "NtWriteFile": ["WriteFile", "FlushFileBuffers"],
        "NtReadFile": ["ReadFile", "GetFileSize"]
    },
    3: {
        "CreateFileW": ["FindFirstFileW"],
        "OpenProcess": ["CreateToolhelp32Snapshot"],
        "VirtualAlloc": ["HeapAlloc"],
        "TerminateProcess": ["CreateRemoteThread"],
        "VirtualProtect": ["WriteProcessMemory"]
    },
    4: {
        "FindFirstFileW": ["FindNextFileW"],
        "CreateRemoteThread": ["ResumeThread"],
        "WriteProcessMemory": ["ReadProcessMemory"]
    }
}

# ================= 图谱构建 =================
G = nx.DiGraph()

# 添加节点并设置层级属性
for layer in node_hierarchy:
    if layer == 0:
        for node in node_hierarchy[0]:
            G.add_node(node, layer=0, size=500)
    elif layer == 1:
        for node in node_hierarchy[1]:
            G.add_node(node, layer=1, size=300)
    else:
        for parent, children in node_hierarchy[layer].items():
            # 确保父节点存在
            if parent in G.nodes:
                parent_layer = G.nodes[parent]['layer']
                for child in children:
                    G.add_node(child, layer=parent_layer + 1, size=200 // (layer - 1))

# ================= 关系构建 =================
edge_definitions = {
    'blue': [  # 超链接关系
        (central_api, "NtCreateFile"),
        (central_api, "NtOpenProcess"),
        ("NtCreateFile", "CreateFileW"),
        ("NtOpenProcess", "OpenProcess"),
        ("CreateFileW", "FindFirstFileW"),
        ("FindFirstFileW", "FindNextFileW")
    ],
    'green': [  # 相似度关系
        ("NtAllocateVirtualMemory", "VirtualAlloc"),
        ("NtWriteFile", "WriteFile"),
        ("VirtualAlloc", "HeapAlloc"),
        ("WriteProcessMemory", "ReadProcessMemory")
    ],
    'red': [  # 危险关系
        ("NtOpenProcess", "TerminateProcess"),
        ("TerminateProcess", "CreateRemoteThread"),
        ("CreateRemoteThread", "ResumeThread"),
        ("VirtualProtect", "WriteProcessMemory")
    ]
}

# 添加带属性的边
for color, edges in edge_definitions.items():
    for src, dst in edges:
        if src in G.nodes and dst in G.nodes:
            width = 2.0 if color == 'red' else 1.5
            G.add_edge(src, dst,
                       color=color,
                       width=width,
                       style='dashed' if color == 'green' else 'solid')

# ================= 属性校验 =================
# 确保所有节点都有layer属性
for node in G.nodes:
    if 'layer' not in G.nodes[node]:
        # 根据连接关系推断层级
        predecessors = list(G.predecessors(node))
        if predecessors:
            parent_layer = G.nodes[predecessors[0]]['layer']
            G.nodes[node]['layer'] = parent_layer + 1
        else:
            G.nodes[node]['layer'] = max(node_hierarchy.keys()) + 1

# 清理孤立节点（保留至少2度的节点）
for node in list(nx.isolates(G)):
    if G.degree(node) < 2:
        G.remove_node(node)

# ================= 可视化布局 =================
plt.figure(figsize=(20, 15))
pos = {}

# 计算分层布局
max_layer = max([G.nodes[n]['layer'] for n in G.nodes])
for node in G.nodes:
    layer = G.nodes[node]['layer']
    angle = random.uniform(0, 360)  # 随机起始角度
    radius = layer * 3  # 层间距离

    # 避免同一层节点重叠
    same_layer_nodes = [n for n in G.nodes if G.nodes[n]['layer'] == layer]
    angle_step = 360 / len(same_layer_nodes) if len(same_layer_nodes) > 0 else 0

    idx = same_layer_nodes.index(node)
    theta = np.radians(angle + idx * angle_step)
    pos[node] = (
        radius * np.cos(theta) + random.uniform(-0.5, 0.5),
        radius * np.sin(theta) + random.uniform(-0.5, 0.5)
    )

# ================= 图形绘制 =================
# 绘制节点
node_colors = [cm.plasma(G.nodes[n]['layer'] / max_layer) for n in G.nodes]
nx.draw_networkx_nodes(
    G, pos,
    node_size=[G.nodes[n]['size'] for n in G.nodes],
    node_color=node_colors,
    alpha=0.9,
    edgecolors='black',
    linewidths=0.5
)

# 绘制边
edge_colors = [G.edges[e]['color'] for e in G.edges]
edge_styles = [G.edges[e]['style'] for e in G.edges]
edge_widths = [G.edges[e]['width'] for e in G.edges]

for color, style, width in zip(edge_colors, edge_styles, edge_widths):
    edges = [e for e in G.edges if G.edges[e]['color'] == color]
    nx.draw_networkx_edges(
        G, pos,
        edgelist=edges,
        edge_color=color,
        style=style,
        width=width,
        arrows=True,
        arrowstyle='->',
        arrowsize=15
    )

# 绘制标签
text_effects = {
    'font_size': 9,
    'font_family': 'SimHei',  # 支持中文
    'font_weight': 'bold',
    'bbox': dict(
        boxstyle="round,pad=0.3",
        facecolor="white",
        edgecolor="gray",
        alpha=0.7
    )
}
nx.draw_networkx_labels(G, pos, **text_effects)

# 添加图例
legend_elements = [
    plt.Line2D([0], [0], color='blue', lw=2, label='调用关系'),
    plt.Line2D([0], [0], color='green', lw=2, linestyle='--', label='功能相似'),
    plt.Line2D([0], [0], color='red', lw=2, label='危险组合')
]
plt.legend(
    handles=legend_elements,
    loc='upper right',
    title="关系类型",
    title_fontsize=12,
    fontsize=10,
    frameon=True,
    framealpha=0.8
)

plt.title("Windows API 威胁分析知识图谱", fontsize=15, pad=18)
plt.axis('off')
plt.tight_layout()
# 在文件开头添加字体配置
plt.rcParams['font.sans-serif'] = ['SimHei']  # 简体中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 保存图像
plt.savefig('api_knowledge_graph.png', dpi=300, bbox_inches='tight')
plt.show()