"""
题目5参考答案：交通车辆检测与流量分析系统
综合应用 OpenCV、Numpy 和 Matplotlib 开发智能交通分析项目
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("交通车辆检测与流量分析系统".center(60))
print("Intelligent Traffic Detection and Analysis System".center(60))
print("=" * 70)

# ============ 第一阶段：图像采集与预处理 ============
print("\n【第一阶段：图像采集与预处理】")

# 读取交通场景图像
image = cv2.imread('traffic.jpg')

# 检查图像是否成功读取
if image is None:
    print("错误：无法读取图像 traffic.jpg")
    print("请确保在当前目录下放置 traffic.jpg 文件")
    print("建议使用包含道路和车辆的交通场景图像")
    exit(1)

print(f"✓ 交通图像读取成功")
print(f"  图像尺寸: {image.shape[1]} x {image.shape[0]} 像素")
print(f"  图像大小: {(image.shape[0] * image.shape[1]) / 1000000:.2f} 百万像素")

# 保存原始图像的副本用于标注
original_image = image.copy()

# 转换为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(f"✓ 已转换为灰度图")

# 直方图均衡化，增强对比度
equalized = cv2.equalizeHist(gray)
print(f"✓ 已应用直方图均衡化，提升图像对比度")

# 高斯模糊降噪
blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
print(f"✓ 已应用高斯模糊降噪 (5x5)")

# ============ 第二阶段：车辆检测与分割 ============
print("\n【第二阶段：车辆检测与分割】")

# 使用 Canny 边缘检测
edges = cv2.Canny(blurred, 50, 150)
print(f"✓ 已应用 Canny 边缘检测")

# 创建形态学操作的核
kernel_close = np.ones((7, 7), np.uint8)
kernel_open = np.ones((3, 3), np.uint8)

# 闭运算：连接车辆的断裂边缘
closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close, iterations=3)
print(f"✓ 已应用闭运算连接车辆边缘")

# 开运算：去除道路标记等干扰
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_open, iterations=1)
print(f"✓ 已应用开运算去除干扰")

# ============ 第三阶段：车辆识别与特征提取 ============
print("\n【第三阶段：车辆识别与特征提取】")

# 查找轮廓
contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"✓ 检测到 {len(contours)} 个候选区域")

# 过滤无效检测（基于面积和长宽比）
min_area = 800  # 最小面积阈值
max_area = 50000  # 最大面积阈值
min_aspect_ratio = 1.2  # 最小长宽比（车辆通常是长方形）
max_aspect_ratio = 5.0  # 最大长宽比

valid_vehicles = []
vehicle_features = []

for contour in contours:
    area = cv2.contourArea(contour)
    
    # 面积过滤
    if area < min_area or area > max_area:
        continue
    
    # 获取边界框
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = max(w, h) / min(w, h)
    
    # 长宽比过滤
    if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
        continue
    
    # 保存有效车辆
    valid_vehicles.append(contour)
    vehicle_features.append({
        'contour': contour,
        'area': area,
        'bbox': (x, y, w, h),
        'aspect_ratio': aspect_ratio,
        'center': (x + w//2, y + h//2)
    })

vehicle_count = len(valid_vehicles)
print(f"✓ 检测到 {vehicle_count} 辆车辆")
print(f"  过滤标准: 面积 [{min_area}, {max_area}], 长宽比 [{min_aspect_ratio:.1f}, {max_aspect_ratio:.1f}]")

# ============ 第四阶段：交通流量数据分析 ============
print("\n【第四阶段：交通流量数据分析】")

# 提取车辆面积数据
vehicle_areas = np.array([v['area'] for v in vehicle_features])

if len(vehicle_areas) > 0:
    # 计算统计信息
    area_min = np.min(vehicle_areas)
    area_max = np.max(vehicle_areas)
    area_mean = np.mean(vehicle_areas)
    area_std = np.std(vehicle_areas)
    
    print(f"车辆面积统计：")
    print(f"  最小面积: {area_min:.0f} 像素²")
    print(f"  最大面积: {area_max:.0f} 像素²")
    print(f"  平均面积: {area_mean:.0f} 像素²")
    print(f"  标准差:   {area_std:.0f} 像素²")
    
    # 车型分类（基于面积）
    threshold_small = area_mean - 0.5 * area_std
    threshold_large = area_mean + 0.5 * area_std
    
    small_vehicles = []  # 小型车
    medium_vehicles = []  # 中型车
    large_vehicles = []  # 大型车
    
    for i, feature in enumerate(vehicle_features):
        area = feature['area']
        if area < threshold_small:
            small_vehicles.append(i)
            feature['type'] = 'small'
        elif area > threshold_large:
            large_vehicles.append(i)
            feature['type'] = 'large'
        else:
            medium_vehicles.append(i)
            feature['type'] = 'medium'
    
    print(f"\n车型分类统计：")
    print(f"  小型车: {len(small_vehicles)} 辆 ({len(small_vehicles)/vehicle_count*100:.1f}%)")
    print(f"  中型车: {len(medium_vehicles)} 辆 ({len(medium_vehicles)/vehicle_count*100:.1f}%)")
    print(f"  大型车: {len(large_vehicles)} 辆 ({len(large_vehicles)/vehicle_count*100:.1f}%)")
    
    # 计算道路车辆密度
    image_area = image.shape[0] * image.shape[1]
    vehicle_coverage = np.sum(vehicle_areas) / image_area * 100
    density = vehicle_count / (image_area / 1000000)  # 车辆数/百万像素
    
    print(f"\n道路流量分析：")
    print(f"  车辆覆盖率: {vehicle_coverage:.2f}%")
    print(f"  车辆密度: {density:.2f} 辆/百万像素")
    
    # 拥堵程度评估
    if vehicle_coverage > 20:
        congestion_level = "严重拥堵"
    elif vehicle_coverage > 10:
        congestion_level = "中度拥堵"
    elif vehicle_coverage > 5:
        congestion_level = "轻度拥堵"
    else:
        congestion_level = "畅通"
    print(f"  拥堵程度: {congestion_level}")
    
    # 在原图上标注车辆
    result_image = original_image.copy()
    
    # 定义车型颜色
    colors = {
        'small': (0, 255, 0),    # 绿色 - 小型车
        'medium': (255, 165, 0),  # 橙色 - 中型车
        'large': (0, 0, 255)      # 红色 - 大型车
    }
    
    for i, feature in enumerate(vehicle_features, 1):
        x, y, w, h = feature['bbox']
        vehicle_type = feature['type']
        color = colors[vehicle_type]
        
        # 绘制边界框
        cv2.rectangle(result_image, (x, y), (x+w, y+h), color, 2)
        
        # 标注车辆编号
        cv2.putText(result_image, f"#{i}", (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # 添加图例和统计信息
    legend_y = 30
    cv2.putText(result_image, f'Total: {vehicle_count} vehicles', 
               (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(result_image, f'Congestion: {congestion_level}', 
               (10, legend_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    print(f"✓ 已在图像上标注所有车辆")
else:
    print("警告：未检测到有效车辆")
    result_image = original_image.copy()
    area_mean = 0
    area_std = 1
    small_vehicles = []
    medium_vehicles = []
    large_vehicles = []

# ============ 第五阶段：可视化报告生成 ============
print("\n【第五阶段：生成可视化分析报告】")

if vehicle_count > 0:
    # 创建 2x3 子图布局
    fig = plt.figure(figsize=(18, 12))
    
    # 子图1：原始交通图像
    ax1 = plt.subplot(2, 3, 1)
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    ax1.imshow(original_rgb)
    ax1.set_title('原始交通图像', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # 子图2：预处理后的图像
    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(blurred, cmap='gray')
    ax2.set_title('预处理 (均衡化+降噪)', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # 子图3：边缘检测结果
    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(opening, cmap='gray')
    ax3.set_title('边缘检测与形态学处理', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # 子图4：车辆检测标注图
    ax4 = plt.subplot(2, 3, 4)
    result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    ax4.imshow(result_rgb)
    ax4.set_title(f'车辆检测结果 (共 {vehicle_count} 辆)', fontsize=14, fontweight='bold')
    ax4.axis('off')
    
    # 添加颜色图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label=f'小型车 ({len(small_vehicles)}辆)'),
        Patch(facecolor='orange', label=f'中型车 ({len(medium_vehicles)}辆)'),
        Patch(facecolor='red', label=f'大型车 ({len(large_vehicles)}辆)')
    ]
    ax4.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # 子图5：车辆面积分布直方图
    ax5 = plt.subplot(2, 3, 5)
    
    # 为不同车型使用不同颜色
    bins = 20
    ax5.hist([vehicle_features[i]['area'] for i in small_vehicles], 
            bins=bins, alpha=0.7, label='小型车', color='green', edgecolor='black')
    ax5.hist([vehicle_features[i]['area'] for i in medium_vehicles], 
            bins=bins, alpha=0.7, label='中型车', color='orange', edgecolor='black')
    ax5.hist([vehicle_features[i]['area'] for i in large_vehicles], 
            bins=bins, alpha=0.7, label='大型车', color='red', edgecolor='black')
    
    ax5.set_title('车辆面积分布直方图', fontsize=14, fontweight='bold')
    ax5.set_xlabel('面积 (像素²)', fontsize=12)
    ax5.set_ylabel('车辆数量', fontsize=12)
    ax5.legend(loc='upper right', fontsize=10)
    ax5.grid(True, linestyle='--', alpha=0.5)
    
    # 添加统计信息
    stats_text = f'平均: {area_mean:.0f}\n标准差: {area_std:.0f}'
    ax5.text(0.02, 0.98, stats_text, transform=ax5.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 子图6：车型占比饼图
    ax6 = plt.subplot(2, 3, 6)
    
    sizes = [len(small_vehicles), len(medium_vehicles), len(large_vehicles)]
    labels = ['小型车', '中型车', '大型车']
    colors_pie = ['#90EE90', '#FFA500', '#FF6B6B']
    explode = (0.05, 0.05, 0.05)
    
    wedges, texts, autotexts = ax6.pie(sizes, explode=explode, labels=labels, 
                                         colors=colors_pie, autopct='%1.1f%%',
                                         shadow=True, startangle=90)
    
    # 美化文字
    for text in texts:
        text.set_fontsize(12)
        text.set_fontweight('bold')
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
    
    ax6.set_title('车型分布占比', fontsize=14, fontweight='bold')
    
    # 添加总标题
    fig.suptitle('交通车辆检测与流量分析系统 - 综合报告', 
                fontsize=18, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    
    print(f"✓ 已生成包含6个子图的完整分析报告")
else:
    print("警告：无法生成报告，未检测到车辆")
    fig = None

# ============ 第六阶段：结果输出与报告 ============
print("\n【第六阶段：保存结果与生成报告】")

# 保存标注后的检测结果图像
cv2.imwrite('traffic_detected.jpg', result_image)
print(f"✓ 车辆检测结果图像已保存为 traffic_detected.jpg")

# 保存完整的分析报告
if fig is not None:
    plt.savefig('traffic_analysis_report.png', dpi=200, bbox_inches='tight')
    print(f"✓ 完整分析报告已保存为 traffic_analysis_report.png")
    plt.close()

# ============ 生成文字版分析报告 ============
print("\n" + "=" * 70)
print("交通流量分析报告".center(60))
print("Traffic Flow Analysis Report".center(60))
print("=" * 70)

if vehicle_count > 0:
    print(f"\n【一、车辆检测统计】")
    print(f"  总检测车辆数: {vehicle_count} 辆")
    print(f"  小型车数量: {len(small_vehicles)} 辆 ({len(small_vehicles)/vehicle_count*100:.1f}%)")
    print(f"  中型车数量: {len(medium_vehicles)} 辆 ({len(medium_vehicles)/vehicle_count*100:.1f}%)")
    print(f"  大型车数量: {len(large_vehicles)} 辆 ({len(large_vehicles)/vehicle_count*100:.1f}%)")
    
    print(f"\n【二、车辆特征分析】")
    print(f"  车辆面积范围: {area_min:.0f} - {area_max:.0f} 像素²")
    print(f"  平均车辆面积: {area_mean:.0f} 像素²")
    print(f"  面积标准差: {area_std:.0f} 像素²")
    print(f"  面积变异系数: {(area_std/area_mean)*100:.1f}%")
    
    print(f"\n【三、道路流量评估】")
    print(f"  图像总面积: {image_area/1000000:.2f} 百万像素")
    print(f"  车辆覆盖率: {vehicle_coverage:.2f}%")
    print(f"  车辆密度: {density:.2f} 辆/百万像素")
    print(f"  拥堵程度评估: {congestion_level}")
    
    print(f"\n【四、流量管理建议】")
    if congestion_level == "严重拥堵":
        print(f"  ⚠️  道路处于严重拥堵状态，建议：")
        print(f"     - 启动交通疏导预案")
        print(f"     - 引导车辆分流至替代路线")
        print(f"     - 增加交警现场指挥")
    elif congestion_level == "中度拥堵":
        print(f"  ⚠️  道路处于中度拥堵状态，建议：")
        print(f"     - 实时监控交通流量变化")
        print(f"     - 适时调整信号灯配时")
        print(f"     - 发布交通提示信息")
    elif congestion_level == "轻度拥堵":
        print(f"  ℹ️  道路处于轻度拥堵状态，建议：")
        print(f"     - 保持正常监控")
        print(f"     - 关注高峰时段变化")
    else:
        print(f"  ✓  道路畅通，交通状况良好")
        print(f"     - 维持当前管理措施")
    
    print(f"\n【五、输出文件清单】")
    print(f"  1. traffic_detected.jpg       - 车辆检测标注图像")
    print(f"  2. traffic_analysis_report.png - 完整可视化分析报告")
else:
    print(f"\n【检测结果】")
    print(f"  未检测到有效车辆")
    print(f"\n【建议】")
    print(f"  1. 检查输入图像是否包含清晰的车辆")
    print(f"  2. 调整检测参数（面积阈值、长宽比等）")
    print(f"  3. 改善图像质量（光照、分辨率、拍摄角度）")

print("\n" + "=" * 70)
print("系统参数设置：".center(60))
print("=" * 70)
print(f"  最小车辆面积: {min_area} 像素²")
print(f"  最大车辆面积: {max_area} 像素²")
print(f"  长宽比范围: {min_aspect_ratio:.1f} - {max_aspect_ratio:.1f}")
print(f"  Canny边缘检测阈值: [50, 150]")
print(f"  形态学核大小: 闭运算 7x7, 开运算 3x3")
print("\n如需优化检测效果，可根据实际图像调整以上参数")
print("=" * 70)
