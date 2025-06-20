#!/usr/bin/env python3
"""
检查图像尺寸分布脚本
用于分析训练和验证数据集中图像的尺寸情况
"""

import os
from PIL import Image
import numpy as np
from collections import defaultdict

def check_image_sizes(data_root):
    """检查指定目录下所有图像的尺寸"""
    sizes = []
    small_images = []
    
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        width, height = img.size
                        sizes.append((width, height))
                        
                        # 记录小尺寸图像
                        if width < 100 or height < 100:
                            small_images.append({
                                'path': file_path,
                                'size': (width, height),
                                'area': width * height
                            })
                except Exception as e:
                    print(f"无法读取图像 {file_path}: {e}")
    
    return sizes, small_images

def analyze_sizes(sizes):
    """分析尺寸分布"""
    if not sizes:
        return {}
    
    widths = [w for w, h in sizes]
    heights = [h for w, h in sizes]
    areas = [w * h for w, h in sizes]
    
    analysis = {
        'total_images': len(sizes),
        'width': {
            'min': min(widths),
            'max': max(widths),
            'mean': np.mean(widths),
            'median': np.median(widths),
            'std': np.std(widths)
        },
        'height': {
            'min': min(heights),
            'max': max(heights),
            'mean': np.mean(heights),
            'median': np.median(heights),
            'std': np.std(heights)
        },
        'area': {
            'min': min(areas),
            'max': max(areas),
            'mean': np.mean(areas),
            'median': np.median(areas),
            'std': np.std(areas)
        }
    }
    
    return analysis

def print_analysis(analysis, dataset_name):
    """打印分析结果"""
    print(f"\n{'='*60}")
    print(f"{dataset_name} 数据集图像尺寸分析")
    print(f"{'='*60}")
    
    if not analysis:
        print("没有找到图像文件")
        return
    
    print(f"总图像数量: {analysis['total_images']}")
    print(f"\n宽度统计:")
    print(f"  最小值: {analysis['width']['min']}")
    print(f"  最大值: {analysis['width']['max']}")
    print(f"  平均值: {analysis['width']['mean']:.1f}")
    print(f"  中位数: {analysis['width']['median']:.1f}")
    print(f"  标准差: {analysis['width']['std']:.1f}")
    
    print(f"\n高度统计:")
    print(f"  最小值: {analysis['height']['min']}")
    print(f"  最大值: {analysis['height']['max']}")
    print(f"  平均值: {analysis['height']['mean']:.1f}")
    print(f"  中位数: {analysis['height']['median']:.1f}")
    print(f"  标准差: {analysis['height']['std']:.1f}")
    
    print(f"\n面积统计:")
    print(f"  最小值: {analysis['area']['min']}")
    print(f"  最大值: {analysis['area']['max']}")
    print(f"  平均值: {analysis['area']['mean']:.1f}")
    print(f"  中位数: {analysis['area']['median']:.1f}")
    print(f"  标准差: {analysis['area']['std']:.1f}")

def print_small_images(small_images, dataset_name):
    """打印小尺寸图像信息"""
    if not small_images:
        print(f"\n{dataset_name} 中没有发现小尺寸图像 (< 100x100)")
        return
    
    print(f"\n{dataset_name} 中发现 {len(small_images)} 张小尺寸图像:")
    print("-" * 80)
    
    # 按面积排序
    small_images.sort(key=lambda x: x['area'])
    
    for i, img_info in enumerate(small_images[:10]):  # 只显示前10张
        rel_path = os.path.relpath(img_info['path'], '.')
        print(f"{i+1:2d}. {rel_path}")
        print(f"    尺寸: {img_info['size'][0]}x{img_info['size'][1]} (面积: {img_info['area']})")
    
    if len(small_images) > 10:
        print(f"    ... 还有 {len(small_images) - 10} 张小尺寸图像")

def main():
    """主函数"""
    print("图像尺寸分析工具")
    print("=" * 60)
    
    # 检查训练数据
    train_root = './data/classify/train'
    if os.path.exists(train_root):
        print(f"\n正在分析训练数据: {train_root}")
        train_sizes, train_small = check_image_sizes(train_root)
        train_analysis = analyze_sizes(train_sizes)
        print_analysis(train_analysis, "训练")
        print_small_images(train_small, "训练数据")
    else:
        print(f"训练数据目录不存在: {train_root}")
    
    # 检查验证数据
    val_root = './data/classify/val'
    if os.path.exists(val_root):
        print(f"\n正在分析验证数据: {val_root}")
        val_sizes, val_small = check_image_sizes(val_root)
        val_analysis = analyze_sizes(val_sizes)
        print_analysis(val_analysis, "验证")
        print_small_images(val_small, "验证数据")
    else:
        print(f"验证数据目录不存在: {val_root}")
    
    print(f"\n{'='*60}")
    print("分析完成")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 