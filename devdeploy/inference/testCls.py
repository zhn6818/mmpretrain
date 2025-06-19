import os
from devdeploy.inference.inference import Inference
from PIL import Image

# 模型路径和图片目录
CHECKPOINT_PATH = 'charCls/works/fullmodel_best.pth'
IMG_DIR = 'data/classify/val/25'

# 创建推理器实例
infer = Inference(CHECKPOINT_PATH, device='cpu', class_names=['20', '25', '50'])

def test_dir():
    """测试目录批量推理功能"""
    print("\n" + "="*80)
    print("目录批量推理测试")
    print("="*80)
    
    # 批量推理
    results = infer.infer_batch(IMG_DIR)

    # 打印结果
    for img_name, result in results.items():
        print(f'class_id: {result["class_id"]}  class_name: {result["class_name"]} 置信度: {result["confidence"]:.3f}      {img_name}')

# 单张图片推理测试
def test_single_image_inference():
    """测试单张图片推理功能"""
    
    # 获取测试图片列表
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    test_images = [
        f for f in os.listdir(IMG_DIR) 
        if f.lower().endswith(image_extensions)
    ]
    
    if not test_images:
        print(f"错误：在目录 {IMG_DIR} 中没有找到测试图片")
        return
    
    print(f"找到 {len(test_images)} 张测试图片")
    print("-" * 60)
    
    # 测试前3张图片的单张推理
    for i, img_name in enumerate(test_images):
        img_path = os.path.join(IMG_DIR, img_name)
        print(f"\n测试图片 {i+1}: {img_name}")
        print(f"图片路径: {img_path}")
        
        try:
            # 单张图片推理
            result = infer.infer_single(img_path, return_prob=True)
            
            # 打印详细结果
            print(f"预测类别ID: {result['class_id']}")
            print(f"预测类别名称: {result['class_name']}")
            print(f"置信度: {result['confidence']:.4f}")
            
            # 如果有概率分布，显示所有类别的概率
            if 'probabilities' in result:
                print("所有类别概率分布:")
                for class_id, prob in enumerate(result['probabilities']):
                    class_name = infer._get_class_name(class_id)
                    print(f"  {class_name}: {prob:.4f}")
            
            print("-" * 40)
            
        except Exception as e:
            print(f"推理失败: {str(e)}")
            print("-" * 40)

def test_top_k_predictions():
    """测试top-k预测功能"""
    
    # 获取第一张测试图片
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    test_images = [
        f for f in os.listdir(IMG_DIR) 
        if f.lower().endswith(image_extensions)
    ]
    
    if not test_images:
        print("没有找到测试图片，跳过top-k测试")
        return
    
    img_path = os.path.join(IMG_DIR, test_images[0])
    print(f"\n测试top-k预测功能")
    print(f"测试图片: {test_images[0]}")
    
    try:
        # 获取top-3预测结果
        top_results = infer.get_top_k_predictions(img_path, k=3)
        
        print("Top-3 预测结果:")
        for i, result in enumerate(top_results):
            print(f"  第{i+1}名: {result['class_name']} (ID: {result['class_id']}) - 置信度: {result['confidence']:.4f}")
            
    except Exception as e:
        print(f"Top-k预测失败: {str(e)}")

def test_model_info():
    """测试模型信息获取功能"""
    print(f"\n模型信息:")
    print("-" * 40)
    
    try:
        # 获取任务类型
        task_type = infer.get_task_type()
        print(f"任务类型: {task_type}")
        
        # 获取模型信息
        model_info = infer.get_model_info()
        print(f"模型信息: {model_info}")
        
        # 验证模型
        is_valid = infer.validate_model()
        print(f"模型验证: {'通过' if is_valid else '失败'}")
        
    except Exception as e:
        print(f"获取模型信息失败: {str(e)}")

def test_pil_image_input():
    """测试使用PIL Image对象作为输入的推理功能"""
    print("\n" + "="*80)
    print("PIL Image输入测试")
    print("="*80)
    
    # 获取所有测试图片
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    test_images = [
        f for f in os.listdir(IMG_DIR) 
        if f.lower().endswith(image_extensions)
    ]
    
    if not test_images:
        print(f"错误：在目录 {IMG_DIR} 中没有找到测试图片")
        return
    
    print(f"找到 {len(test_images)} 张测试图片")
    print("-" * 80)
    
    # 对每张图片进行测试
    for i, test_image in enumerate(test_images, 1):
        img_path = os.path.join(IMG_DIR, test_image)
        
        try:
            # 打开图片
            img = Image.open(img_path)
            
            # 使用PIL Image对象进行推理
            result = infer.infer_single(img, return_prob=True)
            
            # 打印结果
            print(f'class_id: {result["class_id"]}  class_name: {result["class_name"]} 置信度: {result["confidence"]:.3f}      {test_image}')
            
        except Exception as e:
            print(f"测试失败 {test_image}: {str(e)}")

# 执行测试
if __name__ == "__main__":
    # 测试目录批量推理
    # test_dir()
    
    # 测试单张图片推理
    # test_single_image_inference()
    
    # 测试PIL Image输入
    test_pil_image_input()
    
    # 测试模型信息
    test_model_info()
    
    # print("\n" + "="*80)
    # print("所有测试完成")
    # print("="*80)

