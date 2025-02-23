import json
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pycocotools.mask import decode


# 读取并解码 RLE 遮罩
def decode_rle_mask(rle_mask, height, width):
    # 如果是列表形式，转换为字符串格式
    if isinstance(rle_mask, list):
        rle_mask = ' '.join(map(str, rle_mask))

    # 检查 RLE 数据格式
    print(f"Decoding RLE mask with length {len(rle_mask)}")
    print(f"RLE mask example: {rle_mask[:100]}")  # 打印部分 RLE 数据

    # 使用 pycocotools 解码 RLE
    rle = {
        'counts': rle_mask,
        'size': [height, width]
    }

    try:
        return decode(rle)
    except Exception as e:
        print(f"Error decoding RLE mask: {e}")
        return None


# 读取 JSON 并可视化遮罩
def visualize_rle_masks(json_folder, image_folder, output_folder):
    # 查找所有的 JSON 文件
    json_files = [f for f in os.listdir(json_folder) if f.endswith(".json")]

    for json_file in json_files:
        # 读取 JSON 文件
        json_path = os.path.join(json_folder, json_file)
        with open(json_path, 'r') as f:
            mask_data = json.load(f)

        # 获取图像文件名
        image_filename = mask_data["image"]  # 获取图像文件名
        image_path = os.path.join(image_folder, image_filename)
        image = Image.open(image_path)
        image_height, image_width = image.size

        # 遍历每个注释（annotation）并解码遮罩
        for annotation in mask_data["annotations"]:
            rle_mask = annotation["segmentation"]  # 获取 segmentation 数据
            mask = decode_rle_mask(rle_mask, image_height, image_width)

            if mask is not None:
                # 可视化原始图像与遮罩
                fig, ax = plt.subplots(1, 2, figsize=(12, 6))

                # 显示原始图像
                ax[0].imshow(image)
                ax[0].set_title('Original Image')
                ax[0].axis('off')

                # 显示遮罩图像
                ax[1].imshow(mask, cmap='jet', alpha=0.5)  # 使用透明度显示遮罩
                ax[1].set_title('Mask Visualization')
                ax[1].axis('off')

                # 保存可视化结果
                output_image_path = os.path.join(output_folder, f"visualization_{os.path.splitext(json_file)[0]}.png")
                plt.savefig(output_image_path)
                plt.close()

        print(f"Saved visualization for {json_file}.")


if __name__ == "__main__":
    # 更新文件夹路径
    json_folder = "exercise/result/coco_json"
    image_folder = "exercise"
    output_folder = "exercise/result/coco_json/result"

    # 调用可视化函数
    visualize_rle_masks(json_folder, image_folder, output_folder)
