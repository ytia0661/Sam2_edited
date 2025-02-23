import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


# 初始化 SAM2 模型
def initialize_sam2_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

    return sam2, device


# 保存最终的 mask 图片，使用不同颜色标注不同的 mask
def save_combined_mask_image(masks, save_path, image_height, image_width):
    combined_mask = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    # 为每个 mask 分配不同的颜色
    for i, mask in enumerate(masks):
        color = np.random.randint(0, 256, size=3)  # 随机颜色
        mask_array = mask['segmentation']

        # 确保掩码与图像尺寸一致
        if mask_array.shape != (image_height, image_width):
            mask_array_resized = np.zeros((image_height, image_width), dtype=bool)
            # 根据需要调整大小（如果掩码小于图像，则通过插值扩展，否则通过裁剪）
            from skimage.transform import resize
            mask_array_resized = resize(mask_array, (image_height, image_width), preserve_range=True, order=0) > 0.5
            mask_array = mask_array_resized.astype(bool)

        combined_mask[mask_array] = color

    # 将 mask 图像保存为 PNG
    mask_image = Image.fromarray(combined_mask)
    mask_image.save(save_path)


# 计算区域面积
def calculate_area(segmentation):
    return np.sum(segmentation)


# 处理所有图像
def process_images(input_folder, output_folder):
    # 创建输出文件夹
    masks_folder = os.path.join(output_folder, "masks")
    os.makedirs(masks_folder, exist_ok=True)

    sam2, device = initialize_sam2_model()
    mask_generator = SAM2AutomaticMaskGenerator(sam2)

    # 获取所有图片文件
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".png", "jpeg"))]
    print(f"Found {len(image_files)} images to process.")

    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(input_folder, image_file)
        image = Image.open(image_path)
        image_width, image_height = image.size

        # 生成掩码
        masks = mask_generator.generate(np.array(image.convert("RGB")))

        # 保存最终的 mask 图片
        mask_output_path = os.path.join(masks_folder, f"{os.path.splitext(image_file)[0]}_mask.png")
        save_combined_mask_image(masks, mask_output_path, image_height, image_width)


if __name__ == "__main__":
    input_folder = "imagenette/train/n01440764"
    output_folder = "imagenette/result/01440764"
    process_images(input_folder, output_folder)


    # input_folder = "imagenette/train/n01440764"
    # output_folder = "imagenette/result/n01440764"
    # process_images(input_folder, output_folder)
    #
    # input_folder0 = "imagenette/train/n02102040"
    # output_folder0 = "imagenette/result/n02102040"
    # process_images(input_folder0, output_folder0)
    #
    # input_folder1 = "imagenette/train/n02979186"
    # output_folder1 = "imagenette/result/n02979186"
    # process_images(input_folder1, output_folder1)
    #
    # input_folder2 = "imagenette/train/n03000684"
    # output_folder2 = "imagenette/result/n03000684"
    # process_images(input_folder2, output_folder2)
    #
    # input_folder3 = "imagenette/train/n03028079"
    # output_folder3 = "imagenette/result/n03028079"
    # process_images(input_folder3, output_folder3)
    #
    # input_folder4 = "imagenette/train/n03394916"
    # output_folder4 = "imagenette/result/n03394916"
    # process_images(input_folder4, output_folder4)
    #
    # input_folder5 = "imagenette/train/n03417042"
    # output_folder5 = "imagenette/result/n03417042"
    # process_images(input_folder5, output_folder5)
    #
    # input_folder6 = "imagenette/train/n03425413"
    # output_folder6 = "imagenette/result/n03425413"
    # process_images(input_folder6, output_folder6)
    #
    # input_folder7 = "imagenette/train/n03445777"
    # output_folder7 = "imagenette/result/n03445777"
    # process_images(input_folder7, output_folder7)
    #
    # input_folder8 = "imagenette/train/n03888257"
    # output_folder8 = "imagenette/result/n03888257"
    # process_images(input_folder8, output_folder8)
