from PIL import Image
import os

def swap_and_adjust_aspect_ratio(mask_path, save_path):
    """
    调整掩码图片，将宽高调换并保存。

    :param mask_path: 原始掩码图片路径
    :param save_path: 调整后保存路径
    """
    # 打开掩码图片
    mask_image = Image.open(mask_path)

    # 获取当前宽高并调换
    current_width, current_height = mask_image.size
    target_width, target_height = current_height, current_width

    # 调整尺寸
    adjusted_mask = mask_image.resize((target_width, target_height), Image.NEAREST)

    # 保存调整后的掩码图片
    adjusted_mask.save(save_path)
    print(f"Adjusted mask saved to {save_path}")


def batch_swap_and_adjust_masks(input_folder, output_folder):
    """
    批量调整掩码图片，将宽高调换并保存。

    :param input_folder: 输入文件夹路径
    :param output_folder: 输出文件夹路径
    """
    os.makedirs(output_folder, exist_ok=True)
    mask_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".png"))]

    for mask_file in mask_files:
        mask_path = os.path.join(input_folder, mask_file)
        save_path = os.path.join(output_folder, mask_file)
        swap_and_adjust_aspect_ratio(mask_path, save_path)

# 示例调用

# output_folder = "imagenette/swap/n01440764"
# input_folder = "imagenette/result/n01440764/masks"
# batch_swap_and_adjust_masks(input_folder, output_folder)

output_folder0 = "imagenette/swap/n02102040"
input_folder0 = "imagenette/result/n02102040/masks"
batch_swap_and_adjust_masks(input_folder0, output_folder0)
#
# output_folder1 = "imagenette/swap/n02979186"
# input_folder1 = "imagenette/result/n02979186/masks"
# batch_swap_and_adjust_masks(input_folder1, output_folder1)
#
# output_folder2 = "imagenette/swap/n03000684"
# input_folder2 = "imagenette/result/n03000684/masks"
# batch_swap_and_adjust_masks(input_folder2, output_folder2)
#
# output_folder3 = "imagenette/swap/n03028079"
# input_folder3 = "imagenette/result/n03028079/masks"
# batch_swap_and_adjust_masks(input_folder3, output_folder3)
#
# output_folder4 = "imagenette/swap/n03394916"
# input_folder4 = "imagenette/result/n03394916/masks"
# batch_swap_and_adjust_masks(input_folder4, output_folder4)
#
# output_folder5 = "imagenette/swap/n03417042"
# input_folder5 = "imagenette/result/n03417042/masks"
# batch_swap_and_adjust_masks(input_folder5, output_folder5)
#
# output_folder6 = "imagenette/swap/n03425413"
# input_folder6 = "imagenette/result/n03425413/masks"
# batch_swap_and_adjust_masks(input_folder6, output_folder6)
#
# output_folder7 = "imagenette/swap/n03445777"
# input_folder7 = "imagenette/result/n03445777/masks"
# batch_swap_and_adjust_masks(input_folder7, output_folder7)
#
# output_folder8 = "imagenette/swap/n03888257"
# input_folder8 = "imagenette/result/n03888257/masks"
# batch_swap_and_adjust_masks(input_folder8, output_folder8)
