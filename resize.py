from PIL import Image
import os
import glob

# 資料集目錄
dataset_dir = "/home/pc3424/DL_final/high-fidelity-generative-compression/data/Kodak/PhotoCD_PCD0992/test"

# 新資料夾目錄
output_dir = "/home/pc3424/DL_final/high-fidelity-generative-compression/data/Kodak/kodak_recop/test"
os.makedirs(output_dir, exist_ok=True)

# 要resize的目標大小
resize_size = (256, 256)

# 要crop的區域 (left, upper, right, lower)
target_width = 512
target_height = 512  


# 遍歷資料集目錄中的所有圖像
image_files = glob.glob(os.path.join(dataset_dir, "*.png"))
print(image_files)
for img_path in image_files:
    # 開啟圖像
    img = Image.open(img_path)
    # 執行resize操作
    # 獲取圖像的原始大小
    original_width, original_height = img.size
    # print(f'img size : {img.size}')
    left = (original_width - target_width) / 2
    top = (original_height - target_height) / 2
    right = (original_width + target_width) / 2
    bottom = (original_height + target_height) / 2
    # 執行crop操作
    img = img.crop((left, top, right, bottom))
    img = img.resize(resize_size, Image.ANTIALIAS)
    # # 提取圖像文件名
    img_name = os.path.basename(img_path)

    # # 儲存處理過的圖像到新資料夾
    output_path = os.path.join(output_dir, img_name)
    img.save(output_path)

print("complete recrop and save img into {output_dir}")
