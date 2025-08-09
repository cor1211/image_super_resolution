import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from RDN_keras import RDN

# Tạo và load trọng số mô hình
model = RDN(scale_factor=4, num_channels=3, num_features=64, growth_rate=64, num_blocks=16,num_layers=8)
model.load_weights("rdn-C3-D10-G64-G064-x2_PSNR_epoch134.hdf5")  # thay đường dẫn đúng file của bạn
model.trainable = False  # Tắt chế độ huấn luyện

# Load ảnh gốc
image_path = r"D:\học\Tài Liệu CNTT\image_super_resolution\imagenet_dataset\barn\barn573_resized.jpg"
image = Image.open(image_path).convert("RGB")

# Resize ảnh bằng Bicubic để so sánh
bicubic_image = image.resize((256, 256), Image.BICUBIC)
plt.imshow(bicubic_image)
plt.title("Bicubic Interpolation")

# Tiền xử lý ảnh để đưa vào model
input_tensor = img_to_array(image) / 255.0  # Chuẩn hóa về [0,1]
input_tensor = np.expand_dims(input_tensor, axis=0)  # Thêm batch dimension

# Suy luận (inference)
output = model.predict(input_tensor)
output = np.clip(output[0], 0, 1)

# Hiển thị ảnh kết quả
output_image = array_to_img(output)
output_image.show(title="RDN Output")

plt.show()
