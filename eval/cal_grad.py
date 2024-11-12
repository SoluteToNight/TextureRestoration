import os.path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from BuildingObj import BuildingObj
from img_class import TextureImage

def eval_by_grad(Building: BuildingObj):
    for processed_texture in Building.texture_list:
        origin_texture = cv2.imread(os.path.join(os.path.dirname(Building.obj_path), processed_texture.name))
        processed_texture_data = processed_texture.covert2nparray()
        ori_grad = calculate_gradients(origin_texture)
        print("Original Mean Gradient:", ori_grad)
        post_grad = calculate_gradients(processed_texture_data)
        print("Processed Mean Gradient:", post_grad)
        ori_edge = edge(origin_texture)
        post_edge = edge(processed_texture_data)
def calculate_gradients(image):
    if image is None:
        print("Error: Could not open or find the image.")
        exit()
    # 计算 x 方向的梯度
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # ksize=3 表示使用 3x3 的 Sobel 核
    # 计算 y 方向的梯度
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    # 计算梯度幅值
    magnitude = cv2.magnitude(grad_x, grad_y)
    # 将梯度幅值归一化到 0-255 范围内，以便显示
    magnitude_8u = np.uint8(magnitude / np.max(magnitude) * 255)
    mean_magnitude = np.mean(magnitude)
    # 显示结果
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.title('Gradient Magnitude')
    plt.imshow(magnitude_8u, cmap='gray')
    plt.subplot(2, 2, 3)
    plt.title('Gradient X')
    plt.imshow(np.uint8(grad_x / np.max(np.abs(grad_x)) * 255), cmap='gray')
    plt.subplot(2, 2, 4)
    plt.title('Gradient Y')
    plt.imshow(np.uint8(grad_y / np.max(np.abs(grad_y)) * 255), cmap='gray')
    plt.tight_layout()
    plt.show()
    return mean_magnitude
def edge(image):
    edges = cv2.Canny(image, 100, 200)

    # 显示 Canny 边缘检测结果
    plt.figure(figsize=(4, 4))
    plt.title('Canny Edges')
    plt.imshow(edges, cmap='gray')
    plt.show()
    return  edges