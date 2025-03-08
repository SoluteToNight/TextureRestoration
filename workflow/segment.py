import os.path
from os.path import split
from collections import Counter

from .node import Node
from modules.GroundedSam import load_sam2,load_dino,predict_mask

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
upper_threshold = 0.9
lower_threshold = 0.1
def mask_rate(mask):
    # mask = mask.astype(np.uint8)
    total_pixels = mask.shape[0] * mask.shape[1]
    ones = np.count_nonzero(mask)
    rate = ones / total_pixels
    print(f"rate is {rate}")
    return rate
def save_masks_as_files(masks:np.ndarray, labels, output_dir: str):
    if masks.dtype != np.uint8:

        mask_np = masks.astype(np.uint8) * 255
    else:
        mask_np = masks
    output_dir = output_dir.split('.')[0]
    output_dir += f'_{labels}' + '.png'
    # Save the mask as an image file
    mask_image = Image.fromarray(mask_np)
    # cv2.imwrite(output_dir, mask_np)
    mask_image.save(output_dir)
def auto_resize(timage,maxsize=2500000)->bool:
    img = Image.open(timage.img_path)
    total_pixels = img.size[0] * img.size[1]
    if total_pixels <= maxsize:
        return False
    elif total_pixels > maxsize:
        resize_scale = maxsize/total_pixels
        new_size = (int(img.size[0]*resize_scale), int(img.size[1] * resize_scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        tmp_path = timage.building_obj.temp_path
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)
        img.save(os.path.join(timage.building_obj.temp_path ,timage.name), "PNG")
        return True
def has_two_values_more_than_three(lst: list):
    count = Counter(lst)
    values_more_than_three = [value for value, cnt in count.items() if cnt > 2]
    if len(values_more_than_three) >=2:
        return min(values_more_than_three),max(values_more_than_three)
    else:
        return None
def infer_missing(masks):
    kernel = np.ones((3, 3), np.uint8)
    infered_missing_mask = (masks > 0).astype('uint8') * 255
    open = cv2.morphologyEx(infered_missing_mask, cv2.MORPH_OPEN, kernel)
    infered_missing_mask = np.logical_not(open)
    # save_masks_as_files(infered_missing_mask, "missing", mask_path)
    return infered_missing_mask
def analyse_mask(history,*arg):
    masks = None
    infered_missing =None
    # hist_before =None
    # print(history)
    thresholds = np.array([his['threshold'] for his in history if 'threshold' in his])
    rates = np.array([his['rate'] for his in history if 'rate' in his])
    probable_mask =[]
    probable_missing = []
    # 计算变化率（近似导数）
    change_rates = np.diff(rates) / np.diff(thresholds)
    # 设置阈值来区分剧烈变化和平滑变化
    threshold_change_rate = 0.1
    # 找出变化剧烈的区间
    volatile_intervals = []
    for i in range(len(change_rates)):
        if abs(change_rates[i]) > threshold_change_rate:
            volatile_intervals.append((thresholds[i], thresholds[i + 1]))
    # 找出平滑变化的区间
    smooth_intervals = []
    start_smooth = None
    for i in range(len(change_rates)):
        if abs(change_rates[i]) <= threshold_change_rate:
            if start_smooth is None:
                start_smooth = thresholds[i]
        else:
            if start_smooth is not None:
                smooth_intervals.append((start_smooth, thresholds[i]))
                start_smooth = None

    if start_smooth is not None:
        smooth_intervals.append((start_smooth, thresholds[-1]))
    print(f"volatile_intervals:{volatile_intervals}")
    print(f"smooth_intervals:{smooth_intervals}")
    # threshold增大，rate一定减小，若初始为0则排除区间
    for interval in smooth_intervals:
        start_iter = [i for i in range(len(history)) if 'threshold' in history[i] and history[i]['threshold'] == interval[0]][0]
        start_rate = history[start_iter]['rate']
        if start_rate<= 0.05:
            # 欠分割
            continue
        end_iter = [i for i in range(len(history)) if 'threshold' in history[i] and history[i]['threshold'] == interval[1]][0]
        end_rate = history[end_iter]['rate']
        middle_iter = int((start_iter + end_iter)/2)
        middle_rate = history[middle_iter]['rate']
        print(middle_rate)
        masks = history[middle_iter]['mask']
        if 0.85<=middle_rate < 0.99:
            # masks = history[middle_iter]['mask']
            probable_missing.append(infer_missing(masks))
        elif 0.05 < middle_rate < 0.85:
            probable_mask.append(masks)
    else:
        # rate_list = [mask_rate(mask) for mask in probable_mask if mask is not None]
        # aver_rate = np.mean(rate_list)
        if len(probable_mask) !=0:
            returned_masks = probable_mask[int(len(probable_mask)/2)]
        else:
            returned_masks = None
        if len(probable_missing) != 0:
            # infered_missing = probable_missing[int(len(probable_missing)/2)]
            infered_missing = probable_missing[0]

        else:
            infered_missing = None
    for interval in volatile_intervals:
        start_iter = [i for i in range(len(history)) if 'threshold' in history[i] and history[i]['threshold'] == interval[0]][0]
        start_rate = history[start_iter]['rate']
        end_iter = [i for i in range(len(history)) if 'threshold' in history[i] and history[i]['threshold'] == interval[1]][0]
        end_rate = history[end_iter]['rate']

        if (end_rate-start_rate) > 0.7:
            if arg is not None:
                # np.arange(interval[0]+0.01, interval[1], 0.01)
                predict_history = []
                for threshold in np.arange(interval[0]+0.01, interval[1], 0.01):
                    masks = predict_mask(arg[0],arg[1], arg[2], arg[3], threshold, threshold)
                    rate = mask_rate(masks)
                    predict_history.append({
                        'mask': masks,
                        'rate': rate,
                        'threshold': threshold
                    })
                    returned_masks, infered_missing = analyse_mask(predict_history)
            else:
                return history[start_iter]['mask'], infered_missing
    return returned_masks,infered_missing



class Segment(Node):
    def __init__(self, img_list):
        super().__init__(img_list)
        self.sam2_predictor = load_sam2()
        self.dino = load_dino()
    def process(self):

        for img in self.img_list:
            prompt_list = ["window","windows"]
            resized = auto_resize(img)
            if resized:
                img_path = os.path.join(img.building_obj.temp_path ,img.name)
            else:
                img_path = img.img_path
            for prompt in prompt_list:
                text_threshold,box_threshold = lower_threshold,lower_threshold
                iter_count = 0
                step = 0.05
                masks = np.zeros_like(img.img_data)
                rate = 0
                threshold_history = []
                predict_history = []
                mask_path = os.path.join(img.building_obj.temp_path ,img.name)

                if not os.path.exists(img.building_obj.temp_path):
                    os.makedirs(img.building_obj.temp_path)

                while lower_threshold <= box_threshold <= upper_threshold:
                    masks = predict_mask(prompt, self.sam2_predictor, self.dino, img_path, box_threshold, text_threshold)

                    rate = mask_rate(masks)
                    predict_history.append({
                        'mask': masks,
                        'rate': rate,
                        'threshold': box_threshold
                                            })
                    threshold_history.append(box_threshold)
                    print(f"threshold is {box_threshold}")
                    # if iter_count >= 3:
                    #     break
                    # iter_count += 1
                    box_threshold += step
                    text_threshold += step
                masks,missing = analyse_mask(predict_history,prompt, self.sam2_predictor, self.dino)

                if masks is None:
                    masks =  np.zeros_like(cv2.imread(img_path))
                if resized:
                    masks = (masks > 0).astype(np.uint8)*255
                    masks = cv2.resize(masks, img.img_data.size, interpolation=cv2.INTER_CUBIC)
                    masks = (masks>0).astype(bool)

                save_masks_as_files(masks, prompt, mask_path)
                if missing is not None:
                    if resized:
                        missing = (missing > 0).astype(np.uint8) * 255
                        missing = cv2.resize(missing, img.img_data.size, interpolation=cv2.INTER_CUBIC)
                        missing = (missing > 0).astype(bool)
                    save_masks_as_files(missing, "missing", mask_path)


                    # else:
                    #     break
                # if resized:
                #     masks = (masks > 0).astype(np.uint8)*255
                #     masks = cv2.resize(masks,img.img_data.size, interpolation=cv2.INTER_CUBIC)
                #     # masks = cv2.threshold(masks, 1, 255, cv2.THRESH_BINARY)[1]
                #     masks = (masks>0).astype(bool)
                # save_masks_as_files(masks, prompt, mask_path)
    # def process(self):
    #     for img in self.img_list:
    #         # prompt = "window. windows."
    #         prompt_list = ["window","windows"]
    #         resized = auto_resize(img)
    #         if resized:
    #             img_path = os.path.join(img.building_obj.temp_path ,img.name)
    #         else:
    #             img_path = img.img_path
    #         for prompt in prompt_list:
    #             box_threshold = 0.3
    #             text_threshold = 0.3
    #             iter_count = 0
    #             step = 0.05
    #             masks = np.zeros_like(img.img_data)
    #             # rate = 0
    #             threshold_history = []
    #             mask_path = os.path.join(img.building_obj.temp_path ,img.name)
    #             if not os.path.exists(img.building_obj.temp_path):
    #                 os.makedirs(img.building_obj.temp_path)
    #             while lower_threshold <= box_threshold <= upper_threshold:
    #                 masks = predict_mask(prompt, self.sam2_predictor, self.dino, img_path, box_threshold, text_threshold)
    #                 rate = mask_rate(masks)
    #                 threshold_history.append(box_threshold)
    #                 print(f"threshold is {box_threshold}")
    #                 if iter_count >= 3:
    #                     break
    #                 # iter_count += 1
    #                 if has_two_values_more_than_three(threshold_history) is not None:
    #                     # min_threshold, max_threshold = has_two_values_more_than_three(threshold_history)
    #                     if step !=0.01:
    #                         step = 0.01
    #                         threshold_history = []
    #                     elif step == 0.01:
    #                         if box_threshold in threshold_history:
    #                             iter_count += 1
    #
    #                     print(f"step is {step}")
    #                 if rate >= 0.95:
    #                     box_threshold += step
    #                     text_threshold += step
    #                     # masks = predict_mask(prompt, self.sam2_predictor, self.dino, img.img_path, box_threshold, text_threshold)
    #                     # rate = mask_rate(masks)
    #                 if rate < 0.05:
    #                     # box_threshold -= 0.05
    #                     # text_threshold -= 0.05
    #                     box_threshold -= step
    #                     text_threshold -= step
    #                     # masks = predict_mask(prompt, self.sam2_predictor, self.dino, img.img_path, box_threshold, text_threshold)
    #                     # rate = mask_rate(masks)
    #                 elif 0.85 < rate < 0.95:
    #                     # box_threshold += 0.05
    #                     # text_threshold += 0.05
    #                     box_threshold += step
    #                     text_threshold += step
    #                     kernel = np.ones((3, 3), np.uint8)
    #                     infered_missing_mask = (masks > 0).astype('uint8') * 255
    #                     open = cv2.morphologyEx(infered_missing_mask, cv2.MORPH_OPEN, kernel)
    #                     infered_missing_mask = np.logical_not(open)
    #                     save_masks_as_files(infered_missing_mask, "missing", mask_path)
    #
    #                 else:
    #                     break
    #             if resized:
    #                 masks = (masks > 0).astype(np.uint8)*255
    #                 masks = cv2.resize(masks,img.img_data.size, interpolation=cv2.INTER_CUBIC)
    #                 # masks = cv2.threshold(masks, 1, 255, cv2.THRESH_BINARY)[1]
    #                 masks = (masks>0).astype(bool)
    #             save_masks_as_files(masks, prompt, mask_path)
    # def process(self):
    #     for img in self.img_list:
    #         # window用于推测被遮挡区域
    #         prompt = "window. windows."
    #         prompt_list = prompt.split(" ")
    #         for prompts in prompt_list:
    #             masks = predict_mask(prompts, self.sam2_predictor, self.dino, img.img_path,0.3,0.3)
    #             mask_path = img.img_path.replace("obj","tmp")
    #             rate = mask_rate(masks)
    #             box_threshold = 0.3
    #             text_threshold = 0.3
    #             iter_count = 0
    #             while True:
    #                 if iter_count > 5:
    #                     break
    #                 iter_count += 1
    #                 if rate >0.9:
    #                     box_threshold += 0.05
    #                     text_threshold += 0.05
    #                     masks = predict_mask(prompts, self.sam2_predictor, self.dino, img.img_path, box_threshold, text_threshold)
    #                     rate = mask_rate(masks)
    #                 if rate <0.05:
    #                     box_threshold -=0.05
    #                     text_threshold -=0.05
    #                     masks = predict_mask(prompts, self.sam2_predictor, self.dino, img.img_path, box_threshold, text_threshold)
    #                     rate = mask_rate(masks)
    #                 else:
    #                     break
    #             save_masks_as_files(masks, prompts, mask_path)
