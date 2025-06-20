from flask import Flask, render_template, url_for, request, send_file, jsonify, make_response  # 确保make_response已导入
from flask_cors import CORS
from PIL import Image
import cv2
import numpy as np
import io
import os
import traceback
import datetime
from ultralytics import YOLO  # 用于动物识别

app = Flask(__name__)
CORS(app)

# --- 文件路径配置 ---
SPATIAL_TEMPLATE_PATH = os.path.join(app.root_path, 'static', 'spatial_template.png')
LICENSE_PLATE_TEMPLATE_PATH = os.path.join(app.root_path, 'static', 'tamplate_5.jpg')
ANIMAL_MODEL_PATH = os.path.join(app.root_path, 'models', 'best.pt')

# --- Tesseract OCR 路径配置 (Windows特定) ---
if os.name == 'nt':
    tesseract_path_custom = r'E:\tesseract\tesseract.exe'  # 用户指定的Tesseract路径
    # if os.path.exists(tesseract_path_custom): # 检查路径是否存在
    #     import pytesseract
    #     pytesseract.pytesseract.tesseract_cmd = tesseract_path_custom
    #     print(f"[INFO] Tesseract OCR路径已设置为: {tesseract_path_custom}")
    # else:
    #     print(f"[ERROR] Tesseract OCR 未在指定路径 '{tesseract_path_custom}' 找到。")
else:
    pass  # Linux/MacOS下通常Tesseract在系统PATH中

# --- 全局加载动物识别模型 ---
animal_detection_model = None  # 初始化模型变量
if os.path.exists(ANIMAL_MODEL_PATH):
    try:
        animal_detection_model = YOLO(ANIMAL_MODEL_PATH)  # 加载YOLOv8模型
        app.logger.info(f"动物识别模型 '{ANIMAL_MODEL_PATH}' 加载成功。")  # 使用Flask的logger记录成功信息
    except Exception as e:
        app.logger.error(f"加载动物识别模型失败: {e}", exc_info=True)  # 记录详细错误
else:
    app.logger.error(f"关键错误：动物识别模型文件未找到于: {ANIMAL_MODEL_PATH}。")

# 打印配置路径，方便启动时检查
print(f"[CONFIG DEBUG] 项目根目录: {app.root_path}")
print(f"[CONFIG DEBUG] 空域操作模板: {SPATIAL_TEMPLATE_PATH}")
print(f"[CONFIG DEBUG] 车牌识别模板: {LICENSE_PLATE_TEMPLATE_PATH}")
print(f"[CONFIG DEBUG] 动物识别模型: {ANIMAL_MODEL_PATH}")


# --- 图像处理核心函数 ---
def adjust_brightness_contrast_cv(cv_img, brightness=0, contrast=0):
    """调整图像亮度和对比度"""
    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))  # 对比度因子映射
        alpha = f
    else:
        alpha = 1.0
    beta = int(brightness)
    adjusted_img = cv_img.astype(np.float32) * alpha + beta
    adjusted_img = np.clip(adjusted_img, 0, 255)  # 像素值裁剪
    new_img = adjusted_img.astype(np.uint8)
    return new_img


def apply_grayscale_cv(cv_img_bgr):
    """转换为灰度图像"""
    return cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2GRAY)


def adjust_saturation_cv(cv_img_bgr, saturation_scale=1.0):
    """调整图像饱和度"""
    if saturation_scale == 1.0: return cv_img_bgr
    if len(cv_img_bgr.shape) < 3 or cv_img_bgr.shape[2] == 1: return cv_img_bgr  # 灰度图不处理

    hsv_img = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)
    s_float = s.astype(np.float32)
    s_float *= saturation_scale
    s_adjusted = np.clip(s_float, 0, 255).astype(np.uint8)
    hsv_adjusted_img = cv2.merge([h, s_adjusted, v])
    bgr_adjusted_img = cv2.cvtColor(hsv_adjusted_img, cv2.COLOR_HSV2BGR)
    return bgr_adjusted_img


def sharpen_image_cv(cv_img_bgr, amount=0.5):
    """图像锐化处理"""
    if amount == 0: return cv_img_bgr
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # 标准锐化核
    fully_sharpened_img = cv2.filter2D(cv_img_bgr, -1, kernel)
    actual_blend_factor = np.clip(amount, 0.0, 1.0)  # 混合原图与锐化图
    sharpened_output = cv2.addWeighted(cv_img_bgr, 1 - actual_blend_factor, fully_sharpened_img, actual_blend_factor, 0)
    return sharpened_output


def enhance_clarity_cv(cv_img_bgr, amount=1.0):
    """增加图像清晰度 (基于非锐化掩模)"""
    if amount == 0: return cv_img_bgr
    img_float = cv_img_bgr.astype(np.float32)
    # 创建模糊版本
    if len(img_float.shape) == 2 or (len(img_float.shape) == 3 and img_float.shape[2] == 1):
        blurred_float = cv2.GaussianBlur(img_float, (0, 0), 3).astype(np.float32)
        if len(blurred_float.shape) == 2 and len(img_float.shape) == 3: blurred_float = blurred_float[:, :, np.newaxis]
    else:
        blurred_float = cv2.GaussianBlur(img_float, (0, 0), 3).astype(np.float32)
    details_float = img_float - blurred_float  # 提取细节
    enhanced_float = img_float + details_float * amount  # 加回细节
    enhanced_img = np.clip(enhanced_float, 0, 255).astype(np.uint8)
    return enhanced_img


def manual_white_balance_cv(cv_img_bgr, temperature=0, tint=0):
    """手动调整色温和色调"""
    if len(cv_img_bgr.shape) < 3 or cv_img_bgr.shape[2] == 1: return cv_img_bgr
    if temperature == 0 and tint == 0: return cv_img_bgr
    img_float = cv_img_bgr.astype(np.float32);
    b, g, r = cv2.split(img_float)
    temp_factor = temperature / 100.0;
    tint_factor = tint / 100.0
    max_temp_influence = 0.2;
    max_tint_influence = 0.15  # 调整影响因子
    # 色温调整
    if temp_factor > 0:
        r *= (1 + temp_factor * max_temp_influence); b *= (1 - temp_factor * max_temp_influence * 0.5)
    elif temp_factor < 0:
        b *= (1 + abs(temp_factor) * max_temp_influence); r *= (1 - abs(temp_factor) * max_temp_influence * 0.5)
    # 色调调整
    if tint_factor > 0:
        r *= (1 + tint_factor * max_tint_influence * 0.5); b *= (1 + tint_factor * max_tint_influence * 0.5); g *= (
                    1 - tint_factor * max_tint_influence)
    elif tint_factor < 0:
        g *= (1 + abs(tint_factor) * max_tint_influence); r *= (1 - abs(tint_factor) * max_tint_influence * 0.5); b *= (
                    1 - abs(tint_factor) * max_tint_influence * 0.5)
    balanced_img_float = cv2.merge((b, g, r));
    balanced_img_float = np.clip(balanced_img_float, 0, 255)
    return balanced_img_float.astype(np.uint8)


def apply_rotations_and_mirrors(cv_img, rotation_angle=0, mirror_h=False, mirror_v=False):
    """应用旋转和镜像变换"""
    # print(f"[PY_APPLY_ROT_MIR] Input angle: {rotation_angle}, mH: {mirror_h}, mV: {mirror_v}, shape: {cv_img.shape if cv_img is not None else 'None'}")
    if cv_img is None: return None
    img_out = cv_img.copy()
    if rotation_angle == 90:
        img_out = cv2.rotate(img_out, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        img_out = cv2.rotate(img_out, cv2.ROTATE_180)
    elif rotation_angle == 270:
        img_out = cv2.rotate(img_out, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if mirror_h: img_out = cv2.flip(img_out, 1)
    if mirror_v: img_out = cv2.flip(img_out, 0)
    # print(f"[PY_APPLY_ROT_MIR] Output shape: {img_out.shape}")
    return img_out


def user_function1_spatial(cv_img_user_uploaded_bgr):
    """空域图像基本操作：与模板图像进行算术运算并拼接显示"""
    image_user = cv2.resize(cv_img_user_uploaded_bgr, (400, 400));
    gray_user = cv2.cvtColor(image_user, cv2.COLOR_BGR2GRAY)
    image2_template = cv2.imread(SPATIAL_TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
    if image2_template is None:
        app.logger.error(f"空域处理错误: 无法加载模板图像 {SPATIAL_TEMPLATE_PATH}")
        error_img = np.full((400, 800, 3), (200, 200, 200), dtype=np.uint8)
        cv2.putText(error_img, "Template Missing!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return error_img
    image2_template_resized = cv2.resize(image2_template, (400, 400))
    image_sum = cv2.add(gray_user, image2_template_resized);
    image_sub = cv2.subtract(gray_user, image2_template_resized)
    gray_user_f = gray_user.astype(np.float32);
    image2_template_f = image2_template_resized.astype(np.float32)
    image_mul_f = gray_user_f * image2_template_f / 255.0;
    image_mul = np.clip(image_mul_f, 0, 255).astype(np.uint8)
    epsilon = 1e-5;
    image2_template_f_safe = np.where(image2_template_f == 0, epsilon, image2_template_f)
    image_div_f = (gray_user_f / image2_template_f_safe) * 128;
    image_div = np.clip(image_div_f, 0, 255).astype(np.uint8)
    image_sum_bgr = cv2.cvtColor(image_sum, cv2.COLOR_GRAY2BGR);
    image_sub_bgr = cv2.cvtColor(image_sub, cv2.COLOR_GRAY2BGR)
    image_mul_bgr = cv2.cvtColor(image_mul, cv2.COLOR_GRAY2BGR);
    image_div_bgr = cv2.cvtColor(image_div, cv2.COLOR_GRAY2BGR)
    v_stack_1 = cv2.vconcat([image_sum_bgr, image_sub_bgr]);
    v_stack_2 = cv2.vconcat([image_mul_bgr, image_div_bgr])
    return cv2.hconcat([v_stack_1, v_stack_2])


def img_stack(img1, img2, img3, img4, target_size_per_image=(200, 200)):
    """辅助函数：将四张图像2x2拼接"""

    def preprocess_for_stack(img, size):
        if img is None: return np.full((size[1], size[0], 3), (50, 50, 50), dtype=np.uint8)  # 错误时返回深灰色占位图
        img_resized = cv2.resize(img, size)
        if len(img_resized.shape) == 2 or (len(img_resized.shape) == 3 and img_resized.shape[2] == 1):
            return cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
        elif img_resized.shape[2] == 4:
            return cv2.cvtColor(img_resized, cv2.COLOR_BGRA2BGR)  # 处理BGRA到BGR
        return img_resized

    img1_p = preprocess_for_stack(img1, target_size_per_image);
    img2_p = preprocess_for_stack(img2, target_size_per_image)
    img3_p = preprocess_for_stack(img3, target_size_per_image);
    img4_p = preprocess_for_stack(img4, target_size_per_image)
    h_stack1 = np.hstack([img1_p, img2_p]);
    h_stack2 = np.hstack([img3_p, img4_p])
    return np.vstack([h_stack1, h_stack2])


def user_function2_frequency(cv_img_user_uploaded_bgr):
    """频域图像基本操作：低通和高通滤波并拼接显示"""
    gray_user = cv2.cvtColor(cv_img_user_uploaded_bgr, cv2.COLOR_BGR2GRAY)
    gray_user_resized = cv2.resize(gray_user, (400, 400))
    x, y, rect_width, rect_height = 150, 150, 100, 100  # 固定掩码参数
    mask_lowpass = np.zeros((400, 400, 2), np.float32);
    mask_lowpass[y:y + rect_height, x:x + rect_width] = 1
    mask_highpass = np.ones((400, 400, 2), np.float32);
    mask_highpass[y:y + rect_height, x:x + rect_width] = 0
    gray_f = np.float32(gray_user_resized);
    dft = cv2.dft(gray_f, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)  # 中心化频谱
    magnitude_spectrum_display = cv2.normalize(20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1),
                                               None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # 低通滤波
    fshift_low = dft_shift * mask_lowpass;
    f_ishift_low = np.fft.ifftshift(fshift_low)
    img_back_low = cv2.normalize(cv2.magnitude(cv2.idft(f_ishift_low)[:, :, 0], cv2.idft(f_ishift_low)[:, :, 1]), None,
                                 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # 高通滤波
    fshift_high = dft_shift * mask_highpass;
    f_ishift_high = np.fft.ifftshift(fshift_high)
    img_back_high = cv2.normalize(cv2.magnitude(cv2.idft(f_ishift_high)[:, :, 0], cv2.idft(f_ishift_high)[:, :, 1]),
                                  None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img_stack(gray_user_resized, magnitude_spectrum_display, img_back_low, img_back_high,
                     target_size_per_image=(200, 200))


# --- 高级功能函数 ---
def detect_animals_yolo_implemented(cv_img_uploaded):
    """使用YOLO模型进行动物识别，并返回带标注的图像"""
    if animal_detection_model is None:
        print("[ANIMAL FUNC] 动物识别模型未加载。")
        return {"success": False, "error": "动物识别模型未成功加载，请检查服务器日志。", "image_passthrough": False}
    try:
        print(f"[ANIMAL FUNC] 输入图像 shape: {cv_img_uploaded.shape}")
        results = animal_detection_model.predict(source=cv_img_uploaded, save=False, conf=0.5)  # 执行预测
        if not results or not results[0].boxes:  # 检查是否有检测结果
            print("[ANIMAL FUNC] 未检测到任何物体。")
            return {"success": True, "message": "未检测到动物。", "image_passthrough": True, "detections": []}  # 返回原图和提示

        annotated_frame = results[0].plot()  # 获取带标注的图像 (YOLOv8方法)
        print(f"[ANIMAL FUNC] 带标注图像 shape: {annotated_frame.shape}")

        is_success_encode, buffer = cv2.imencode(".png", annotated_frame)  # 编码为PNG
        if not is_success_encode:
            print("[ANIMAL FUNC ERROR] 无法编码带标注的动物识别图像。")
            return {"success": False, "error": "无法编码处理后的动物识别图像。"}
        return {"success": True, "image_buffer": buffer}  # 返回图像的字节缓冲区
    except Exception as e:
        print(f"[ANIMAL FUNC CRITICAL] 动物识别时发生错误: {e}\n{traceback.format_exc()}")
        return {"success": False, "error": f"动物识别出错: {str(e)}"}


def recognize_license_plate_custom(cv_img_uploaded):
    """车牌识别函数，基于模板匹配（使用您确认能工作的版本）"""
    log_file_path = os.path.join(app.root_path, "lpr_debug_log.txt")
    with open(log_file_path, "w", encoding="utf-8") as f_init:
        f_init.write(f"--- [LPR LOG START AT {datetime.datetime.now()}] ---\n")

    def write_log(message):
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.datetime.now()}] {message}\n")

    write_log("--- [LPR START] recognize_license_plate_custom ---")
    if cv_img_uploaded is None or cv_img_uploaded.size == 0:
        write_log("[LPR ERROR] 输入图像为空!")
        return {"success": False, "plate_number": "错误：内部图像数据错误"}
    write_log(f"[LPR INFO] 收到图像 shape: {cv_img_uploaded.shape}, dtype: {cv_img_uploaded.dtype}")

    image = cv_img_uploaded.copy()

    # 辅助函数：轮廓排序 (定义在车牌识别函数内部，确保作用域)
    def sort_contours_for_plate(cnts_plate, method="left-to-right"):
        write_log(
            f"  [LPR_SORT] sort_contours_for_plate: 输入轮廓数量: {len(cnts_plate) if cnts_plate is not None else 'None'}")
        if not isinstance(cnts_plate, (list, tuple)) or not cnts_plate: write_log(
            "  [LPR_SORT] 输入轮廓为空或格式不正确。返回 [], []"); return [], []
        valid_cnts = [c for c in cnts_plate if isinstance(c, np.ndarray) and c.ndim >= 2 and c.shape[0] > 0]
        if not valid_cnts: write_log("  [LPR_SORT] 过滤后无有效轮廓。返回 [], []"); return [], []
        try:
            boundingBoxes = [];
            valid_cnts_for_zip = []
            for c_idx, c in enumerate(valid_cnts):
                rect = cv2.boundingRect(c)
                if rect is not None and len(rect) == 4:
                    boundingBoxes.append(rect); valid_cnts_for_zip.append(c)
                else:
                    write_log(f"  [LPR_SORT] 轮廓 {c_idx} 未返回有效边界框。")
            if not boundingBoxes or not valid_cnts_for_zip: write_log(
                "  [LPR_SORT] 无有效边界框用于排序。"); return valid_cnts_for_zip, boundingBoxes
            if len(valid_cnts_for_zip) != len(boundingBoxes): write_log(
                "  [LPR_SORT] 轮廓和边界框数量不匹配。"); return valid_cnts_for_zip, boundingBoxes
            zipped = list(zip(valid_cnts_for_zip, boundingBoxes));
            if not zipped: write_log("  [LPR_SORT] zip操作后结果为空。"); return [], []
            if method == "right-to-left":
                sorted_zipped = sorted(zipped, key=lambda b: b[1][0], reverse=True)
            elif method == "top-to-bottom":
                sorted_zipped = sorted(zipped, key=lambda b: b[1][1], reverse=False)
            elif method == "bottom-to-top":
                sorted_zipped = sorted(zipped, key=lambda b: b[1][1], reverse=True)
            else:
                sorted_zipped = sorted(zipped, key=lambda b: b[1][0], reverse=False)  # 默认从左到右
            if not sorted_zipped: write_log("  [LPR_SORT] 排序后结果为空。"); return [], []
            for item_idx, item in enumerate(sorted_zipped):
                if not (isinstance(item, tuple) and len(item) == 2): write_log(
                    f"  [LPR_SORT ERROR] sorted_zipped 元素 {item_idx} 格式不正确: {item}。"); return [], []
            s_cnts, s_boundingBoxes = zip(*sorted_zipped);
            write_log("  [LPR_SORT] 解包成功。");
            return list(s_cnts), list(s_boundingBoxes)
        except ValueError as ve:
            write_log(f"  [LPR_SORT CRITICAL] zip解包 ValueError: {ve}\n{traceback.format_exc()}"); return [], []
        except Exception as e_sort_outer:
            write_log(
                f"  [LPR_SORT CRITICAL] 轮廓排序外部错误: {e_sort_outer}\n{traceback.format_exc()}"); return valid_cnts_for_zip, boundingBoxes

    try:
        # 1. 模板处理
        write_log("[LPR INFO] 开始模板处理...");
        Numbers_template_orig = cv2.imread(LICENSE_PLATE_TEMPLATE_PATH)
        if Numbers_template_orig is None: write_log(
            f"[LPR ERROR] 无法加载车牌模板: {LICENSE_PLATE_TEMPLATE_PATH}"); return {"success": False,
                                                                                     "plate_number": "错误：无法加载车牌模板"}
        _Numbers_template_resized = cv2.resize(Numbers_template_orig, None, fx=0.36, fy=0.36,
                                               interpolation=cv2.INTER_AREA)
        template_gray = cv2.cvtColor(_Numbers_template_resized, cv2.COLOR_BGR2GRAY)
        ret_template_thresh, template_binary = cv2.threshold(template_gray, 127, 255, cv2.THRESH_BINARY_INV)
        write_log(f"[LPR INFO] 模板二值化 retval: {ret_template_thresh}")
        contours_template, _ = cv2.findContours(template_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours_template: write_log("[LPR ERROR] 车牌模板中未找到字符轮廓"); return {"success": False,
                                                                                             "plate_number": "错误：车牌模板中未找到字符轮廓"}
        sorted_template_contours_tuple = sort_contours_for_plate(contours_template)
        if not sorted_template_contours_tuple or not sorted_template_contours_tuple[0]:
            write_log("[LPR WARNING] 模板轮廓排序后为空或无效"); sorted_template_contours = []
        else:
            sorted_template_contours = sorted_template_contours_tuple[0]
        digits = {}
        for (i, c_template) in enumerate(sorted_template_contours):
            x, y, w, h = cv2.boundingRect(c_template);
            roi_color = None
            if w > 30:
                roi_color = _Numbers_template_resized[y:y + h,
                            max(0, x + w - 90): min(_Numbers_template_resized.shape[1], x + w)]
            else:
                roi_color = _Numbers_template_resized[y:y + h,
                            max(0, x - 45): min(_Numbers_template_resized.shape[1], x + 45)]
            if roi_color is not None and roi_color.size > 0:
                digits[i] = roi_color
            else:
                write_log(f"[LPR WARNING] 模板字符 {i} ROI为空。")
        if not digits: write_log("[LPR ERROR] 未能从模板中提取有效字符ROI"); return {"success": False,
                                                                                     "plate_number": "错误：未能从模板中提取有效字符ROI"}
        write_log(f"[LPR INFO] 模板字符ROI提取完毕，数量: {len(digits)}")

        # 2. 输入图像处理和车牌定位
        write_log("[LPR INFO] 开始输入图像处理和车牌定位...")
        input_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);
        input_blue_channel = image[:, :, 0]
        input_blue_eq = cv2.equalizeHist(input_blue_channel);
        input_gray_eq = cv2.equalizeHist(input_gray)
        reformed_plate_area = cv2.subtract(input_blue_eq, input_gray_eq)
        ret_plate_loc, binary_plate_loc = cv2.threshold(reformed_plate_area, 63, 255, cv2.THRESH_BINARY)
        kernel_plate = np.ones((5, 5), np.uint8);
        dilated_plate_loc = cv2.dilate(binary_plate_loc, kernel_plate, iterations=3)
        eroded_plate_loc = cv2.erode(dilated_plate_loc, kernel_plate, iterations=3)
        plate_candidate_contours, _ = cv2.findContours(eroded_plate_loc.copy(), cv2.RETR_EXTERNAL,
                                                       cv2.CHAIN_APPROX_SIMPLE)
        if not plate_candidate_contours: write_log("[LPR WARNING] 初步轮廓检测未找到车牌候选。"); return {
            "success": False, "plate_number": "未能定位到车牌区域候选"}
        cnt_plate_loc = plate_candidate_contours[0]  # 原始脚本直接取第一个
        x_plate, y_plate, w_plate, h_plate = cv2.boundingRect(cnt_plate_loc)
        y_start = max(0, y_plate + 4);
        y_end = min(image.shape[0], y_plate + h_plate - 4)
        x_start = max(0, x_plate + 4);
        x_end = min(image.shape[1], x_plate + w_plate - 4)
        License_image_extracted = image[y_start:y_end, x_start:x_end]
        if License_image_extracted.size == 0: write_log("[LPR ERROR] 车牌区域提取后为空"); return {"success": False,
                                                                                                   "plate_number": "车牌区域提取失败"}
        write_log(f"[LPR INFO] 车牌区域提取成功, 尺寸: {License_image_extracted.shape}")

        # 3. 车牌字符分割与识别
        lic_h, lic_w = License_image_extracted.shape[:2]
        if lic_h == 0: return {"success": False, "plate_number": "提取的车牌高度为0"}
        ratio_to_180 = 180.0 / lic_h
        License_std_h = cv2.resize(License_image_extracted, None, fx=ratio_to_180, fy=ratio_to_180,
                                   interpolation=cv2.INTER_AREA)
        write_log(f"[LPR INFO] 车牌缩放到标准高度180后尺寸: {License_std_h.shape}")
        gray_lic_std = cv2.cvtColor(License_std_h, cv2.COLOR_BGR2GRAY)
        gray_lic_std_eq = cv2.equalizeHist(gray_lic_std)
        ret_lic_chars_thresh, binary_lic_chars_for_contours = cv2.threshold(gray_lic_std_eq, 191, 255,
                                                                            cv2.THRESH_BINARY)
        char_contours, _ = cv2.findContours(binary_lic_chars_for_contours.copy(), cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
        if not char_contours: write_log("[LPR WARNING] 车牌字符分割未找到轮廓"); return {"success": False,
                                                                                         "plate_number": "车牌上未分割出字符"}
        sorted_char_contours_tuple = sort_contours_for_plate(char_contours)
        if not sorted_char_contours_tuple or not sorted_char_contours_tuple[0]:
            write_log("[LPR WARNING] 字符轮廓排序后为空或无效"); sorted_char_contours = []
        else:
            sorted_char_contours = sorted_char_contours_tuple[0]
        locs = []
        write_log("[LPR INFO] 开始筛选字符轮廓 (使用原始固定像素条件)...")
        for i_c, c_char in enumerate(sorted_char_contours):
            x_c, y_c, w_c, h_c = cv2.boundingRect(c_char)
            condition_log = f"h={h_c}. Condition: (140 < h < 145)"  # 使用原始固定条件
            if h_c > 140 and h_c < 145:
                locs.append((x_c, y_c, w_c, h_c));
                write_log(f"  [LPR Char Cand {i_c} PASSED]: {condition_log}")
            else:
                write_log(f"  [LPR Char Cand {i_c} FAILED]: {condition_log}")
        if not locs or len(locs) < 5: write_log(f"[LPR WARNING] 筛选出的有效字符区域数量不足: {len(locs)}"); return {
            "success": False, "plate_number": f"未能筛选出足够字符区域 ({len(locs)} found)"}
        write_log(f"[LPR INFO] 筛选出有效字符区域数量: {len(locs)}")
        output_chars_final = []
        if not digits or 0 not in digits or digits[0] is None or digits[0].size == 0: write_log(
            "[LPR ERROR] 模板字符 digits[0] 无效!"); return {"success": False, "plate_number": "错误: 模板数据准备失败"}
        template_match_size = (90, 172)
        for (gX, gY, gW, gH) in locs:
            char_roi_color = License_std_h[gY:gY + gH, gX:gX + gW]
            if char_roi_color.size == 0: continue
            char_roi_gray = cv2.cvtColor(char_roi_color, cv2.COLOR_BGR2GRAY)
            char_roi_gray_resized = cv2.resize(char_roi_gray, template_match_size)
            scores = []
            for (digit_idx_str, template_digit_roi_color) in digits.items():
                if template_digit_roi_color is None or template_digit_roi_color.size == 0: continue
                template_digit_gray = cv2.cvtColor(template_digit_roi_color, cv2.COLOR_BGR2GRAY)
                template_digit_gray_inv = 255 - template_digit_gray
                template_digit_final_for_match = cv2.resize(template_digit_gray_inv, template_match_size)
                result = cv2.matchTemplate(char_roi_gray_resized, template_digit_final_for_match, cv2.TM_CCOEFF_NORMED)
                _, score, _, _ = cv2.minMaxLoc(result);
                scores.append(score)
            if scores:
                best_match_score = max(scores)
                if best_match_score < 0.30: output_chars_final.append("?"); continue
                best_match_idx_in_scores = np.argmax(scores)
                original_template_index_str = list(digits.keys())[best_match_idx_in_scores]
                char_value_recognized = ""
                idx_int = int(original_template_index_str)
                char_map_example = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # 你校准的映射表
                if 0 <= idx_int < len(char_map_example):
                    char_value_recognized = char_map_example[idx_int]
                else:
                    char_value_recognized = f"?(i{idx_int})"
                output_chars_final.append(char_value_recognized)
            else:
                output_chars_final.append("?")
        plate_str_final = "".join(output_chars_final)
        write_log(f"[LPR INFO] 识别结果: {plate_str_final}")
        if not plate_str_final or plate_str_final.count("?") >= max(3, len(plate_str_final) / 2.0): return {
            "success": False, "plate_number": f"识别结果不可靠: {plate_str_final}"}
        return {"success": True, "plate_number": plate_str_final}
    except Exception as e_main_lpr:
        write_log(
            f"[LPR CRITICAL] recognize_license_plate_custom 主逻辑发生严重错误: {e_main_lpr}\n{traceback.format_exc()}")
        return {"success": False, "plate_number": "车牌识别过程中发生未知内部错误"}


# --- API 端点 ---
@app.route('/api/process/image_v2', methods=['POST'])
def process_image_api_v2():
    api_log_file_path = os.path.join(app.root_path, "api_call_log.txt")
    with open(api_log_file_path, "a", encoding="utf-8") as f_init_api:
        f_init_api.write(f"--- [/api/process/image_v2 START AT {datetime.datetime.now()}] ---\n")

    def write_api_log(message):
        with open(api_log_file_path, "a", encoding="utf-8") as f_api: f_api.write(
            f"[{datetime.datetime.now()}] {message}\n")

    write_api_log("API /api/process/image_v2: 收到请求")
    if 'image' not in request.files: write_api_log("[API ERROR] 请求中没有 'image' 文件"); return jsonify(
        {"error": "未提供原始图像文件"}), 400
    original_file = request.files['image'];
    current_operation_category = request.form.get('current_operation_category', '')
    write_api_log(f"[API INFO] 操作类别: {current_operation_category}")
    if original_file.filename == '': write_api_log("[API ERROR] 未选择原始文件"); return jsonify(
        {"error": "未选择原始文件"}), 400
    try:
        filestr = original_file.read();
        npimg = np.frombuffer(filestr, np.uint8)
        cv_img_original_decoded = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
        if cv_img_original_decoded is None or cv_img_original_decoded.size == 0: write_api_log(
            "[API ERROR] 无法解码原始图像或图像为空"); return jsonify({"error": "无法解码原始图像或图像为空"}), 400
        write_api_log(f"[API INFO] 原始图像解码成功, shape: {cv_img_original_decoded.shape}")
        alpha_channel = None
        if cv_img_original_decoded.ndim == 3 and cv_img_original_decoded.shape[2] == 4:
            alpha_channel = cv_img_original_decoded[:, :, 3];
            cv_img_bgr_original = cv_img_original_decoded[:, :, :3]
        elif cv_img_original_decoded.ndim == 2:
            cv_img_bgr_original = cv2.cvtColor(cv_img_original_decoded, cv2.COLOR_GRAY2BGR)
        else:
            cv_img_bgr_original = cv_img_original_decoded
        write_api_log(f"[API INFO] 准备调用处理函数，cv_img_bgr_original shape: {cv_img_bgr_original.shape}")
        final_output_image = None
        if current_operation_category == 'animal_recognition':
            animal_result = detect_animals_yolo_implemented(cv_img_bgr_original.copy());  # 调用实际函数
            write_api_log(f"[API INFO] detect_animals_yolo_implemented 返回: success={animal_result.get('success')}")
            if animal_result.get("success"):
                if "image_buffer" in animal_result:
                    img_io = io.BytesIO(animal_result["image_buffer"]); return send_file(img_io, mimetype='image/png')
                elif animal_result.get("image_passthrough"):
                    is_success_encode, buffer = cv2.imencode(".png", cv_img_original_decoded)
                    if not is_success_encode: write_api_log("[API ERROR] 动物识别：编码占位图失败"); return jsonify(
                        {"error": "无法编码占位图像"}), 500
                    img_io = io.BytesIO(buffer);
                    return send_file(img_io, mimetype='image/png')
                else:
                    write_api_log(f"[API WARNING] 动物识别成功但未返回图像: {animal_result}"); return jsonify(
                        animal_result)
            else:
                write_api_log(f"[API ERROR] 动物识别函数调用失败: {animal_result.get('error')}"); return jsonify(
                    {"error": animal_result.get("error", "动物识别失败")}), 500
        elif current_operation_category == 'license_plate_recognition':
            plate_result = recognize_license_plate_custom(cv_img_bgr_original.copy());
            write_api_log(f"[API INFO] recognize_license_plate_custom 返回: {plate_result}");
            return jsonify(plate_result)
        elif current_operation_category == 'spatial_domain':
            final_output_image = user_function1_spatial(cv_img_bgr_original.copy())
        elif current_operation_category == 'frequency_domain':
            final_output_image = user_function2_frequency(cv_img_bgr_original.copy())
        else:
            is_grayscale_mode = request.form.get('is_grayscale_mode', 'false').lower() == 'true';
            brightness = float(request.form.get('brightness', 0))
            contrast = float(request.form.get('contrast', 0));
            saturation = float(request.form.get('saturation', 1.0))
            sharpen = float(request.form.get('sharpen', 0.0));
            clarity = float(request.form.get('clarity', 0.0))
            wb_temperature = float(request.form.get('wb_temperature', 0));
            wb_tint = float(request.form.get('wb_tint', 0))
            rotation_total_angle = int(request.form.get('rotation_angle', 0));
            mirror_horizontal_state = request.form.get('mirror_h', 'false').lower() == 'true'
            mirror_vertical_state = request.form.get('mirror_v', 'false').lower() == 'true'
            write_api_log(
                f"[API INFO] 基础调整参数: gray={is_grayscale_mode}, bright={brightness}, contr={contrast}, sat={saturation}, sharp={sharpen}, clarity={clarity}, temp={wb_temperature}, tint={wb_tint}, rot={rotation_total_angle}, mirrorH={mirror_horizontal_state}, mirrorV={mirror_vertical_state}")
            temp_img = cv_img_bgr_original.copy()
            if not is_grayscale_mode:
                temp_img = manual_white_balance_cv(temp_img, wb_temperature, wb_tint);
                temp_img = adjust_brightness_contrast_cv(temp_img, brightness, contrast)
                temp_img = adjust_saturation_cv(temp_img, saturation)
            else:
                temp_img = adjust_brightness_contrast_cv(temp_img, brightness, contrast)
            temp_img = sharpen_image_cv(temp_img, sharpen);
            temp_img = enhance_clarity_cv(temp_img, clarity)
            if is_grayscale_mode: temp_img = apply_grayscale_cv(temp_img)
            final_processed_main_part = apply_rotations_and_mirrors(temp_img, rotation_total_angle,
                                                                    mirror_horizontal_state, mirror_vertical_state)
            final_output_image = final_processed_main_part
            if alpha_channel is not None and not is_grayscale_mode and current_operation_category not in [
                'spatial_domain', 'frequency_domain', 'animal_recognition', 'license_plate_recognition']:
                transformed_alpha = apply_rotations_and_mirrors(alpha_channel, rotation_total_angle,
                                                                mirror_horizontal_state, mirror_vertical_state)
                if final_processed_main_part.ndim == 3 and final_processed_main_part.shape[
                                                           :2] == transformed_alpha.shape[:2]:
                    final_output_image = cv2.merge((final_processed_main_part, transformed_alpha))
                else:
                    print("[WARNING] Alpha通道未重新附加")
        if final_output_image is not None:
            if final_output_image.size == 0: return jsonify({"error": "图像处理后结果为空"}), 500
            is_success, buffer = cv2.imencode(".png", final_output_image)
            if not is_success: return jsonify({"error": "无法编码处理后的图像"}), 500
            img_io = io.BytesIO(buffer);
            return send_file(img_io, mimetype='image/png')
        else:
            write_api_log("[API ERROR] final_output_image 为 None，但操作未返回JSON。"); return jsonify(
                {"error": "服务器内部逻辑错误"}), 500
    except Exception as e_api_outer:
        write_api_log(f"[API CRITICAL] /api/process/image_v2 发生严重错误: {e_api_outer}\n{traceback.format_exc()}")
        print(f"!!! API CRITICAL ERROR IN /api/process/image_v2 !!!\n{traceback.format_exc()}")
        return jsonify({"error": f"服务器错误V2 (外部捕获): {str(e_api_outer)}"}), 500


# --- 页面路由 ---
MENU_STRUCTURE = {
    "基础功能": {"grayscale": "灰度化处理", "brightness_contrast": "亮度与对比度", "saturation": "饱和度调整",
                 "rotate_mirror": "旋转与镜像", "sharpen": "图像锐化调整", "clarity": "清晰度增加",
                 "white_balance": "色温与色调", "spatial_domain": "空域图像操作", "frequency_domain": "频域图像操作", },
    "高级功能": {"animal_recognition": "动物识别", "license_plate_recognition": "车牌识别", }
}
ALL_OPTIONS_DISPLAY_MAP = {key: name for category in MENU_STRUCTURE.values() for key, name in category.items()}


@app.route('/')
def index(): return render_template('index.html', menu_data=MENU_STRUCTURE)


@app.route('/processing/<option_category_key>/<option_name_key>')
def processing_page(option_category_key, option_name_key):
    display_name = ALL_OPTIONS_DISPLAY_MAP.get(option_name_key, "未知处理")
    template_to_render = 'basic_processing.html'
    if option_category_key == "高级功能":
        if option_name_key == "animal_recognition":
            template_to_render = 'advanced/animal_recognition.html'
        elif option_name_key == "license_plate_recognition":
            template_to_render = 'advanced/license_plate_recognition.html'
        else:
            return "未知高级功能", 404
    return render_template(template_to_render, current_category=option_category_key,
                           option_name_internal=option_name_key, option_name_display=display_name)


if __name__ == '__main__': app.run(debug=True, port=5001)