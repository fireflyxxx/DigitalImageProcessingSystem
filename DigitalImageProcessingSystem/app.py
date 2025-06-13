from flask import Flask, render_template, url_for, request, send_file, jsonify
from flask_cors import CORS
from PIL import Image
import cv2
import numpy as np
import io
import os
import pytesseract  # 用于OCR

app = Flask(__name__)
CORS(app)
# 定义空域操作模板图片的绝对路径
SPATIAL_TEMPLATE_PATH = os.path.join(app.root_path, 'static', 'spatial_template.png')

# --- Pytesseract 配置 ---
if os.name == 'nt':  # Windows系统
    # 你指定的Tesseract OCR路径
    tesseract_path_custom = r'E:\tesseract\tesseract.exe'
    if os.path.exists(tesseract_path_custom):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path_custom
        app.logger.info(f"Tesseract OCR路径已设置为: {tesseract_path_custom}")
    else:
        # 尝试其他常见路径作为备选
        tesseract_path_option2 = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        tesseract_path_option3 = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
        if os.path.exists(tesseract_path_option2):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path_option2
            app.logger.info(f"Tesseract OCR路径已设置为 (备选1): {tesseract_path_option2}")
        elif os.path.exists(tesseract_path_option3):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path_option3
            app.logger.info(f"Tesseract OCR路径已设置为 (备选2): {tesseract_path_option3}")
        else:
            app.logger.error(
                f"Tesseract OCR 未在指定路径 '{tesseract_path_custom}' 或常见默认路径找到。请确保Tesseract已安装并正确配置路径。OCR功能可能无法使用。")
else:  # Linux/MacOS
    # 在Linux/MacOS上，如果通过包管理器安装，tesseract通常会自动加入到系统PATH中。
    # 如果Pytesseract仍然找不到它，你可能需要取消下面这行的注释并指定路径。
    # default_linux_path = '/usr/bin/tesseract'
    # default_mac_path = '/usr/local/bin/tesseract' # Homebrew on Intel Mac
    # default_mac_arm_path = '/opt/homebrew/bin/tesseract' # Homebrew on Apple Silicon
    # if os.path.exists(default_linux_path) :
    #     pytesseract.pytesseract.tesseract_cmd = default_linux_path
    # elif os.path.exists(default_mac_path):
    #      pytesseract.pytesseract.tesseract_cmd = default_mac_path
    # elif os.path.exists(default_mac_arm_path):
    #      pytesseract.pytesseract.tesseract_cmd = default_mac_arm_path
    # else:
    #      app.logger.warn("Tesseract OCR 未在常见路径找到 (Linux/MacOS)。如果OCR功能失败，请确保tesseract在系统PATH中或在app.py中配置tesseract_cmd。")
    pass  # 假设在Linux/macOS下tesseract在PATH中，或者用户会按需配置


# --- 图像处理核心函数 ---
def adjust_brightness_contrast_cv(cv_img, brightness=0, contrast=0):
    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast)); alpha = f
    else:
        alpha = 1.0
    beta = int(brightness);
    adjusted_img = cv_img.astype(np.float32) * alpha + beta
    adjusted_img = np.clip(adjusted_img, 0, 255);
    new_img = adjusted_img.astype(np.uint8)
    return new_img


def apply_grayscale_cv(cv_img_bgr): return cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2GRAY)


def adjust_saturation_cv(cv_img_bgr, saturation_scale=1.0):
    if saturation_scale == 1.0: return cv_img_bgr
    if len(cv_img_bgr.shape) < 3 or cv_img_bgr.shape[2] == 1: return cv_img_bgr
    hsv_img = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2HSV);
    h, s, v = cv2.split(hsv_img)
    s_float = s.astype(np.float32);
    s_float *= saturation_scale
    s_float = np.clip(s_float, 0, 255);
    s_adjusted = s_float.astype(np.uint8)
    hsv_adjusted_img = cv2.merge([h, s_adjusted, v]);
    bgr_adjusted_img = cv2.cvtColor(hsv_adjusted_img, cv2.COLOR_HSV2BGR)
    return bgr_adjusted_img


def sharpen_image_cv(cv_img_bgr, amount=0.5):
    if amount == 0: return cv_img_bgr
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    fully_sharpened_img = cv2.filter2D(cv_img_bgr, -1, kernel)
    actual_blend_factor = np.clip(amount, 0.0, 1.0)
    sharpened_output = cv2.addWeighted(cv_img_bgr, 1 - actual_blend_factor, fully_sharpened_img, actual_blend_factor, 0)
    return sharpened_output


def enhance_clarity_cv(cv_img_bgr, amount=1.0):
    if amount == 0: return cv_img_bgr
    img_float = cv_img_bgr.astype(np.float32)
    if len(img_float.shape) == 2 or (len(img_float.shape) == 3 and img_float.shape[2] == 1):
        blurred_float = cv2.GaussianBlur(img_float, (0, 0), 3).astype(np.float32)
        if len(blurred_float.shape) == 2 and len(img_float.shape) == 3: blurred_float = blurred_float[:, :, np.newaxis]
    else:
        blurred_float = cv2.GaussianBlur(img_float, (0, 0), 3).astype(np.float32)
    details_float = img_float - blurred_float;
    enhanced_float = img_float + details_float * amount
    enhanced_float = np.clip(enhanced_float, 0, 255);
    enhanced_img = enhanced_float.astype(np.uint8)
    return enhanced_img


def manual_white_balance_cv(cv_img_bgr, temperature=0, tint=0):
    if len(cv_img_bgr.shape) < 3 or cv_img_bgr.shape[2] == 1: return cv_img_bgr
    if temperature == 0 and tint == 0: return cv_img_bgr
    img_float = cv_img_bgr.astype(np.float32);
    b, g, r = cv2.split(img_float)
    temp_factor = temperature / 100.0;
    tint_factor = tint / 100.0
    max_temp_influence = 0.2;
    max_tint_influence = 0.15
    if temp_factor > 0:
        r *= (1 + temp_factor * max_temp_influence); b *= (1 - temp_factor * max_temp_influence * 0.5)
    elif temp_factor < 0:
        b *= (1 + abs(temp_factor) * max_temp_influence); r *= (1 - abs(temp_factor) * max_temp_influence * 0.5)
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
    img_out = cv_img
    if rotation_angle == 90:
        img_out = cv2.rotate(img_out, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        img_out = cv2.rotate(img_out, cv2.ROTATE_180)
    elif rotation_angle == 270:
        img_out = cv2.rotate(img_out, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if mirror_h: img_out = cv2.flip(img_out, 1);
    if mirror_v: img_out = cv2.flip(img_out, 0)
    return img_out


def user_function1_spatial(cv_img_user_uploaded_bgr):
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
    def preprocess_for_stack(img, size):
        if img is None: return np.full((size[1], size[0], 3), (50, 50, 50), dtype=np.uint8)
        img_resized = cv2.resize(img, size)
        if len(img_resized.shape) == 2 or (len(img_resized.shape) == 3 and img_resized.shape[2] == 1):
            return cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
        elif img_resized.shape[2] == 4:
            return cv2.cvtColor(img_resized, cv2.COLOR_BGRA2BGR)
        return img_resized

    img1_p = preprocess_for_stack(img1, target_size_per_image);
    img2_p = preprocess_for_stack(img2, target_size_per_image)
    img3_p = preprocess_for_stack(img3, target_size_per_image);
    img4_p = preprocess_for_stack(img4, target_size_per_image)
    h_stack1 = np.hstack([img1_p, img2_p]);
    h_stack2 = np.hstack([img3_p, img4_p])
    return np.vstack([h_stack1, h_stack2])


def user_function2_frequency(cv_img_user_uploaded_bgr):
    gray_user = cv2.cvtColor(cv_img_user_uploaded_bgr, cv2.COLOR_BGR2GRAY)
    gray_user_resized = cv2.resize(gray_user, (400, 400))
    x, y, rect_width, rect_height = 150, 150, 100, 100
    mask_lowpass = np.zeros((400, 400, 2), np.float32);
    mask_lowpass[y:y + rect_height, x:x + rect_width] = 1
    mask_highpass = np.ones((400, 400, 2), np.float32);
    mask_highpass[y:y + rect_height, x:x + rect_width] = 0
    gray_f = np.float32(gray_user_resized);
    dft = cv2.dft(gray_f, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum_display = cv2.normalize(20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1),
                                               None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    fshift_low = dft_shift * mask_lowpass;
    f_ishift_low = np.fft.ifftshift(fshift_low)
    img_back_low = cv2.normalize(cv2.magnitude(cv2.idft(f_ishift_low)[:, :, 0], cv2.idft(f_ishift_low)[:, :, 1]), None,
                                 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    fshift_high = dft_shift * mask_highpass;
    f_ishift_high = np.fft.ifftshift(fshift_high)
    img_back_high = cv2.normalize(cv2.magnitude(cv2.idft(f_ishift_high)[:, :, 0], cv2.idft(f_ishift_high)[:, :, 1]),
                                  None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img_stack(gray_user_resized, magnitude_spectrum_display, img_back_low, img_back_high,
                     target_size_per_image=(200, 200))


def order_points_for_bill(pts):
    rect = np.zeros((4, 2), dtype="float32");
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)];
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)];
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform_for_bill(image, pts):
    rect = order_points_for_bill(pts);
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2));
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2));
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst);
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def resize_for_bill(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None;
    (h, w) = image.shape[:2]
    if width is None and height is None: return image
    if width is None:
        r = height / float(h); dim = (int(w * r), height)
    else:
        r = width / float(w); dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)


def process_bill_content(cv_img_uploaded):
    try:
        image = cv_img_uploaded
        if image is None or image.size == 0:
            return {"success": False, "text": "错误：传入图像为空。"}
        ratio = image.shape[0] / 500.0
        orig = image.copy()
        image_resized_for_detection = resize_for_bill(orig, height=500)
        gray = cv2.cvtColor(image_resized_for_detection, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray_blurred, 75, 200)
        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return {"success": False, "text": "未能检测到轮廓，请确保账单边缘清晰。"}
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        screenCnt = None
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                screenCnt = approx
                break
        if screenCnt is None:
            return {"success": False, "text": "未能找到账单的四边形轮廓。"}
        warped_doc = four_point_transform_for_bill(orig, screenCnt.reshape(4, 2) * ratio)
        warped_gray = cv2.cvtColor(warped_doc, cv2.COLOR_BGR2GRAY)
        ref_ocr = cv2.medianBlur(warped_gray, 3)
        try:
            text = pytesseract.image_to_string(Image.fromarray(ref_ocr), lang='eng')
        except Exception as ocr_error:
            app.logger.error(f"Pytesseract OCR 错误: {ocr_error}")
            return {"success": False, "text": f"OCR识别失败: {str(ocr_error)}。请检查Tesseract安装和语言包。"}
        if text and text.strip():
            return {"success": True, "text": text.strip()}
        else:
            return {"success": True, "text": "未能从图片中提取到有效文本。"}
    except Exception as e:
        app.logger.error(f"处理账单内容时发生错误: {e}", exc_info=True)
        return {"success": False, "text": f"处理账单时发生内部错误: {str(e)}"}


# --- API 端点 ---
@app.route('/api/process/image_v2', methods=['POST'])
def process_image_api_v2():
    if 'image' not in request.files: return jsonify({"error": "未提供原始图像文件"}), 400
    original_file = request.files['image']
    current_operation_category = request.form.get('current_operation_category', '')

    if original_file.filename == '': return jsonify({"error": "未选择原始文件"}), 400
    try:
        filestr = original_file.read();
        npimg = np.frombuffer(filestr, np.uint8)
        cv_img_original_decoded = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
        if cv_img_original_decoded is None or cv_img_original_decoded.size == 0:
            return jsonify({"error": "无法解码原始图像或图像为空"}), 400

        alpha_channel = None
        if cv_img_original_decoded.ndim == 3 and cv_img_original_decoded.shape[2] == 4:
            alpha_channel = cv_img_original_decoded[:, :, 3]
            cv_img_bgr_original = cv_img_original_decoded[:, :, :3]
        elif cv_img_original_decoded.ndim == 2:
            cv_img_bgr_original = cv2.cvtColor(cv_img_original_decoded, cv2.COLOR_GRAY2BGR)
        else:
            cv_img_bgr_original = cv_img_original_decoded

        if current_operation_category == 'bill_recognition':
            bill_result = process_bill_content(cv_img_bgr_original.copy())
            return jsonify(bill_result)  # 总是返回200 OK, 前端根据success字段判断
        elif current_operation_category == 'spatial_domain':
            final_output_image = user_function1_spatial(cv_img_bgr_original.copy())
        elif current_operation_category == 'frequency_domain':
            final_output_image = user_function2_frequency(cv_img_bgr_original.copy())
        else:
            is_grayscale_mode = request.form.get('is_grayscale_mode', 'false').lower() == 'true'
            brightness = float(request.form.get('brightness', 0))
            contrast = float(request.form.get('contrast', 0))
            saturation = float(request.form.get('saturation', 1.0))
            sharpen = float(request.form.get('sharpen', 0.0))
            clarity = float(request.form.get('clarity', 0.0))
            wb_temperature = float(request.form.get('wb_temperature', 0))
            wb_tint = float(request.form.get('wb_tint', 0))
            rotation_total_angle = int(request.form.get('rotation_angle', 0))
            mirror_horizontal_state = request.form.get('mirror_h', 'false').lower() == 'true'
            mirror_vertical_state = request.form.get('mirror_v', 'false').lower() == 'true'
            temp_img = cv_img_bgr_original.copy()
            if not is_grayscale_mode:
                temp_img = manual_white_balance_cv(temp_img, wb_temperature, wb_tint)
                temp_img = adjust_brightness_contrast_cv(temp_img, brightness, contrast)
                temp_img = adjust_saturation_cv(temp_img, saturation)
            else:
                temp_img = adjust_brightness_contrast_cv(temp_img, brightness, contrast)
            temp_img = sharpen_image_cv(temp_img, sharpen)
            temp_img = enhance_clarity_cv(temp_img, clarity)
            if is_grayscale_mode: temp_img = apply_grayscale_cv(temp_img)
            final_processed_main_part = apply_rotations_and_mirrors(temp_img, rotation_total_angle,
                                                                    mirror_horizontal_state, mirror_vertical_state)
            final_output_image = final_processed_main_part
            if alpha_channel is not None and not is_grayscale_mode and current_operation_category not in [
                'spatial_domain', 'frequency_domain', 'bill_recognition']:
                transformed_alpha = apply_rotations_and_mirrors(alpha_channel, rotation_total_angle,
                                                                mirror_horizontal_state, mirror_vertical_state)
                if final_processed_main_part.ndim == 3 and final_processed_main_part.shape[
                                                           :2] == transformed_alpha.shape[:2]:
                    final_output_image = cv2.merge((final_processed_main_part, transformed_alpha))
                else:
                    app.logger.warn("Alpha通道未重新附加")

        if final_output_image is None or final_output_image.size == 0: return jsonify(
            {"error": "图像处理后结果为空"}), 500
        is_success, buffer = cv2.imencode(".png", final_output_image)
        if not is_success: return jsonify({"error": "无法编码处理后的图像"}), 500
        img_io = io.BytesIO(buffer);
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"图像处理V2时出错: {e}", exc_info=True)
        return jsonify({"error": f"服务器错误V2: {str(e)}"}), 500


# --- 页面路由 ---
MENU_STRUCTURE = {
    "基础功能": {
        "grayscale": "灰度化处理", "brightness_contrast": "亮度与对比度",
        "saturation": "饱和度调整", "rotate_mirror": "旋转与镜像",
        "sharpen": "图像锐化调整", "clarity": "清晰度增加",
        "white_balance": "色温与色调", "spatial_domain": "空域图像操作",
        "frequency_domain": "频域图像操作",
    },
    "高级功能": {
        "animal_recognition": "动物识别 (占位)", "bill_recognition": "账单内容识别",
    }
}
ALL_OPTIONS_DISPLAY_MAP = {key: name for category in MENU_STRUCTURE.values() for key, name in category.items()}


@app.route('/')
def index(): return render_template('index.html', menu_data=MENU_STRUCTURE)


@app.route('/processing/<option_category_key>/<option_name_key>')
def processing_page(option_category_key, option_name_key):
    display_name = ALL_OPTIONS_DISPLAY_MAP.get(option_name_key, "未知处理")
    template_to_render = 'basic_processing.html'
    if option_category_key == "高级功能":
        if option_name_key == "bill_recognition":
            template_to_render = 'advanced/bill_recognition.html'
        elif option_name_key == "animal_recognition":
            template_to_render = 'advanced/animal_recognition.html'
        else:
            return "未知高级功能", 404
    return render_template(template_to_render, current_category=option_category_key,
                           option_name_internal=option_name_key, option_name_display=display_name)


if __name__ == '__main__': app.run(debug=True, port=5001)