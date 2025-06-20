// static/js/basic_processing_scripts.js
document.addEventListener('initBasicProcessing', function(event) {
    const currentTopLevelOperation = event.detail.currentOperation;
    const API_ENDPOINT_V2 = event.detail.apiEndpoint;

    console.log("[JS INIT] Basic Processing Init. Operation:", currentTopLevelOperation, "API:", API_ENDPOINT_V2);

    const imageUpload = document.getElementById('imageUpload');
    const previewImage = document.getElementById('previewImage');
    const mainProcessButton = document.getElementById('processButton');
    const outputImage = document.getElementById('outputImage');
    const outputPlaceholder = document.getElementById('outputPlaceholder');
    const saveButton = document.getElementById('saveButton');
    const operationControlsDiv = document.getElementById('operationControls');
    const errorMessageDiv = document.getElementById('errorMessage');

    let originalUploadedFile = null;
    let currentOutputBlobURL = null;

    let adjustmentParams = {
        brightness: 0, contrast: 0, saturation: 100,
        sharpen: 0, clarity: 0,
        wb_temperature: 0, wb_tint: 0,
        is_grayscale_mode: false,
        rotationAngle: 0, mirrorH: false, mirrorV: false
    };
    const defaultAdjustmentParams = JSON.parse(JSON.stringify(adjustmentParams));
    let activeSliders = {};

    function createSliderControl(id, label, min, max, step, defaultValue, paramKey, hints = null) {
        const itemDiv = document.createElement('div'); itemDiv.classList.add('control-item');
        const labelEl = document.createElement('label'); labelEl.setAttribute('for', id); labelEl.textContent = label + ": ";
        if (hints) { const hintL = document.createElement('span'); hintL.className = `color-hint color-hint-${hints.leftColor}`; hintL.textContent = hints.leftText; labelEl.appendChild(hintL); labelEl.append("↔"); const hintR = document.createElement('span'); hintR.className = `color-hint color-hint-${hints.rightColor}`; hintR.textContent = hints.rightText; labelEl.appendChild(hintR); }
        const slider = document.createElement('input'); slider.type = 'range'; slider.id = id; slider.min = min; slider.max = max; slider.step = step; slider.value = defaultValue;
        const outputEl = document.createElement('output'); outputEl.setAttribute('for', id); outputEl.textContent = defaultValue;
        slider.addEventListener('input', () => {
            outputEl.textContent = slider.value;
            if (paramKey) {
                adjustmentParams[paramKey] = parseFloat(slider.value);
                console.log(`[JS SLIDER CHANGE] Param: ${paramKey}, New Value: ${adjustmentParams[paramKey]}`);
            }
        });
        itemDiv.appendChild(labelEl); itemDiv.appendChild(slider); itemDiv.appendChild(outputEl);
        activeSliders[id] = slider;
        return itemDiv;
    }

    function createActionButton(id, text, actionType, toggleParamKey = null) {
        const button = document.createElement('button'); button.id = id; button.textContent = text;
        button.classList.add('action-button'); button.dataset.action = actionType;
        button.addEventListener('click', async () => {
            if (!originalUploadedFile) { showError('请先上传一张图片！'); return; }

            let oldRotation = adjustmentParams.rotationAngle;
            let oldMirrorH = adjustmentParams.mirrorH;
            let oldMirrorV = adjustmentParams.mirrorV;
            let oldGrayscale = adjustmentParams.is_grayscale_mode;

            if (toggleParamKey) {
                adjustmentParams[toggleParamKey] = !adjustmentParams[toggleParamKey];
                 if (toggleParamKey === 'is_grayscale_mode') {
                     button.textContent = adjustmentParams.is_grayscale_mode ? "彩色模式" : "灰度模式";
                     button.classList.toggle('active', adjustmentParams.is_grayscale_mode);
                 }
            } else {
                if (actionType === 'rotate_left') { adjustmentParams.rotationAngle = (adjustmentParams.rotationAngle - 90 + 360) % 360; }
                else if (actionType === 'rotate_right') { adjustmentParams.rotationAngle = (adjustmentParams.rotationAngle + 90) % 360; }
                else if (actionType === 'mirror_horizontal') { adjustmentParams.mirrorH = !adjustmentParams.mirrorH; }
                else if (actionType === 'mirror_vertical') { adjustmentParams.mirrorV = !adjustmentParams.mirrorV; }
            }
            console.log(`[JS ACTION BUTTON] Action: ${actionType}`);
            console.log(`  Old Params: rot=${oldRotation}, mH=${oldMirrorH}, mV=${oldMirrorV}, gray=${oldGrayscale}`);
            console.log(`  New Params: rot=${adjustmentParams.rotationAngle}, mH=${adjustmentParams.mirrorH}, mV=${adjustmentParams.mirrorV}, gray=${adjustmentParams.is_grayscale_mode}`);

            await processImageWithAllParams(actionType);
        });
        return button;
    }

    function setupOperationControls() {
        operationControlsDiv.innerHTML = ''; activeSliders = {};
        const groupDiv = document.createElement('div'); groupDiv.classList.add('controls-group');
        mainProcessButton.style.display = 'none';
        mainProcessButton.textContent = "应用参数调整";

        const nonSpecialPagesForGrayscaleToggle = ['brightness_contrast', 'saturation', 'sharpen', 'clarity', 'white_balance'];
        if (nonSpecialPagesForGrayscaleToggle.includes(currentTopLevelOperation)) {
             const gsButtonText = adjustmentParams.is_grayscale_mode ? "彩色模式" : "灰度模式";
             const gsButton = createActionButton('btnToggleGrayscale', gsButtonText, 'toggle_grayscale', 'is_grayscale_mode');
             if(adjustmentParams.is_grayscale_mode) gsButton.classList.add('active'); else gsButton.classList.remove('active');
             const topGroup = document.createElement('div'); topGroup.classList.add('controls-group');
             topGroup.appendChild(gsButton); operationControlsDiv.appendChild(topGroup);
        }

        if (currentTopLevelOperation === 'brightness_contrast') { mainProcessButton.style.display = 'block';
            groupDiv.appendChild(createSliderControl('brightnessSlider', '亮度', -100, 100, 1, adjustmentParams.brightness, 'brightness'));
            groupDiv.appendChild(createSliderControl('contrastSlider', '对比度', -100, 100, 1, adjustmentParams.contrast, 'contrast'));
        } else if (currentTopLevelOperation === 'grayscale') {
            mainProcessButton.style.display = 'block';
            adjustmentParams.is_grayscale_mode = true;
            const p = document.createElement('p'); p.textContent = "当前为灰度模式。可调整参数后应用。"; groupDiv.appendChild(p);
            groupDiv.appendChild(createSliderControl('brightnessSlider', '亮度 (灰度)', -100, 100, 1, adjustmentParams.brightness, 'brightness'));
            groupDiv.appendChild(createSliderControl('contrastSlider', '对比度 (灰度)', -100, 100, 1, adjustmentParams.contrast, 'contrast'));
        } else if (currentTopLevelOperation === 'saturation') { mainProcessButton.style.display = 'block';
            groupDiv.appendChild(createSliderControl('saturationSlider', '饱和度', 0, 200, 1, adjustmentParams.saturation, 'saturation'));
        } else if (currentTopLevelOperation === 'rotate_mirror') {
            groupDiv.appendChild(createActionButton('btnRotateLeft', '向左旋转90°', 'rotate_left'));
            groupDiv.appendChild(createActionButton('btnRotateRight', '向右旋转90°', 'rotate_right'));
            groupDiv.appendChild(createActionButton('btnMirrorHorizontal', '水平镜像', 'mirror_horizontal'));
            groupDiv.appendChild(createActionButton('btnMirrorVertical', '垂直镜像', 'mirror_vertical'));
        } else if (currentTopLevelOperation === 'sharpen') { mainProcessButton.style.display = 'block';
            groupDiv.appendChild(createSliderControl('sharpenSlider', '锐化强度', 0, 100, 1, adjustmentParams.sharpen, 'sharpen'));
        } else if (currentTopLevelOperation === 'clarity') { mainProcessButton.style.display = 'block';
            groupDiv.appendChild(createSliderControl('claritySlider', '清晰度量', 0, 200, 1, adjustmentParams.clarity, 'clarity'));
        } else if (currentTopLevelOperation === 'white_balance') { mainProcessButton.style.display = 'block';
            groupDiv.appendChild(createSliderControl('wbTempSlider', '色温', -100, 100, 1, adjustmentParams.wb_temperature, 'wb_temperature', {leftText: '冷', leftColor: 'blue', rightText: '暖', rightColor: 'yellow'}));
            groupDiv.appendChild(createSliderControl('wbTintSlider', '色调', -100, 100, 1, adjustmentParams.wb_tint, 'wb_tint', {leftText: '绿', leftColor: 'green', rightText: '品红', rightColor: 'magenta'}));
        } else if (currentTopLevelOperation === 'spatial_domain' || currentTopLevelOperation === 'frequency_domain') {
            const pText = currentTopLevelOperation === 'spatial_domain' ? "对上传图像执行预设空域操作。" : "对上传图像执行预设频域操作。";
            const btnText = currentTopLevelOperation === 'spatial_domain' ? "执行空域处理" : "执行频域处理";
            const p = document.createElement('p'); p.textContent = pText; groupDiv.appendChild(p);
            mainProcessButton.textContent = btnText; mainProcessButton.style.display = 'block';
        }
        if (groupDiv.hasChildNodes()) { operationControlsDiv.appendChild(groupDiv); }
    }

    imageUpload.addEventListener('change', async function(event) {
        const file = event.target.files[0];
        if (file) {
            originalUploadedFile = file;
            if (currentOutputBlobURL) { URL.revokeObjectURL(currentOutputBlobURL); currentOutputBlobURL = null; }
            adjustmentParams = JSON.parse(JSON.stringify(defaultAdjustmentParams));
            console.log("[JS DEBUG] Image Uploaded. Params Reset to:", JSON.parse(JSON.stringify(adjustmentParams)));
            setupOperationControls();
            const previewUrl = URL.createObjectURL(file);
            previewImage.src = previewUrl; previewImage.style.display = 'block';
            outputImage.style.display = 'none';
            outputPlaceholder.textContent = "处理后的结果将显示在这里";
            outputPlaceholder.style.display = 'block';
            saveButton.style.display = 'none';
            errorMessageDiv.style.display = 'none';
            await processImageWithAllParams(currentTopLevelOperation);
        }
    });

    setupOperationControls();
    mainProcessButton.addEventListener('click', () => processImageWithAllParams(currentTopLevelOperation));

    async function processImageWithAllParams(actionTrigger = null) {
        if (!originalUploadedFile) { showError('请先上传一张原始图片！'); return; }
        const formData = new FormData();
        formData.append('image', originalUploadedFile);
        formData.append('current_operation_category', actionTrigger || currentTopLevelOperation);

        // 添加所有 adjustmentParams 到 formData，并确保键名与后端匹配
        Object.keys(adjustmentParams).forEach(key => {
            let valueToAppend = adjustmentParams[key];
            let keyForForm = key;

            // 特殊处理键名以匹配后端期望 (snake_case)
            if (key === 'rotationAngle') { keyForForm = 'rotation_angle'; }
            else if (key === 'mirrorH') { keyForForm = 'mirror_h'; }
            else if (key === 'mirrorV') { keyForForm = 'mirror_v'; }
            else if (key === 'is_grayscale_mode') { keyForForm = 'is_grayscale_mode'; }
            else if (key === 'wb_temperature') { keyForForm = 'wb_temperature'; }
            else if (key === 'wb_tint') { keyForForm = 'wb_tint'; }
            // saturation, sharpen, clarity 的值转换
            else if (key === 'saturation' || key === 'clarity') { valueToAppend = parseFloat(adjustmentParams[key]) / 100.0; }
            else if (key === 'sharpen') { valueToAppend = parseFloat(adjustmentParams[key]) / 100.0;}

            // 确保布尔值作为字符串 "true" 或 "false" 发送
            if (typeof valueToAppend === 'boolean') {
                valueToAppend = valueToAppend.toString();
            }
            formData.append(keyForForm, valueToAppend);
        });

        console.log("[JS DEBUG] Sending FormData to API_ENDPOINT_V2:");
        for (let pair of formData.entries()) {
            console.log(`  ${pair[0]}: ${pair[1]}`);
        }

        showError(''); outputPlaceholder.textContent = '正在处理中...'; outputPlaceholder.style.display = 'block';
        outputImage.style.display = 'none';
        document.querySelectorAll('#operationControls button, #processButton').forEach(btn => btn.disabled = true);
        try {
            const response = await fetch(API_ENDPOINT_V2, { method: 'POST', body: formData });
            if (!response.ok) {
                let eMsg = `处理失败(${response.status})`;
                try{ const ed=await response.json(); eMsg = ed.error || eMsg; }catch(e){}
                throw new Error(eMsg);
            }
            const newImageBlob = await response.blob();
            if (currentOutputBlobURL) { URL.revokeObjectURL(currentOutputBlobURL); }
            currentOutputBlobURL = URL.createObjectURL(newImageBlob);
            outputImage.src = currentOutputBlobURL; outputImage.style.display = 'block';
            outputPlaceholder.style.display = 'none';
            const showSaveOps = ['brightness_contrast', 'grayscale', 'saturation', 'sharpen', 'clarity', 'white_balance', 'spatial_domain', 'frequency_domain'];
            if (showSaveOps.includes(actionTrigger || currentTopLevelOperation)) {
                saveButton.style.display = 'block';
            } else {
                saveButton.style.display = 'block';
            }
        } catch (error) {
            console.error('处理出错:', error); showError(error.message);
            outputPlaceholder.textContent = '处理失败';
        } finally {
            document.querySelectorAll('#operationControls button, #processButton').forEach(btn => btn.disabled = false);
        }
    }
    function showError(message) { errorMessageDiv.textContent = message; errorMessageDiv.style.display = message ? 'block' : 'none'; }
    saveButton.addEventListener('click', function() {
        if (currentOutputBlobURL) {
            const link = document.createElement('a'); link.href = currentOutputBlobURL;
            const timestamp = new Date().toISOString().replace(/[:.-]/g, '').slice(0, -4);
            const baseFileName = originalUploadedFile ? originalUploadedFile.name.split('.').slice(0, -1).join('.') : 'processed';
            link.download = `${baseFileName}_${currentTopLevelOperation}_${timestamp}.png`;
            document.body.appendChild(link); link.click(); document.body.removeChild(link);
        } else { showError('没有可保存的已处理图片。'); }
    });
});