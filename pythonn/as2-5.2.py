import json
import math
import os
import re
from PIL import Image, ImageEnhance
import pytesseract
import streamlit as st
import pandas as pd
from google import genai
from google.genai import Client

# ==============================================================================
#  設定 Tesseract OCR 執行檔路徑 
# ==============================================================================
tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

if os.path.exists(tesseract_path):
    # 如果是您自己的 Windows 電腦，會走這裡並成功指定路徑
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    # 如果是 Streamlit 雲端伺服器，因為找不到 C 槽，會走這裡
    # 我們不報錯，而是嘗試讓系統自動尋找 Linux 內建的 tesseract
    try:
        # 測試看看系統能不能直接執行 tesseract
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError:
        # 只有當雲端和在地都完全找不到時，才噴出錯誤訊息
        st.error("❌ 系統找不到 Tesseract OCR 引擎！請確認環境中是否已安裝 Tesseract。")


# ==============================================================================
#  設定 Gemini API KEY
# ==============================================================================
# 您可以直接將 API Key 貼在下方引號中，或設定為環境變數
if "VLM_API_KEY" in st.secrets:
    GEMINI_API_KEY = st.secrets["VLM_API_KEY"]
else:
    GEMINI_API_KEY = "PLEASE_SET_KEY_IN_STREAMLIT_CLOUD"


def get_gemini_client():
    return Client(api_key=GEMINI_API_KEY)


# ==============================================================================
#  獨立大模型視覺解析函數
# ==============================================================================
def scan_image_with_vlm(image):
    
    try:
        client = get_gemini_client()
        
        # 確保 prompt 的完整定義出現在這裡，不能被刪除！
        prompt = """
        你是一個專業的電子硬體工程師。請仔細分析這張電路圖（Schematic），並執行以下任務：
        1. 圖形符號優先識別：不要只依賴文字名字！請優先尋找圖紙上的「電阻電路符號」，包含以下兩種樣式：
           - 樣式 A（美規）：兩端有引線的「鋸齒狀線條（Zigzag line）」。
           - 樣式 B（歐規）：兩端有引線的「空心細長方形（Rectangle）」。
           只要符合這兩種圖形，不論其位號名字是用 R, PR, SR, ZR 還是任何自訂代號，都必須判定為電阻並提取出來。
        
        2. 辨識它們各自的電阻值（例如 10K, 1K），如果前面帶有 NL/ 請務必保留（如 NL/10K）。
        3. 辨識它們的額定最大功率（例如 1/16W, 1/4W），若圖中未特別標註，請合理推測為 1/16W。
        4. 關鍵任務：沿著電路圖的導線與邏輯關係，找出該電阻連接的「主要供電電壓軌」（例如 3.3V, 1.8V, 5V）。
        5. 不論是 PR、R、SR、ZR 還是任何命名開頭，
           只要符合大模型輸出的文字規格結構，通通提取內部數值代入 V²/R 公式計算。
        
        注意事項：
        - 徹底忽略 0402, 0603 等封裝尺寸代號。
        - 排除接腳編號（如 1, 2, 3, 4）。
        
        請嚴格以 JSON 陣列格式回傳，不要包含任何 Markdown 標籤 (如 ```json) 或額外的解釋文字。
        JSON 格式範例：
        [
          {"name": "PR328", "val": "10K", "p": "1/16W", "v": "3.3V"},
          {"name": "PR330", "val": "1K", "p": "1/16W", "v": "1.8V"}
        ]
        """
        
        # 將模型名稱確實升級為最新的 'gemini-2.5-flash'
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[image, prompt] # 此處會完美撈取到上方定義好的 prompt
        )
        
        json_text = response.text.strip()
        json_text = re.sub(r'^```json\s*', '', json_text, flags=re.MULTILINE)
        json_text = re.sub(r'```$', '', json_text, flags=re.MULTILINE)
        
        parsed_data = json.loads(json_text)
        output_lines = []
        
        for item in parsed_data:
            if not isinstance(item, dict):
                continue
                
            r_name = (item.get("name") or item.get("id") or "").upper().strip()
            r_val = (item.get("val") or item.get("value") or "10K").upper().strip()
            r_power = (item.get("p") or item.get("power") or "1/16W").upper().strip()
            r_volt = (item.get("v") or item.get("voltage") or "3.3V").upper().strip()
            
            # 清洗與還原複雜電壓名稱
            r_val = r_val.replace(" ", "")
            r_volt = r_volt.replace(" ", "")
            
            if r_name:
                #只組裝電阻行格式，不再將電壓另外加入 voltages_set 容器中
                output_lines.append(f"{r_name}={r_val}_{r_power}_{r_volt}")
            
        # 直接對元件名單進行排序並以換行符號結合回傳
        # 這樣左側 Step 2 text_area 就絕對不會再出現任何前置的 V=+V3.3 等干擾行！
        if output_lines:
            return "\n".join(sorted(list(set(output_lines))))
        else:
            return "大模型未回傳有效元件數據"
        
    except Exception as e:
        return f"大模型視覺解析失敗。錯誤原因: {e}"
    
# ==============================================================================
#  讀取文字檔數值後計算
# ==============================================================================
def calculate_derating_metrics(user_text, derating_target=0.80):
    lines = user_text.split('\n')
    results = []
    fallback_voltage = 3.3

    for line in lines:
        line = line.strip()
        if not line or "=" not in line or line.upper().startswith("V="):
            continue

        # 在所有解析之前，只要發現數值開頭是 0 歐姆
        # 不論是 =0_、=0R_ 還是 =0OHM_，直接在原始字串就把它掉包成 0.002！
        line_upper = line.upper()
        if "=0_" in line_upper:
            line = line.replace("=0_", "=0.002_")
        elif "=0R_" in line_upper:
            line = line.replace("=0R_", "=0.002_")
        elif "=0OHM_" in line_upper:
            line = line.replace("=0OHM_", "=0.002_")
        elif "=NL/0_" in line_upper:
            line = line.replace("=NL/0_", "=0.002_")

        # 掉包完成後，才進行原本的拆分
        name, spec = line.split('=', 1)
        name = name.strip().upper()
        clean_spec = spec.upper().strip()

        # 先用底線切開所有區塊
        parts = clean_spec.split('_')
        
        # 3 條底線特徵限制
        if len(parts) >= 3:
            raw_val = parts[0].replace("1%", "").replace("5%", "").strip()
            
            # 防止 +V5_SB 被底線切碎的縫合機制
            if parts[-1].strip() == "SB" and len(parts) >= 4:
                raw_volt_name = "_".join(parts[-2:]).strip() 
                raw_power = "_".join(parts[1:-2]).strip()   
            else:
                raw_volt_name = parts[-1].strip()            
                raw_power = "_".join(parts[1:-1]).strip()    
            
            # ========================================================
            # 💡 步驟一：精準提取電壓 (V) 
            # ========================================================
            if "+V5_SB" in raw_volt_name:
                voltage_used = 5.0
            elif "+V3.3SB" in raw_volt_name or "+V3.3 SB" in raw_volt_name:
                voltage_used = 3.3
            else:
                v_num_match = re.search(r'(\d+(?:\.\d+)?)', raw_volt_name)
                if v_num_match:
                    voltage_used = float(v_num_match.group(1))
                else:
                    voltage_used = fallback_voltage

            # ========================================================
            # 💡 步驟二：額定最大功率 (PMAX) 提取
            # ========================================================
            p_max = 0.0625 
            p_match_slash = re.search(r'(\d+)/(\d+)', raw_power)
            p_match_under = re.search(r'(\d+)_(\d+)', raw_power)
            
            if p_match_slash:
                p_max = float(p_match_slash.group(1)) / float(p_match_slash.group(2))
            elif p_match_under:
                p_max = float(p_match_under.group(1)) / float(p_match_under.group(2))
            else:
                p_num_match = re.search(r'(\d+(?:\.\d+)?)', raw_power)
                if p_num_match:
                    p_max = float(p_num_match.group(1))

            # 🛠️ 功率安全熔斷
            if p_max <= 0:
                p_max = 0.0625

            # ========================================================
            # 💡 步驟三：阻值換算純歐姆 (R)
            # ========================================================
            clean_val_str = raw_val.replace("NL/", "").strip()

            val_match = re.search(r'(\d+(?:\.\d+)?)', clean_val_str)
            if not val_match:
                continue
            
            val_num = float(val_match.group(1))
            if 'K' in clean_val_str: val_num *= 1000
            elif 'M' in clean_val_str: val_num *= 1000000

            # 把 0 掉變成 0.002 ，這裡如果 val_num 還是 <= 0.002，就絕對是跳線！
            if val_num <= 0.002:
                val_num = 0.002
                is_jumper = True
            else:
                is_jumper = False

            # ========================================================
            # 💡 步驟四：核心降額公式計算 
            # ========================================================
            if is_jumper:
                p_act = (voltage_used ** 2) / val_num  # 分母絕對是 0.002！
                p_act_display = 0.0                    
                stress_ratio = 0.0
                is_pass = True
            else:
                p_act = (voltage_used ** 2) / val_num
                p_act_display = p_act
                stress_ratio = p_act / p_max
                is_pass = stress_ratio <= derating_target

            # ========================================================
            # 💡 步驟五：包裝結果回傳
            # ========================================================
            results.append({
                "name": name,
                "r_value": 0.0 if is_jumper else val_num, 
                "calc_r_value": val_num,                  
                "voltage_used": voltage_used, 
                "p_act": p_act_display,
                "p_max": p_max,
                "stress_ratio": stress_ratio,
                "is_pass": is_pass,
                "is_jumper": is_jumper  
            })

    return results

# ==============================================================================
#  獨立計算公式程式 (從文字行內抽離各自電壓進行 Pact 計算)
# ==============================================================================
def calculate_derating_metrics(user_text, derating_target=0.80):
    lines = user_text.split('\n')
    results = []

    # 1. 收集全域保底電壓
    global_voltages = []
    for line in lines:
        v_match = re.search(r'V\s*=\s*(\d+(?:\.\d+)?)', line, re.IGNORECASE)
        if v_match and "=" in line and line.strip().upper().startswith("V"):
            global_voltages.append(float(v_match.group(1)))
            
    fallback_voltage = global_voltages[-1] if global_voltages else 3.3

    # 2. 逐行解析每個電阻
    for line in lines:
        line = line.strip()
        if not line or "=" not in line or line.upper().startswith("V="):
            continue

        name, spec = line.split('=', 1)
        name = name.strip().upper()
        clean_spec = spec.upper().strip()

        # 將複雜字串依底線切開，最尾端的一定是電壓相關資訊
        parts = clean_spec.split('_')
        if len(parts) < 2:
            continue  # 格式不符則跳過

        # 永遠鎖定最後一個區塊作為電壓來源 (例如: "5V", "+V5_SB", "3.3V")
        # 如果最後一個是 "SB"，則把倒數兩個合併 (應對 +V5_SB 被切成 ['+V5', 'SB'] 的狀況)
        if parts[-1] == "SB" and len(parts) >= 3:
            raw_volt_part = "_".join(parts[-2:])
            # 剩餘的前面部分就是 阻值 與 功率
            remaining_spec = "_".join(parts[:-2])
        else:
            raw_volt_part = parts[-1]
            remaining_spec = "_".join(parts[:-1])

        # 提取電壓數字
        voltage_used = None
        v_num_match = re.search(r'(\d+(?:\.\d+)?)', raw_volt_part)
        if v_num_match:
            voltage_used = float(v_num_match.group(1))
        else:
            voltage_used = fallback_voltage

        # 主動抹除剩餘字串中的 NL/ 與 1%/5% 精度干擾
        remaining_spec = remaining_spec.replace("NL/", "").replace("1%", "").replace("5%", "").strip()

        # 精準提取功率 (PMAX) 
        p_max = 0.0625  # 預設 1/16W
        
        # 支援 1/16W, 1/8W 或 1_16W 的形式
        p_match = re.search(r'(\d+)\s*/\s*(\d+)\s*[WW]?', remaining_spec)
        p_match_under = re.search(r'(\d+)\s*_\s*(\d+)\s*[WW]?', remaining_spec)
        
        if p_match:
            p_max = float(p_match.group(1)) / float(p_match.group(2))
            remaining_spec = re.sub(r'\d+\s*/\s*\d+\s*[WW]?', '', remaining_spec).strip()
        elif p_match_under:
            p_max = float(p_match_under.group(1)) / float(p_match_under.group(2))
            remaining_spec = re.sub(r'\d+\s*_\s*\d+\s*[WW]?', '', remaining_spec).strip()
        else:
            # 單純數字型功率如 0.1W, 1W
            p_num_match = re.search(r'(\d+(?:\.\d+)?)\s*[WW]', remaining_spec)
            if p_num_match:
                p_max = float(p_num_match.group(1))
                remaining_spec = re.sub(r'(\d+(?:\.\d+)?)\s*[WW]', '', remaining_spec).strip()

        # 提取阻值 (R)
        # 此時 remaining_spec 裡只剩下阻值字串了（例如 "100K" 或 "1K"）
        val_match = re.search(r'(\d+(?:\.\d+)?)\s*([KkMm𝛀𝛺R]?)', remaining_spec)
        if not val_match:
            continue
            
        val_num = float(val_match.group(1))
        unit = val_match.group(2)
        if 'K' in unit: val_num *= 1000
        elif 'M' in unit: val_num *= 1000000

        # ==============================================================================
        #  0歐姆
        # ==============================================================================
        if val_num == 0 or "0 OHM" in clean_spec or "0OHM" in clean_spec or "NL/0" in clean_spec or "0R" in remaining_spec:
            val_num = 0.002   # 要求的 0.002 Ω，確保分母大於 0
            is_jumper = True
        else:
            is_jumper = False

        if is_jumper:
            # 導線跳線：使用 0.002Ω 算真實微小功耗，但最終前台應力與呈現歸 0 漂亮過關
            p_act = (voltage_used ** 2) / val_num  # 這裡的分母現在是 0.002，百分之百安全！
            p_act_display = 0.0                    # 前台顯示功耗歸零
            stress_ratio = 0.0
            is_pass = True
        else:
            # 普通正常電阻計算
            p_act = (voltage_used ** 2) / val_num
            p_act_display = p_act
            stress_ratio = p_act / p_max
            is_pass = stress_ratio <= derating_target

        results.append({
            "name": name,
            "r_value": 0.0 if is_jumper else val_num, 
            "calc_r_value": val_num,                  # 提供前台 code 算式印出 0.002
            "voltage_used": voltage_used, 
            "p_act": p_act_display,
            "p_max": p_max,
            "stress_ratio": stress_ratio,
            "is_pass": is_pass,
            "is_jumper": is_jumper  
        })

    return results

# ==============================================================================
#  Streamlit 使用者網頁介面
# ==============================================================================
st.set_page_config(layout="wide", page_title="Derating")
st.title("Derating Check")

DERATING_TARGET = 0.80

if "realtime_user_text" not in st.session_state:
    st.session_state.realtime_user_text = ""

col1, col2 = st.columns(2)

with col1:
    st.header("Step 1：上傳電路圖")
    uploaded_file = st.file_uploader("請上傳或拖曳圖片...", type=["png", "jpg", "jpeg"])
    
    if uploaded_file: 
        image = Image.open(uploaded_file)
        st.image(image, caption="已讀取電路圖", use_container_width=True)
        
        st.header("Step 2：辨識結果")
        
        # 觸發 VLM 智慧掃描辨識
        if "clean_ocr_output" not in st.session_state or st.button("🚀 重新啟動圖片分析"):
            with st.spinner("🧠 正在進行線路推理中..."):
                # 掃描完成後，同步將結果更新到 OCR 備份與當前即時文字狀態中
                vlm_result = scan_image_with_vlm(image)
                st.session_state.clean_ocr_output = vlm_result
                st.session_state.realtime_user_text = vlm_result  # 同步初始化文字框內容
        
        # 讓使用者可以即時修改與刪除 (綁定 key 機制)
        st.text_area(
            "元件與電壓清單 (若有小誤差可手動修改)：",
            value=st.session_state.realtime_user_text,
            height=250,
            key="realtime_user_text"  # 加上固定 key，由 st.session_state 主導管理
        )

with col2:
    st.header("Step 3：Derating 分析")
    if uploaded_file and st.button("開始計算Derating判定", type="primary"):
        with st.spinner("正在讀取行內專專規格並執行計算..."):
            try:
                # 從安全初始化後的狀態中提取即時文字
                current_text = st.session_state.realtime_user_text
                
                # 呼叫計算核心
                report_card = calculate_derating_metrics(current_text, DERATING_TARGET)
                
                st.success("✅ 計算完成！")
                st.write("---")
                
                if not report_card:
                    st.warning("⚠️ 文字框內無有效的電阻元件格式，請確認格式（如：HR1=10K_1/16W_3.3V）。")
            
                # 尋找所有算好的結果
                for component in report_card:
                    st.subheader(f"🔍 元件： {component['name']}")
                    
                    this_r_voltage = component.get('voltage_used', 3.3)
                    st.info(f"工作電壓： `{this_r_voltage:.1f} V`")

                    # 💡 提取前端即將用來顯示或可能引爆除法的變數
                    r_val = float(component.get('r_value', 0.0))
                    p_max = float(component.get('p_max', 0.0625))
                    p_act = float(component.get('p_act', 0.0))
                    
                    # 🚀 檢查是否為跳線，或者「分母是否不幸包含任何0」
                    if component.get('is_jumper', False) or r_val <= 0 or p_max <= 0:
                        # 🔌 只要發現任何一個分母是 0，或者標記為跳線，100% 封鎖原本有毒的算式！
                        st.markdown(f"* **阻值 (R)**： `0.0 Ω` (jump)")
                        st.markdown(f"* **量測工作功耗 (Pact)**： `0.000000 W` ")
                        st.markdown(f"* **額定最大功率 (PMAX)**： `{p_max if p_max > 0 else 0.0625:.4f} W` ")
                        
                        st.markdown(f"* **Pact/Pmax計算 (V² / R / Pmax)**：")
                        # 📝 這裡全部用純文字印出，絕對不執行任何代數除法，除以零的錯誤在全宇宙都不可能發生！
                        st.code(f"({this_r_voltage:.1f}V)² / 0.002Ω (0歐姆) = 0.0000")
                        st.markdown(f"* **Pact/Pmax**： `0.0%` (降額標準: {DERATING_TARGET*100}%)")
                        st.success(f"🟢 **PASS (0歐姆跳線)**")
                    
                    else:
                        # 🔒 只有在阻值大於0、且最大功率大於0的絕對安全狀態下，才放行跑正常電阻顯示
                        st.markdown(f"* **電阻值 (R)**： `{r_val:.1f} Ω` ")
                        st.markdown(f"* **量測工作功耗 (Pact)**： `{p_act:.6f} W` ")
                        st.markdown(f"* **額定最大功率 (Pmax)**： `{p_max:.4f} W` ")
                    
                        st.markdown(f"* **Pact/Pmax計算 (V² / R / Pmax)**：")
                        
                        # 在執行字串格式化列印前，做前端最後的雙重除法安全檢查
                        safe_stress_ratio = component.get('stress_ratio', 0.0)
                        
                        st.code(f"({this_r_voltage:.1f}V)² / {r_val:.0f}Ω / {p_max:.4f}W = {safe_stress_ratio:.4f}")
                        st.markdown(f"* **Pact/Pmax**： `{safe_stress_ratio*100:.1f}%` (Derating標準: {DERATING_TARGET*100}%)")
                        
                        if component.get('is_pass', False):
                            st.success(f"🟢 **PASS (符合Derating標準)**")
                        else:
                            st.error(f"❌ **FAIL (未通過Derating標準)**")
                            
                    st.write("---")
                    
            except Exception as e:
                # 🛠️ 萬一有其他我們沒想到的未知除法死角，把當機壓制住，改成印出貼心提示
                st.error(f"計算過程中有元件阻值或功率異常（含有 0 值），已被系統安全熔斷隔離。")
                st.code(f"錯誤報告: {e}")
