import os
import urllib.request
import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

# 自動檢查並下載 OpenCV 的人臉模型 XML 檔到專案資料夾中
xml_filename = "haarcascade_frontalface_default.xml"
if not os.path.exists(xml_filename):
  url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
  try:
    urllib.request.urlretrieve(url, xml_filename)
  except Exception as e:
    pass

st.set_page_config(
    page_title="數位影像處理與人臉分析工具", page_icon="🖼️", layout="wide"
)

st.title("🖼️ 互動式數位影像處理與人臉分析工具")
st.sidebar.header("控制面板")

# 1. 檔案上傳 (支援 PNG, JPG/JPEG, BMP)
uploaded_file = st.sidebar.file_uploader(
    "上傳影像檔案", type=["png", "jpg", "jpeg", "bmp"]
)

if uploaded_file is not None:
  # 計算檔案大小
  file_size_bytes = uploaded_file.size
  if file_size_bytes < 1024 * 1024:
    file_size_str = f"{file_size_bytes / 1024:.2f} KB"
  else:
    file_size_str = f"{file_size_bytes / (1024 * 1024):.2f} MB"

  # 讀取影像
  image = Image.open(uploaded_file)
  img_array = np.array(image)

  # 取得解析度與色彩維度
  height, width = img_array.shape[:2]
  if len(img_array.shape) == 3:
    channels = img_array.shape[2]
    color_mode = f"彩色 (RGB/RGBA, 通道數: {channels})"
  else:
    channels = 1
    color_mode = f"灰階 (Grayscale, 單通道)"

  # 2. 顯示中繼資訊
  st.sidebar.markdown("---")
  st.sidebar.subheader("📋 影像中繼資訊")
  st.sidebar.text(f"檔案格式: {image.format}")
  st.sidebar.text(f"影像解析度: {width} x {height} 像素")
  st.sidebar.text(f"色彩維度: {color_mode}")
  st.sidebar.text(f"檔案大小: {file_size_str}")

  # 3. 功能模組切換
  st.sidebar.markdown("---")
  module_choice = st.sidebar.radio(
      "選擇功能模組",
      [
          "色彩頻道與直方圖分析",
          "自訂卷積濾波與 PSNR 評估",
          "人臉偵測與信心度分析",
      ],
  )

  # 預先處理影像格式確保為 RGB
  base_img = img_array.copy()
  if len(base_img.shape) == 3 and base_img.shape[2] == 4:
    base_img = cv2.cvtColor(base_img, cv2.COLOR_RGBA2RGB)
  elif len(base_img.shape) == 2:
    base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2RGB)

  if module_choice == "色彩頻道與直方圖分析":
    view_mode = st.sidebar.selectbox(
        "選擇顯示檢視模式",
        [
            "原始影像 (Original)",
            "R 分量 (Red 頻道)",
            "G 分量 (Green 頻道)",
            "B 分量 (Blue 頻道)",
            "Y 分量 (亮度通道)",
            "U 分量 (色度通道 U)",
            "V 分量 (色度通道 V)",
        ],
    )

    col1, col2 = st.columns(2)
    r, g, b = cv2.split(base_img)
    yuv = cv2.cvtColor(base_img, cv2.COLOR_RGB2YUV)
    y, u, v = cv2.split(yuv)

    channel_to_plot = None
    hist_color = "gray"
    display_img = base_img.copy()

    with col1:
      st.subheader(f"檢視模式: {view_mode}")
      if view_mode == "原始影像 (Original)":
        channel_to_plot = cv2.cvtColor(base_img, cv2.COLOR_RGB2GRAY)
        hist_color = "gray"
      elif view_mode == "R 分量 (Red 頻道)":
        display_img = r
        channel_to_plot = r
        hist_color = "red"
      elif view_mode == "G 分量 (Green 頻道)":
        display_img = g
        channel_to_plot = g
        hist_color = "green"
      elif view_mode == "B 分量 (Blue 頻道)":
        display_img = b
        channel_to_plot = b
        hist_color = "blue"
      elif view_mode == "Y 分量 (亮度通道)":
        display_img = y
        channel_to_plot = y
        hist_color = "purple"
      elif view_mode == "U 分量 (色度通道 U)":
        display_img = u
        channel_to_plot = u
        hist_color = "orange"
      elif view_mode == "V 分量 (色度通道 V)":
        display_img = v
        channel_to_plot = v
        hist_color = "brown"

      st.image(
          display_img,
          use_column_width=True,
          clamp=True,
          channels="RGB" if len(display_img.shape) == 3 else "GRAY",
      )

    with col2:
      st.subheader("📊 強度分佈直方圖 (Histogram)")
      if channel_to_plot is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        hist = cv2.calcHist([channel_to_plot], [0], None, [256], [0, 256])
        color_map = {
            "gray": "dimgray",
            "red": "red",
            "green": "green",
            "blue": "blue",
            "purple": "purple",
            "orange": "darkorange",
            "brown": "saddlebrown",
        }
        p_color = color_map.get(hist_color, "black")
        ax.plot(hist, color=p_color, linewidth=1.5)
        ax.fill_between(range(256), hist.ravel(), color=p_color, alpha=0.3)
        ax.set_xlim([0, 255])
        ax.set_title(f"{view_mode} 像素強度分佈", fontsize=11)
        ax.set_xlabel("像素強度 (Pixel Intensity: 0–255)")
        ax.set_ylabel("像素數量 (Pixel Count)")
        ax.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig)

  elif module_choice == "自訂卷積濾波與 PSNR 評估":
    st.sidebar.markdown("### 🎛️ 卷積濾波參數設定")
    kernel_preset = st.sidebar.selectbox(
        "選擇濾波範本 (或手動自訂)",
        [
            "自訂輸入 (Custom)",
            "邊緣銳化 (Sharpen)",
            "拉普拉斯邊緣檢測 (Edge Detection)",
            "高斯平滑 (Gaussian Smoothing)",
            "浮雕效果 (Emboss)",
        ],
    )

    if kernel_preset == "邊緣銳化 (Sharpen)":
      default_kernel_str = "0, -1, 0\n-1, 5, -1\n0, -1, 0"
    elif kernel_preset == "拉普拉斯邊緣檢測 (Edge Detection)":
      default_kernel_str = "0, 1, 0\n1, -4, 1\n0, 1, 0"
    elif kernel_preset == "高斯平滑 (Gaussian Smoothing)":
      default_kernel_str = "1, 2, 1\n2, 4, 2\n1, 2, 1"
    elif kernel_preset == "浮雕效果 (Emboss)":
      default_kernel_str = "-2, -1, 0\n-1, 1, 1\n0, 1, 2"
    else:
      default_kernel_str = "0, 0, 0\n0, 1, 0\n0, 0, 0"

    kernel_input_str = st.sidebar.text_area(
        "輸入二維卷積核 (以逗號分隔欄，換行分隔列)",
        value=default_kernel_str,
        height=100,
    )
    do_normalize = st.sidebar.checkbox("啟用正規化 (Normalization)", value=False)

    try:
      rows = kernel_input_str.strip().split("\n")
      kernel_list = [
          [float(val.strip()) for val in row.split(",")] for row in rows
      ]
      kernel = np.array(kernel_list, dtype=np.float32)
      if len(set([len(r) for r in kernel_list])) > 1:
        kernel = np.array([[1]], dtype=np.float32)
    except:
      kernel = np.array([[1]], dtype=np.float32)

    if do_normalize:
      k_sum = np.sum(kernel)
      if k_sum != 0:
        kernel = kernel / k_sum

    filtered_img = cv2.filter2D(base_img, -1, kernel)

    img1_f = base_img.astype(np.float64)
    img2_f = filtered_img.astype(np.float64)
    mse = np.mean((img1_f - img2_f) ** 2)
    psnr_val = (
        float("inf") if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))
    )

    col1, col2 = st.columns(2)
    with col1:
      st.subheader("原始影像 (Original)")
      st.image(base_img, use_column_width=True)
    with col2:
      st.subheader("濾波後處理結果 (Filtered)")
      st.image(filtered_img, use_column_width=True, clamp=True)

    st.markdown("---")
    st.subheader("📈 影像品質量化評估 (PSNR Metrics)")
    m_col1, m_col2 = st.columns(2)
    with m_col1:
      st.metric(label="均方誤差 (MSE)", value=f"{mse:.4f}")
    with m_col2:
      st.metric(
          label="峰值信噪比 (PSNR)",
          value="∞ dB" if psnr_val == float("inf") else f"{psnr_val:.2f} dB",
      )

  elif module_choice == "人臉偵測與信心度分析":
    st.sidebar.markdown("### 👤 人臉偵測參數設定")
    scale_factor = st.sidebar.slider(
        "縮放比例因子 (Scale Factor)", 1.05, 1.5, 1.1, 0.05
    )
    min_neighbors = st.sidebar.slider(
        "最小相鄰矩形數 (Min Neighbors)", 1, 10, 5, 1
    )

    # 執行人臉偵測
    gray_for_detection = cv2.cvtColor(base_img, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(xml_filename)

    detected_img = base_img.copy()

    if face_cascade.empty():
      st.error("人臉模型載入失敗，請確認 XML 檔案是否存在。")
    else:
      # 使用 detectMultiScale3 同時取得邊界框與檢測信心權重 (Level Weights)
      faces, reject_levels, level_weights = face_cascade.detectMultiScale3(
          gray_for_detection,
          scaleFactor=scale_factor,
          minNeighbors=min_neighbors,
          minSize=(30, 30),
          outputRejectLevels=True,
      )

      face_count = len(faces)

      # 繪製介面與標記
      col1, col2 = st.columns(2)
      with col1:
        st.subheader("原始影像 (Original)")
        st.image(base_img, use_column_width=True)

      with col2:
        st.subheader(f"人臉偵測結果 (共偵測到 {face_count} 張人臉)")

        # 畫出邊界框與信心度標籤
        for i, (x, y, w, h) in enumerate(faces):
          # 畫綠色邊界框
          cv2.rectangle(
              detected_img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2
          )

          # 取得對應的信心權重作為信心分數指標
          weight = (
              level_weights[i]
              if (level_weights is not None and len(level_weights) > i)
              else 0.0
          )
          # 正規化顯示為直觀的信心分數百分比或指數
          confidence_score = float(weight)

          # 標註文字
          label = f"Face #{i+1} (Conf: {confidence_score:.1f})"
          cv2.putText(
              detected_img,
              label,
              (x, max(y - 10, 10)),
              cv2.FONT_HERSHEY_SIMPLEX,
              0.6,
              (0, 255, 0),
              2,
          )

        st.image(detected_img, use_column_width=True)

      # 統計數據顯示
      st.markdown("---")
      st.subheader("📊 人臉偵測量化統計")
      stat_col1, stat_col2 = st.columns(2)
      with stat_col1:
        st.metric(label="偵測到的人臉總數", value=f"{face_count} 張")
      with stat_col2:
        avg_conf = (
            np.mean(level_weights)
            if (level_weights is not None and len(level_weights) > 0)
            else 0.0
        )
        st.metric(label="平均檢測信心權重", value=f"{avg_conf:.2f}")

      # 詳細清單
      if face_count > 0:
        st.markdown("#### 🔍 各人臉詳細資訊清單")
        face_data = []
        for i, (x, y, w, h) in enumerate(faces):
          wgt = (
              level_weights[i]
              if (level_weights is not None and len(level_weights) > i)
              else 0.0
          )
          face_data.append({
              "編號": f"Face #{i+1}",
              "X 座標": x,
              "Y 座標": y,
              "寬度 (Width)": w,
              "高度 (Height)": h,
              "信心權重 (Score)": f"{wgt:.2f}",
          })
        st.table(face_data)
      else:
        st.info("畫面中未偵測到人臉，可嘗試調整左側的縮放比例或相鄰矩形數。")

    with st.expander("📖 點此查看人臉偵測演算法原理與說明"):
      st.markdown("""
        ### 經典人臉偵測 (Haar-Cascade / Viola-Jones) 原理
        
        1. **Haar-like 特徵**：
           利用黑白相鄰的矩形模板（如邊緣特徵、線條特徵、中心對應特徵）來計算影像局部的明暗對比。
        2. **積分圖 (Integral Image)**：
           透過積分圖技術加速矩形特徵值的計算，使演算法能達到即時運算。
        3. **級聯分類器 (Cascade Classifier)**：
           將許多簡單的分類器（Stumps）串聯成多級結構。背景區域會在前幾級快速被排除，只有疑似人臉的區域才會通過所有級聯考驗。
        4. **信心權重 (Level Weights)**：
           在多尺度檢測中，透過 `detectMultiScale3` 回傳的決策邊界距離（Level Weights）作為該偵測結果的信心指標，數值越高代表該區域符合人臉特徵的信心越強。
        """)
else:
  st.info(
      "請從左側側邊欄上傳影像檔案（支援 PNG、JPG/JPEG、BMP）以開始進行分析與人臉偵測！"
  )