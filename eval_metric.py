# Infra
# import cv2
# import numpy as np
# from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
# import torch
# import lpips
# from torchvision import transforms, models
# from scipy import linalg
# import os

# # 检查torchvision版本并选择合适的导入方式
# try:
#     from torchvision.models.video import R3D_18_Weights
#     HAS_NEW_TORCHVISION = True
# except ImportError:
#     HAS_NEW_TORCHVISION = False
#     print("警告: 使用旧版torchvision API")

# # -------------------------------
# # 1. 读取视频帧
# # -------------------------------
# def read_video_frames(video_path, to_rgb=True, resize=None):
#     """安全地读取视频帧，处理各种错误情况"""
#     if not os.path.exists(video_path):
#         print(f"错误: 视频文件不存在: {video_path}")
#         return []
    
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"错误: 无法打开视频文件: {video_path}")
#         return []
    
#     frames = []
#     frame_count = 0
#     max_frames = 100  # 限制最大帧数，避免内存问题
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
            
#         if to_rgb:
#             # 确保是3通道图像
#             if len(frame.shape) == 2:  # 灰度图
#                 frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
#             elif frame.shape[2] == 3:  # BGR
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             elif frame.shape[2] == 4:  # BGRA
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        
#         if resize is not None:
#             frame = cv2.resize(frame, resize)
        
#         frames.append(frame)
#         frame_count += 1
        
#         if frame_count >= max_frames:
#             break
    
#     cap.release()
    
#     if len(frames) == 0:
#         print(f"警告: 视频 {video_path} 没有读取到任何帧")
    
#     print(f"从 {video_path} 读取了 {len(frames)} 帧")
#     return frames


# # -------------------------------
# # 2. 计算 PSNR 和 SSIM
# # -------------------------------
# def compute_psnr_ssim(gt_frames, pre_frames):
#     if len(gt_frames) == 0 or len(pre_frames) == 0:
#         print("错误: 无法计算PSNR/SSIM，帧列表为空")
#         return 0, 0
    
#     psnr_list, ssim_list = [], []
#     min_len = min(len(gt_frames), len(pre_frames))
    
#     for i in range(min_len):
#         gt = gt_frames[i].astype(np.float32)
#         pre = pre_frames[i].astype(np.float32)
        
#         # 确保图像尺寸相同
#         if gt.shape != pre.shape:
#             pre = cv2.resize(pre, (gt.shape[1], gt.shape[0]))
        
#         try:
#             psnr_val = psnr(gt, pre, data_range=255)
#             # SSIM需要处理多通道
#             if len(gt.shape) == 3 and gt.shape[2] == 3:
#                 ssim_val = ssim(gt, pre, data_range=255, channel_axis=2, win_size=7)
#             else:
#                 ssim_val = ssim(gt, pre, data_range=255, win_size=7)
            
#             psnr_list.append(psnr_val)
#             ssim_list.append(ssim_val)
#         except Exception as e:
#             print(f"计算PSNR/SSIM时出错: {e}")
#             continue
    
#     if len(psnr_list) == 0:
#         return 0, 0
    
#     return np.mean(psnr_list), np.mean(ssim_list)


# # -------------------------------
# # 3. 计算 LPIPS
# # -------------------------------
# def compute_lpips(gt_frames, pre_frames, device='cpu'):
#     if len(gt_frames) == 0 or len(pre_frames) == 0:
#         print("错误: 无法计算LPIPS，帧列表为空")
#         return 1.0  # 最差情况
    
#     try:
#         loss_fn = lpips.LPIPS(net='alex').to(device)
#         to_tensor = transforms.ToTensor()
#         lpips_scores = []
#         min_len = min(len(gt_frames), len(pre_frames))
        
#         for i in range(min_len):
#             gt = gt_frames[i]
#             pre = pre_frames[i]
            
#             # 确保图像尺寸相同
#             if gt.shape != pre.shape:
#                 pre = cv2.resize(pre, (gt.shape[1], gt.shape[0]))
            
#             # 确保是3通道
#             if len(gt.shape) == 2:
#                 gt = cv2.cvtColor(gt, cv2.COLOR_GRAY2RGB)
#             if len(pre.shape) == 2:
#                 pre = cv2.cvtColor(pre, cv2.COLOR_GRAY2RGB)
            
#             gt_t = to_tensor(gt).unsqueeze(0).to(device) * 2 - 1
#             pre_t = to_tensor(pre).unsqueeze(0).to(device) * 2 - 1
            
#             with torch.no_grad():
#                 d = loss_fn(gt_t, pre_t)
#             lpips_scores.append(d.item())
        
#         return np.mean(lpips_scores) if lpips_scores else 1.0
#     except Exception as e:
#         print(f"计算LPIPS时出错: {e}")
#         return 1.0


# # -------------------------------
# # 4. 计算 FVD（简化版，用 R3D-18 特征）
# # -------------------------------
# def extract_video_feature(frames, device='cpu'):
#     """提取视频特征，处理空帧情况"""
#     if len(frames) == 0:
#         print("错误: 无法提取视频特征，帧列表为空")
#         return np.zeros(512)  # 返回零向量
    
#     try:
#         # 根据torchvision版本选择合适的API
#         if HAS_NEW_TORCHVISION:
#             weights = R3D_18_Weights.DEFAULT
#             model = models.video.r3d_18(weights=weights).to(device)
#         else:
#             # 使用旧版API
#             model = models.video.r3d_18(pretrained=True).to(device)
        
#         model.eval()

#         transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize((112, 112)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], 
#                                std=[0.22803, 0.22145, 0.216989])
#         ])

#         # 选择前16帧或所有帧（如果少于16帧）
#         num_frames = min(16, len(frames))
#         selected_frames = frames[:num_frames]
        
#         tensors = []
#         for frame in selected_frames:
#             # 确保是3通道
#             if len(frame.shape) == 2:
#                 frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
#             tensor = transform(frame)
#             tensors.append(tensor)
        
#         if not tensors:
#             return np.zeros(512)
            
#         # 堆叠张量 (C, T, H, W)
#         video_tensor = torch.stack(tensors, dim=1).unsqueeze(0).to(device)

#         with torch.no_grad():
#             # 使用R3D-18的前向传播
#             features = model(video_tensor)
#             feat = features.mean(dim=[2, 3, 4]) if features.dim() > 2 else features
        
#         return feat.cpu().numpy().flatten()
        
#     except Exception as e:
#         print(f"提取视频特征时出错: {e}")
#         return np.zeros(512)


# def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
#     """计算Frechet距离"""
#     diff = mu1 - mu2
    
#     # 防止奇异矩阵
#     sigma1 = sigma1 + eps * np.eye(sigma1.shape[0])
#     sigma2 = sigma2 + eps * np.eye(sigma2.shape[0])
    
#     covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
#     if np.iscomplexobj(covmean):
#         covmean = covmean.real
    
#     return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)


# def compute_fvd(gt_frames, pre_frames, device='cpu'):
#     """计算FVD，处理各种错误情况"""
#     if len(gt_frames) == 0 or len(pre_frames) == 0:
#         print("错误: 无法计算FVD，帧列表为空")
#         return float('inf')
    
#     try:
#         feat1 = extract_video_feature(gt_frames, device)
#         feat2 = extract_video_feature(pre_frames, device)
        
#         # 计算均值和协方差
#         mu1, mu2 = np.mean(feat1), np.mean(feat2)
        
#         # 对于单个样本，使用方差作为协方差矩阵的近似
#         sigma1 = np.cov(feat1) if len(feat1) > 1 else np.var(feat1) * np.eye(len(feat1))
#         sigma2 = np.cov(feat2) if len(feat2) > 1 else np.var(feat2) * np.eye(len(feat2))
        
#         # 确保是2D数组
#         if sigma1.ndim == 0:
#             sigma1 = np.array([[sigma1]])
#         if sigma2.ndim == 0:
#             sigma2 = np.array([[sigma2]])
        
#         return frechet_distance(np.array([mu1]), sigma1, np.array([mu2]), sigma2)
        
#     except Exception as e:
#         print(f"计算FVD时出错: {e}")
#         return float('inf')


# # -------------------------------
# # 5. 主函数
# # -------------------------------
# if __name__ == "__main__":
#     # 使用绝对路径确保文件可访问
#     gt_path = "/data/zhounan/DiffSynth-Studio/data/my_fire_mask/Infrared_832_480_17/video_007.mp4"
#     pre_path = "/data/zhounan/DiffSynth-Studio/My_Results/video_Wan2.1-VACE-1.3B_Type1_e86_007-consistency.mp4"
#     # pre_path = "data/my_fire_mask/Infrared_832_480_17/video_006.mp4"
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     print(f"✅ 使用设备: {device}")

#     # 读取视频帧
#     print("正在读取真实视频...")
#     gt_frames = read_video_frames(gt_path)  # 调整尺寸以节省内存

#     print("正在读取预测视频...")
#     pre_frames = read_video_frames(pre_path)
#     pre_frames = pre_frames[16:33]
#     # pre_frames = pre_frames[0:16]

#     # 检查是否成功读取
#     if len(gt_frames) == 0 or len(pre_frames) == 0:
#         print("❌ 错误: 无法读取视频文件，请检查路径和文件格式")
#         exit(1)

#     # 对齐帧数与尺寸
#     min_len = min(len(gt_frames), len(pre_frames))
#     gt_frames = gt_frames[:min_len]
#     pre_frames = pre_frames[:min_len]

#     print(f"视频帧数: {min_len}")

#     # ---- 计算各指标 ----
#     print("计算PSNR和SSIM...")
#     psnr_val, ssim_val = compute_psnr_ssim(gt_frames, pre_frames)
    
#     print("计算LPIPS...")
#     lpips_val = compute_lpips(gt_frames, pre_frames, device=device)
    
#     print("计算FVD...")
#     fvd_val = compute_fvd(gt_frames, pre_frames, device=device)

#     print("\n📊 计算结果：")
#     print(f"PSNR  = {psnr_val:.4f}")
#     print(f"SSIM  = {ssim_val:.4f}")
#     print(f"LPIPS = {lpips_val:.4f}")
#     print(f"FVD(approx) = {fvd_val:.4f}")

# Mask
import os
import cv2
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
import matplotlib.pyplot as plt
import torch

def load_video_as_mask(video_path, num_frames=17, height=480, width=832, threshold=0.9):
    """
    将视频读为 mask 序列 (T, H, W), 二值化假设掩码通道为灰度或单通道。
    如果是 RGB 掩码，需要先转为灰度或取某一通道。
    threshold: 灰度>threshold 时判为 1.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Cannot read frame from {video_path}")
        # 假设掩码是 BGR 灰度叠加或彩色，但掩码用白色=1，黑色=0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 归一化
        norm = gray.astype('float32') / 255.0
        binary = (norm >= threshold).astype(np.uint8)
        frames.append(binary)
    cap.release()
    arr = np.stack(frames, axis=0)  # (T, H, W)
    return arr

def compute_metrics(true_mask, pred_mask):
    """
    true_mask, pred_mask: shape (T, H, W), 0/1 values
    Returns: dict of AUPRC, F1, IoU, MSE
    """
    T, H, W = true_mask.shape
    # Flatten all pixels across time
    y_true = true_mask.reshape(-1)
    y_pred = pred_mask.reshape(-1)
    # AUPRC: need scores, but这里用 pred_mask (0/1)作为分数近似
    # 如果你的预测是概率，则用概率。这里用二值结果一样可得 precision_recall_curve
    ap = average_precision_score(y_true, y_pred)
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    # 找到最佳 F1 across thresholds
    f1s = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.nanargmax(f1s)
    best_f1 = f1s[best_idx]
    # IoU
    intersection = np.logical_and(y_true==1, y_pred==1).sum()
    union = np.logical_or(y_true==1, y_pred==1).sum()
    iou = intersection / union if union > 0 else 0.0
    # MSE (按像素值 0/1)
    mse = np.mean((y_true.astype('float32') - y_pred.astype('float32'))**2)
    return {"AUPRC": ap, "BestF1": best_f1, "IoU": iou, "MSE": mse}

def visualize_masks(true_mask, pred_mask, output_dir=None, max_frames=5):
    """
    true_mask, pred_mask: numpy arrays of shape (T, H, W)
    输出前 max_frames 帧的可视化对比：
      - 左侧：真值掩码
      - 右侧：预测掩码
    如果指定 output_dir，会将图保存为 .jpg 文件。
    """
    T, H, W = true_mask.shape
    os.makedirs(output_dir, exist_ok=True) if output_dir else None
    
    for i in range(min(T, max_frames)):
        plt.figure(figsize=(8,4))
        plt.suptitle(f"Frame {i}")
        
        plt.subplot(1,2,1)
        plt.title("True mask")
        plt.imshow(true_mask[i], cmap='gray')
        plt.axis('off')
        
        plt.subplot(1,2,2)
        plt.title("Pred mask")
        plt.imshow(pred_mask[i], cmap='gray')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        if output_dir:
            save_path = os.path.join(output_dir, f"mask_compare_frame_{i:02d}.jpg")
            plt.savefig(save_path)
            plt.close()

if __name__ == "__main__":
    pred_path = "/data/zhounan/DiffSynth-Studio/My_Results/video_Wan2.1-VACE-1.3B_Multi-Setting1_e86_007-consistency.mp4"
    true_path = "/data/zhounan/DiffSynth-Studio/data/5_Regions/Lida_video/Mask_832_480_17/video_007.mp4"
    # num_frames = 17
    height = 480
    width = 832

    true_mask = load_video_as_mask(true_path, num_frames=16, height=height, width=width, threshold=0.5)
    pred_mask = load_video_as_mask(pred_path, num_frames=33, height=height, width=width, threshold=0.9)
    pred_mask = pred_mask[17:33]
    # pred_mask = pred_mask[:17]

    # visualize_masks(true_mask[-5:], pred_mask[-5:], output_dir="mask_vis", max_frames=16)

    metrics = compute_metrics(true_mask, pred_mask)
    print("Metrics:", metrics)