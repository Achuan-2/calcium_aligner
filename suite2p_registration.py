import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter1d
import logging
from typing import Tuple, Optional
import warnings

warnings.filterwarnings("ignore")

# 尝试导入优化的FFT库
try:
    from mkl_fft import fft2, ifft2

    FFT_BACKEND = "mkl_fft"
except ImportError:
    try:
        import torch
        from torch.fft import fft2 as torch_fft2, ifft2 as torch_ifft2

        FFT_BACKEND = "torch"

        def fft2(data):
            data_torch = torch.from_numpy(data)
            result = torch.fft.fft(torch.fft.fft(data_torch, dim=-1), dim=-2)
            return result.cpu().numpy()

        def ifft2(data):
            data_torch = torch.from_numpy(data)
            result = torch.fft.ifft(torch.fft.ifft(data_torch, dim=-1), dim=-2)
            return result.cpu().numpy()

    except ImportError:
        from numpy.fft import fft2, ifft2

        FFT_BACKEND = "numpy"

from numpy.fft import ifftshift


class Suite2PRegistration:
    """
    Suite2p 图像配准模块
    基于相位相关法实现快速、精确的刚性配准

    关键改进：
    1. 修正数据维度为 (nframes, Ly, Lx)
    2. 正确实现相位相关算法
    3. 使用优化的FFT库
    4. 正确的掩码应用方法
    """

    def __init__(
        self,
        smooth_sigma: float = 1.15,
        maxregshift: float = 0.1,
        smooth_sigma_time: float = 0,
        niter: int = 8,
        num_matches: int = 20,
    ):
        """
        初始化配准参数

        Parameters
        ----------
        smooth_sigma : float
            空间平滑的标准差，默认1.15
        maxregshift : float
            最大位移比例 (相对于图像最小维度)
        smooth_sigma_time : float
            时间平滑的标准差
        niter : int
            参考图像迭代次数
        num_matches : int
            选择最佳匹配帧的数量
        """
        self.smooth_sigma = smooth_sigma
        self.maxregshift = maxregshift
        self.smooth_sigma_time = smooth_sigma_time
        self.niter = niter
        self.num_matches = num_matches
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"使用 FFT 后端: {FFT_BACKEND}")

    def compute_reference(self, frames: np.ndarray) -> np.ndarray:
        """
        从帧序列计算参考图像

        Parameters
        ----------
        frames : np.ndarray, shape (n_frames, Ly, Lx)
            输入帧序列

        Returns
        -------
        ref_image : np.ndarray, shape (Ly, Lx)
            计算得到的参考图像
        """
        self.logger.info("开始计算参考图像...")

        # 选择初始参考帧
        ref_image = self._pick_initial_reference(frames)

        # 迭代优化参考图像
        for iter_idx in range(self.niter):
            self.logger.info(f"参考图像迭代 {iter_idx + 1}/{self.niter}")

            # 计算掩码
            mask_mul, mask_offset = self._compute_masks(
                ref_image, 3 * self.smooth_sigma
            )

            # 应用掩码
            frames_taper = self._apply_masks(frames, mask_mul, mask_offset)

            # 准备参考图像用于相位相关
            ref_smooth = self._phasecorr_reference(ref_image, self.smooth_sigma)

            # 计算位移
            ymax, xmax, cmax = self._phasecorr(
                frames_taper, ref_smooth, self.maxregshift, self.smooth_sigma_time
            )

            # 位移帧
            aligned_frames = np.zeros_like(frames)
            for i in range(frames.shape[0]):
                aligned_frames[i] = self._shift_frame(frames[i], ymax[i], xmax[i])

            # 选择相关性最高的帧
            nmax = max(2, int(frames.shape[0] * (1 + iter_idx) / (2 * self.niter)))
            sort_idx = np.argsort(cmax)[::-1][1 : nmax + 1]  # 跳过第一个（自己）

            # 更新参考图像
            ref_image = np.mean(aligned_frames[sort_idx], axis=0).astype(frames.dtype)

            # 调整参考图像位置
            ref_image = self._shift_frame(
                ref_image,
                int(np.round(-ymax[sort_idx].mean())),
                int(np.round(-xmax[sort_idx].mean())),
            )

        self.logger.info("参考图像计算完成")
        return ref_image

    def register_frames(
        self, frames: np.ndarray, ref_image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        配准帧序列到参考图像

        Parameters
        ----------
        frames : np.ndarray, shape (n_frames, Ly, Lx)
            输入帧序列
        ref_image : np.ndarray, shape (Ly, Lx)
            参考图像

        Returns
        -------
        reg_frames : np.ndarray, shape (n_frames, Ly, Lx)
            配准后的帧序列
        ymax : np.ndarray, shape (n_frames,)
            Y方向位移
        xmax : np.ndarray, shape (n_frames,)
            X方向位移
        cmax : np.ndarray, shape (n_frames,)
            相关性最大值
        """
        self.logger.info("开始配准帧序列...")

        # 准备参考图像掩码
        mask_mul, mask_offset = self._compute_masks(ref_image, 3 * self.smooth_sigma)
        ref_fft = self._phasecorr_reference(ref_image, self.smooth_sigma)

        # 应用掩码
        frames_taper = self._apply_masks(frames, mask_mul, mask_offset)

        # 计算相位相关
        ymax, xmax, cmax = self._phasecorr(
            frames_taper, ref_fft, self.maxregshift, self.smooth_sigma_time
        )

        # 应用位移
        reg_frames = np.zeros_like(frames)
        for i in range(frames.shape[0]):
            reg_frames[i] = self._shift_frame(frames[i], ymax[i], xmax[i])

        self.logger.info(f"配准完成")
        return reg_frames, ymax, xmax, cmax

    def process_tiff_stack(
        self, input_path: str, output_path: str, ref_frames: Optional[int] = None
    ) -> dict:
        """
        处理TIFF文件堆栈

        Parameters
        ----------
        input_path : str
            输入TIFF文件路径
        output_path : str
            输出TIFF文件路径
        ref_frames : int, optional
            用于计算参考图像的帧数，默认使用所有帧

        Returns
        -------
        info : dict
            配准信息，包含位移数据
        """
        self.logger.info(f"读取TIFF文件: {input_path}")

        # 读取TIFF文件
        with tifffile.TiffFile(input_path) as tif:
            frames = tif.asarray()

        # 确保维度为 (n_frames, Ly, Lx)
        if frames.ndim == 2:
            frames = frames[np.newaxis, :, :]
        elif frames.ndim == 3:
            pass

        n_frames, Ly, Lx = frames.shape
        self.logger.info(f"图像尺寸: {Ly}x{Lx}, 帧数: {n_frames}")

        # 选择参考帧
        if ref_frames is None or ref_frames > n_frames:
            ref_frames = n_frames
        ref_frame_data = frames[:ref_frames]

        # 计算参考图像
        ref_image = self.compute_reference(ref_frame_data)

        # 配准所有帧
        reg_frames, ymax, xmax, cmax = self.register_frames(frames, ref_image)

        # 保存结果
        self.logger.info(f"保存配准结果: {output_path}")
        tifffile.imwrite(output_path, reg_frames)

        # 保存参考图像
        # ref_path = output_path.replace(".tif", "_ref.tif")
        # tifffile.imwrite(ref_path, ref_image)

        # 返回配准信息
        info = {
            "ymax": ymax.tolist(),
            "xmax": xmax.tolist(),
            "cmax": cmax.tolist(),
            "image_shape": (Ly, Lx),
            "n_frames": n_frames,
            "mean_shift_y": float(ymax.mean()),
            "mean_shift_x": float(xmax.mean()),
            "mean_correlation": float(cmax.mean()),
        }

        return info

    def _pick_initial_reference(self, frames: np.ndarray) -> np.ndarray:
        """选择初始参考帧"""
        n_frames, Ly, Lx = frames.shape

        # 重塑并中心化处理 - 注意维度是 (n_frames, Ly, Lx)
        reshaped = frames.reshape(n_frames, -1).astype(np.float32)
        reshaped -= reshaped.mean(axis=1, keepdims=True)

        # 计算归一化互相关矩阵
        cc_matrix = reshaped @ reshaped.T
        norm_matrix = np.outer(np.sqrt(np.diag(cc_matrix)), np.sqrt(np.diag(cc_matrix)))
        norm_cc_matrix = cc_matrix / (norm_matrix + 1e-10)

        # 选择最佳帧
        num_matches = min(self.num_matches, n_frames - 1)
        cc_sort = np.sort(norm_cc_matrix, axis=1)[:, ::-1]
        best_cc = cc_sort[:, 1 : num_matches + 1].mean(axis=1)
        best_frame_idx = np.argmax(best_cc)

        # 选择最相关的帧
        frame_corr = norm_cc_matrix[best_frame_idx, :]
        sort_idx = np.argsort(frame_corr)[::-1][: num_matches + 1]

        # 计算参考图像
        ref_image = frames[sort_idx].mean(axis=0).astype(frames.dtype)

        return ref_image

    def _compute_masks(
        self, ref_image: np.ndarray, mask_slope: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算空间掩码"""
        Ly, Lx = ref_image.shape

        # 计算空间taper
        mask_mul = self._spatial_taper(mask_slope, Ly, Lx)
        mask_offset = ref_image.mean() * (1 - mask_mul)

        return mask_mul, mask_offset

    def _spatial_taper(self, sig: float, Ly: int, Lx: int) -> np.ndarray:
        """空间taper掩码"""
        xx, yy = self._meshgrid_mean_centered(Lx, Ly)

        mY = ((Ly - 1) / 2) - 2 * sig
        mX = ((Lx - 1) / 2) - 2 * sig

        mask_y = 1 / (1 + np.exp((yy - mY) / sig))
        mask_x = 1 / (1 + np.exp((xx - mX) / sig))

        return mask_y * mask_x

    def _meshgrid_mean_centered(self, x: int, y: int) -> Tuple[np.ndarray, np.ndarray]:
        """生成中心化的网格"""
        x_vals = np.arange(x)
        y_vals = np.arange(y)

        x_centered = np.abs(x_vals - x_vals.mean())
        y_centered = np.abs(y_vals - y_vals.mean())

        return np.meshgrid(x_centered, y_centered)

    def _phasecorr_reference(
        self, ref_image: np.ndarray, smooth_sigma: float
    ) -> np.ndarray:
        """准备参考图像用于相位相关"""
        # FFT变换并归一化
        ref_fft = fft2(ref_image)
        ref_fft = np.conj(ref_fft)  # 取共轭
        ref_fft = ref_fft / (1e-5 + np.abs(ref_fft))

        # 应用高斯滤波
        gaussian_filter = self._gaussian_fft(smooth_sigma, *ref_image.shape)
        ref_fft *= gaussian_filter

        return ref_fft.astype(np.complex64)

    def _phasecorr(
        self,
        data: np.ndarray,
        ref_fft: np.ndarray,
        maxregshift: float,
        smooth_sigma_time: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """相位相关计算位移"""
        n_frames, Ly, Lx = data.shape

        # 计算最大位移
        min_dim = min(Ly, Lx)
        lcorr = int(np.minimum(np.round(maxregshift * min_dim), min_dim // 2))

        # 卷积计算
        data_conv = self._convolve(data, ref_fft)

        # 提取相关区域 - 使用 np.block 组合四个角落
        cc = np.real(
            np.block(
                [
                    [
                        data_conv[:, -lcorr:, -lcorr:],
                        data_conv[:, -lcorr:, : lcorr + 1],
                    ],
                    [
                        data_conv[:, : lcorr + 1, -lcorr:],
                        data_conv[:, : lcorr + 1, : lcorr + 1],
                    ],
                ]
            )
        )

        # 时间平滑 - 在相关矩阵上平滑
        if smooth_sigma_time > 0:
            cc = self._temporal_smooth(cc, smooth_sigma_time)

        # 找到最大值位置
        ymax = np.zeros(n_frames, dtype=np.int32)
        xmax = np.zeros(n_frames, dtype=np.int32)
        cmax = np.zeros(n_frames, dtype=np.float32)

        for t in range(n_frames):
            ymax[t], xmax[t] = np.unravel_index(
                np.argmax(cc[t], axis=None), (2 * lcorr + 1, 2 * lcorr + 1)
            )
            cmax[t] = cc[t, ymax[t], xmax[t]]

        # 转换为相对于中心的位移
        ymax -= lcorr
        xmax -= lcorr

        return ymax, xmax, cmax

    def _convolve(self, mov: np.ndarray, img_fft: np.ndarray) -> np.ndarray:
        """频域卷积 - 批量处理多帧"""
        # mov: (n_frames, Ly, Lx)
        # img_fft: (Ly, Lx)

        # 对每一帧进行FFT
        mov_fft = fft2(mov)  # 对最后两个维度进行FFT

        # 归一化并应用参考图像FFT
        mov_fft = mov_fft / (1e-5 + np.abs(mov_fft))
        mov_fft *= img_fft[np.newaxis, :, :]  # 广播到所有帧

        # 逆FFT
        return np.real(ifft2(mov_fft))

    def _temporal_smooth(self, data: np.ndarray, sigma: float) -> np.ndarray:
        """时间方向平滑 - 在第一个维度（时间）上平滑"""
        if sigma <= 0:
            return data
        # 使用高斯滤波，只在时间轴（axis=0）上平滑
        return gaussian_filter1d(data, sigma=sigma, axis=0)

    def _shift_frame(self, frame: np.ndarray, dy: int, dx: int) -> np.ndarray:
        """位移单帧 - 注意这里dy, dx已经包含正确的符号"""
        return np.roll(frame, (-dy, -dx), axis=(0, 1))

    def _apply_masks(
        self, data: np.ndarray, mask_mul: np.ndarray, mask_offset: np.ndarray
    ) -> np.ndarray:
        """应用掩码 - data是(n_frames, Ly, Lx)格式"""
        # 广播掩码到所有帧
        return (
            data.astype(np.float32) * mask_mul[np.newaxis, :, :]
            + mask_offset[np.newaxis, :, :]
        )

    def _gaussian_fft(self, sig: float, Ly: int, Lx: int) -> np.ndarray:
        """FFT域高斯滤波器"""
        xx, yy = self._meshgrid_mean_centered(Lx, Ly)

        hgx = np.exp(-((xx / sig) ** 2) / 2)
        hgy = np.exp(-((yy / sig) ** 2) / 2)
        hgg = hgy * hgx
        hgg /= hgg.sum()

        return np.real(fft2(ifftshift(hgg)))
