#!/usr/bin/env python3
"""
钙成像配准软件 - 基于PySide6和Suite2p
"""

import sys
import json
import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import tifffile
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QPushButton,
    QLabel,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QFileDialog,
    QMessageBox,
    QProgressBar,
    QGroupBox,
    QFormLayout,
    QSlider,
    QCheckBox,
    QTextEdit,
    QSplitter,
    QProgressDialog,
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap
import matplotlib

matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import Slider as MplSlider

from suite2p_registration import Suite2PRegistration

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class RegistrationThread(QThread):
    """配准处理线程"""

    progress = Signal(int)
    finished = Signal(dict)
    error = Signal(str)
    progress_text = Signal(str)

    def __init__(
        self,
        input_path: str,
        output_path: str,
        registrator: Suite2PRegistration,
        ref_image_path: Optional[str] = None,
    ):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.registrator = registrator
        self.ref_image_path = ref_image_path

    def run(self):
        try:
            # 如果有自定义参考图像，加载它
            if self.ref_image_path and Path(self.ref_image_path).exists():
                logger.info(f"使用自定义参考图像: {self.ref_image_path}")
                self.progress_text.emit(
                    f"加载参考图像: {Path(self.ref_image_path).name}"
                )
                with tifffile.TiffFile(self.ref_image_path) as tif:
                    ref_image = tif.asarray()
            else:
                ref_image = None

            # 读取输入数据
            logger.info(f"读取输入文件: {self.input_path}")
            self.progress_text.emit(f"读取输入文件: {Path(self.input_path).name}")
            self.progress.emit(10)
            with tifffile.TiffFile(self.input_path) as tif:
                frames = tif.asarray()

            # 确保维度为 (n_frames, Ly, Lx)
            if frames.ndim == 2:
                frames = frames[np.newaxis, :, :]
            elif frames.ndim == 3:
                pass

            n_frames, Ly, Lx = frames.shape

            # 计算参考图像或加载自定义参考图像
            if ref_image is None:
                logger.info("计算参考图像...")
                self.progress_text.emit("计算参考图像...")
                self.progress.emit(30)
                ref_frame_data = frames[: min(n_frames, 300)]  # 默认使用300帧
                ref_image = self.registrator.compute_reference(ref_frame_data)

            # 配准帧
            logger.info("开始配准...")
            self.progress_text.emit("配准中...")
            self.progress.emit(50)
            reg_frames, ymax, xmax, cmax = self.registrator.register_frames(
                frames, ref_image
            )

            # 保存结果
            logger.info(f"保存配准结果: {self.output_path}")
            self.progress_text.emit("保存结果...")
            self.progress.emit(80)
            # reg_frames 已经是 (n_frames, Ly, Lx) 格式，直接保存
            tifffile.imwrite(self.output_path, reg_frames)

            # 保存参考图像
            ref_path = self.output_path.replace(".tif", "_ref.tif")
            tifffile.imwrite(ref_path, ref_image)

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
                "output_path": self.output_path,
                "ref_path": ref_path,
            }

            self.progress.emit(100)
            self.finished.emit(info)

        except Exception as e:
            logger.error(f"配准失败: {str(e)}")
            self.error.emit(str(e))


class ImageCanvas(FigureCanvas):
    """自定义图像显示画布"""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.image_data = None
        self.current_frame = 0
        self.im = None  # 保存图像对象，避免重复创建

        # 初始化显示黑色背景
        self._init_black_background()

    def _init_black_background(self):
        """初始化黑色背景"""
        self.axes.clear()
        self.axes.set_facecolor("black")
        self.axes.set_xlim(0, 1)
        self.axes.set_ylim(0, 1)
        self.axes.axis("off")
        self.draw()

    def display_image(
        self,
        image_data: np.ndarray,
        frame_idx: int = 0,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ):
        """显示图像"""
        self.image_data = image_data
        self.current_frame = frame_idx

        if image_data is None:
            self._init_black_background()
            return

        # 获取当前帧 - image_data 格式是 (n_frames, Ly, Lx)
        if image_data.ndim == 3:
            if frame_idx >= image_data.shape[0]:
                frame_idx = image_data.shape[0] - 1
            frame = image_data[frame_idx]
        else:
            frame = image_data

        # 自动计算亮度范围
        if vmin is None:
            vmin = np.percentile(frame, 1)
        if vmax is None:
            vmax = np.percentile(frame, 99)

        # 显示图像 - 重用图像对象避免重复创建
        if self.im is None:
            # 第一次显示
            self.axes.clear()
            self.im = self.axes.imshow(frame, cmap="gray", vmin=vmin, vmax=vmax)
            self.axes.set_title(
                f"Frame {frame_idx + 1}/{image_data.shape[0] if image_data.ndim == 3 else 1}"
            )
            self.axes.axis("off")
        else:
            # 更新现有图像数据
            self.im.set_data(frame)
            self.im.set_clim(vmin, vmax)
            self.axes.set_title(
                f"Frame {frame_idx + 1}/{image_data.shape[0] if image_data.ndim == 3 else 1}"
            )

        self.draw()

    def get_frame_count(self) -> int:
        """获取帧数"""
        if self.image_data is None:
            return 0
        return self.image_data.shape[0] if self.image_data.ndim == 3 else 1


class SingleRegistrationTab(QWidget):
    """单个配准标签页"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.input_data = None
        self.registered_data = None
        self.registrator = Suite2PRegistration()
        self.output_folder_edit = None  # 初始化输出文件夹编辑框
        self.progress_dialog = None
        self.init_ui()

    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)

        # 文件选择区域
        file_group = QGroupBox("文件选择")
        file_layout = QHBoxLayout()

        self.input_file_edit = QLineEdit()
        self.input_file_edit.setPlaceholderText("选择输入TIFF文件...")
        self.input_file_btn = QPushButton("浏览...")
        self.input_file_btn.clicked.connect(self.select_input_file)

        self.output_folder_edit = QLineEdit()
        self.output_folder_edit.setPlaceholderText(
            "选择输出文件夹（默认为输入文件夹）..."
        )
        self.output_folder_btn = QPushButton("浏览...")
        self.output_folder_btn.clicked.connect(self.select_output_folder)

        self.ref_file_edit = QLineEdit()
        self.ref_file_edit.setPlaceholderText("选择参考图像(可选)...")
        self.ref_file_btn = QPushButton("浏览...")
        self.ref_file_btn.clicked.connect(self.select_ref_file)

        file_layout.addWidget(QLabel("输入:"))
        file_layout.addWidget(self.input_file_edit)
        file_layout.addWidget(self.input_file_btn)
        file_layout.addWidget(QLabel("输出文件夹:"))
        file_layout.addWidget(self.output_folder_edit)
        file_layout.addWidget(self.output_folder_btn)
        file_layout.addWidget(QLabel("参考:"))
        file_layout.addWidget(self.ref_file_edit)
        file_layout.addWidget(self.ref_file_btn)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # 参数设置区域
        param_group = QGroupBox("配准参数")
        param_layout = QFormLayout()

        self.smooth_sigma_spin = QDoubleSpinBox()
        self.smooth_sigma_spin.setRange(0.1, 10.0)
        self.smooth_sigma_spin.setValue(1.125)
        self.smooth_sigma_spin.setSingleStep(0.1)

        self.max_shift_spin = QDoubleSpinBox()
        self.max_shift_spin.setRange(0.01, 0.5)
        self.max_shift_spin.setValue(0.1)
        self.max_shift_spin.setSingleStep(0.01)

        self.smooth_sigma_time_spin = QDoubleSpinBox()
        self.smooth_sigma_time_spin.setRange(0.0, 10.0)
        self.smooth_sigma_time_spin.setValue(0.0)
        self.smooth_sigma_time_spin.setSingleStep(0.1)

        self.ref_frames_spin = QSpinBox()
        self.ref_frames_spin.setRange(0, 10000)
        self.ref_frames_spin.setValue(300)
        self.ref_frames_spin.setSingleStep(100)

        param_layout.addRow("空间平滑σ:", self.smooth_sigma_spin)
        param_layout.addRow("最大位移比例:", self.max_shift_spin)
        param_layout.addRow("时间平滑σ:", self.smooth_sigma_time_spin)
        param_layout.addRow("参考帧数:", self.ref_frames_spin)

        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # 控制按钮
        btn_layout = QHBoxLayout()
        self.register_btn = QPushButton("开始配准")
        self.register_btn.clicked.connect(self.start_registration)
        self.register_btn.setEnabled(False)

        btn_layout.addWidget(self.register_btn)
        layout.addLayout(btn_layout)

        # 图像显示区域
        splitter = QSplitter(Qt.Horizontal)

        # 输入图像
        input_group = QGroupBox("输入图像")
        input_layout = QVBoxLayout()
        self.input_canvas = ImageCanvas(self, width=4, height=3)
        input_layout.addWidget(self.input_canvas)

        # 输入图像控制
        input_ctrl_layout = QVBoxLayout()

        # 帧控制
        frame_layout = QHBoxLayout()
        self.input_frame_slider = QSlider(Qt.Horizontal)
        self.input_frame_slider.setEnabled(False)
        self.input_frame_slider.valueChanged.connect(self.update_input_display)

        frame_layout.addWidget(QLabel("帧:"))
        frame_layout.addWidget(self.input_frame_slider)
        input_ctrl_layout.addLayout(frame_layout)

        # 亮度范围控制
        brightness_layout = QHBoxLayout()

        input_layout.addLayout(input_ctrl_layout)
        input_group.setLayout(input_layout)
        splitter.addWidget(input_group)

        # 配准后图像
        output_group = QGroupBox("配准后图像")
        output_layout = QVBoxLayout()
        self.output_canvas = ImageCanvas(self, width=4, height=3)
        output_layout.addWidget(self.output_canvas)

        # 输出图像控制
        output_ctrl_layout = QHBoxLayout()
        self.output_frame_slider = QSlider(Qt.Horizontal)
        self.output_frame_slider.setEnabled(False)
        self.output_frame_slider.valueChanged.connect(self.update_output_display)

        output_ctrl_layout.addWidget(QLabel("帧:"))
        output_ctrl_layout.addWidget(self.output_frame_slider)

        output_layout.addLayout(output_ctrl_layout)
        output_group.setLayout(output_layout)
        splitter.addWidget(output_group)

        layout.addWidget(splitter)

        # 状态信息
        self.status_label = QLabel("就绪")
        layout.addWidget(self.status_label)

    def select_input_file(self):
        """选择输入文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择输入TIFF文件", "", "TIFF Files (*.tif *.tiff)"
        )
        if file_path:
            self.input_file_edit.setText(file_path)
            # 自动加载并显示图像
            self.load_data()

    def select_output_folder(self):
        """选择输出文件夹"""
        folder_path = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if folder_path:
            self.output_folder_edit.setText(folder_path)

    def select_ref_file(self):
        """选择参考文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择参考图像", "", "TIFF Files (*.tif *.tiff)"
        )
        if file_path:
            self.ref_file_edit.setText(file_path)

    def load_data(self):
        """加载输入数据"""
        input_path = self.input_file_edit.text()
        if not input_path or not Path(input_path).exists():
            QMessageBox.warning(self, "错误", "请选择有效的输入文件")
            return

        try:
            self.status_label.setText("加载数据中...")
            QApplication.processEvents()

            # 读取TIFF文件
            with tifffile.TiffFile(input_path) as tif:
                self.input_data = tif.asarray()

            # 确保维度为 (n_frames, Ly, Lx)
            if self.input_data.ndim == 2:
                self.input_data = self.input_data[np.newaxis, :, :]
            elif self.input_data.ndim == 3:
                pass

            # 更新UI
            n_frames = self.input_data.shape[0]
            self.input_frame_slider.setMaximum(max(0, n_frames - 1))
            self.input_frame_slider.setEnabled(n_frames > 1)
            self.ref_frames_spin.setMaximum(n_frames)

            # 显示第一帧
            self.update_input_display()

            self.status_label.setText(f"加载完成: {n_frames}帧")
            self.register_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载数据失败: {str(e)}")
            self.status_label.setText("加载失败")

    def update_input_display(self):
        """更新输入显示"""
        if self.input_data is not None:
            frame_idx = self.input_frame_slider.value()

            # input_data 格式是 (n_frames, Ly, Lx)
            if self.input_data.ndim == 3:
                frame = self.input_data[frame_idx]
            else:
                frame = self.input_data

            self.input_canvas.display_image(self.input_data, frame_idx, None, None)

    def update_output_display(self):
        """更新输出显示"""
        if self.registered_data is not None:
            frame_idx = self.output_frame_slider.value()
            # registered_data 格式是 (n_frames, Ly, Lx)
            if self.registered_data.ndim == 3:
                frame = self.registered_data[frame_idx]
            else:
                frame = self.registered_data

            self.output_canvas.display_image(self.registered_data, frame_idx)

    def start_registration(self):
        """开始配准"""
        input_path = self.input_file_edit.text()

        if not input_path:
            QMessageBox.warning(self, "错误", "请选择输入文件")
            return

        # 确保数据已加载
        if self.input_data is None:
            self.load_data()
            if self.input_data is None:  # 如果加载失败
                return

        # 获取输出文件夹路径（如果为空则使用输入文件夹）
        output_folder = self.output_folder_edit.text()
        if not output_folder:
            output_folder = str(Path(input_path).parent)

        # 生成输出文件名：输入文件名 + "_reg" + 扩展名
        input_file = Path(input_path)
        output_filename = f"{input_file.stem}_reg{input_file.suffix}"
        output_path = str(Path(output_folder) / output_filename)

        # 检查输出文件是否已存在
        if Path(output_path).exists():
            reply = QMessageBox.question(
                self,
                "文件已存在",
                f"输出文件 {output_filename} 已存在，是否覆盖？",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.No:
                self.status_label.setText("配准已取消")
                return

        # 更新配准参数
        self.registrator.smooth_sigma = self.smooth_sigma_spin.value()
        self.registrator.maxregshift = self.max_shift_spin.value()
        self.registrator.smooth_sigma_time = self.smooth_sigma_time_spin.value()

        # 禁用按钮
        self.register_btn.setEnabled(False)
        self.status_label.setText("配准中...")

        # 创建进度对话框
        self.progress_dialog = QProgressDialog("配准中...", "取消", 0, 100, self)
        self.progress_dialog.setWindowTitle("配准进度")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.setAutoReset(True)
        self.progress_dialog.show()

        # 创建配准线程
        ref_image_path = (
            self.ref_file_edit.text() if self.ref_file_edit.text() else None
        )
        self.reg_thread = RegistrationThread(
            input_path, output_path, self.registrator, ref_image_path
        )
        self.reg_thread.finished.connect(self.registration_finished)
        self.reg_thread.error.connect(self.registration_error)
        self.reg_thread.progress.connect(self.progress_dialog.setValue)
        self.reg_thread.progress_text.connect(self.progress_dialog.setLabelText)
        self.reg_thread.start()

    def registration_finished(self, info: dict):
        """配准完成"""
        # 关闭进度对话框
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None

        try:
            # 加载配准结果
            with tifffile.TiffFile(info["output_path"]) as tif:
                self.registered_data = tif.asarray()

            # 确保维度为 (n_frames, Ly, Lx)
            if self.registered_data.ndim == 2:
                self.registered_data = self.registered_data[np.newaxis, :, :]
            elif self.registered_data.ndim == 3:
                pass
            # 更新UI
            n_frames = self.registered_data.shape[0]
            self.output_frame_slider.setMaximum(max(0, n_frames - 1))
            self.output_frame_slider.setEnabled(n_frames > 1)

            # 显示第一帧
            self.update_output_display()

            # 显示配准信息
            info_text = f"配准完成!"
            self.status_label.setText(info_text)

            # 保存配准信息
            info_path = info["output_path"].replace(".tif", "_reg_info.json")
            with open(info_path, "w") as f:
                json.dump(info, f, indent=2)

            QMessageBox.information(self, "成功", "配准完成!")

        except Exception as e:
            self.status_label.setText("显示结果失败")
            QMessageBox.warning(self, "警告", f"配准完成但显示结果失败: {str(e)}")

        finally:
            self.register_btn.setEnabled(True)

    def registration_error(self, error_msg: str):
        """配准错误"""
        # 关闭进度对话框
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None

        self.status_label.setText("配准失败")
        QMessageBox.critical(self, "错误", f"配准失败: {error_msg}")
        self.register_btn.setEnabled(True)


class BatchRegistrationTab(QWidget):
    """批量配准标签页"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.registrator = Suite2PRegistration()
        self.init_ui()

    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)

        # 文件夹选择
        folder_group = QGroupBox("批量处理设置")
        folder_layout = QHBoxLayout()

        self.input_folder_edit = QLineEdit()
        self.input_folder_edit.setPlaceholderText("选择输入文件夹...")
        self.input_folder_btn = QPushButton("浏览...")
        self.input_folder_btn.clicked.connect(self.select_input_folder)

        self.output_folder_edit = QLineEdit()
        self.output_folder_edit.setPlaceholderText("选择输出文件夹...")
        self.output_folder_btn = QPushButton("浏览...")
        self.output_folder_btn.clicked.connect(self.select_output_folder)

        self.ref_file_edit = QLineEdit()
        self.ref_file_edit.setPlaceholderText("选择参考图像(可选，用于所有文件)...")
        self.ref_file_btn = QPushButton("浏览...")
        self.ref_file_btn.clicked.connect(self.select_ref_file)

        folder_layout.addWidget(QLabel("输入文件夹:"))
        folder_layout.addWidget(self.input_folder_edit)
        folder_layout.addWidget(self.input_folder_btn)
        folder_layout.addWidget(QLabel("输出文件夹:"))
        folder_layout.addWidget(self.output_folder_edit)
        folder_layout.addWidget(self.output_folder_btn)
        folder_layout.addWidget(QLabel("参考图像:"))
        folder_layout.addWidget(self.ref_file_edit)
        folder_layout.addWidget(self.ref_file_btn)

        folder_group.setLayout(folder_layout)
        layout.addWidget(folder_group)

        # 参数设置
        param_group = QGroupBox("配准参数")
        param_layout = QFormLayout()

        self.smooth_sigma_spin = QDoubleSpinBox()
        self.smooth_sigma_spin.setRange(0.1, 10.0)
        self.smooth_sigma_spin.setValue(1.125)
        self.smooth_sigma_spin.setSingleStep(0.1)

        self.max_shift_spin = QDoubleSpinBox()
        self.max_shift_spin.setRange(0.01, 0.5)
        self.max_shift_spin.setValue(0.1)
        self.max_shift_spin.setSingleStep(0.01)

        self.smooth_sigma_time_spin = QDoubleSpinBox()
        self.smooth_sigma_time_spin.setRange(0.0, 10.0)
        self.smooth_sigma_time_spin.setValue(0.0)
        self.smooth_sigma_time_spin.setSingleStep(0.1)

        self.ref_frames_spin = QSpinBox()
        self.ref_frames_spin.setRange(0, 10000)
        self.ref_frames_spin.setValue(300)
        self.ref_frames_spin.setSingleStep(100)

        param_layout.addRow("空间平滑σ:", self.smooth_sigma_spin)
        param_layout.addRow("最大位移比例:", self.max_shift_spin)
        param_layout.addRow("时间平滑σ:", self.smooth_sigma_time_spin)
        param_layout.addRow("参考帧数:", self.ref_frames_spin)

        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # 控制按钮
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始批量配准")
        self.start_btn.clicked.connect(self.start_batch_registration)

        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self.stop_batch_registration)
        self.stop_btn.setEnabled(False)

        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setFormat("%v/%m 文件 - %p%")
        layout.addWidget(self.progress_bar)

        # 日志显示
        log_group = QGroupBox("处理日志")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        # 状态标签
        self.status_label = QLabel("就绪")
        layout.addWidget(self.status_label)

    def select_input_folder(self):
        """选择输入文件夹"""
        folder_path = QFileDialog.getExistingDirectory(self, "选择输入文件夹")
        if folder_path:
            self.input_folder_edit.setText(folder_path)

    def select_output_folder(self):
        """选择输出文件夹"""
        folder_path = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if folder_path:
            self.output_folder_edit.setText(folder_path)

    def select_ref_file(self):
        """选择参考文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择参考图像", "", "TIFF Files (*.tif *.tiff)"
        )
        if file_path:
            self.ref_file_edit.setText(file_path)

    def start_batch_registration(self):
        """开始批量配准"""
        input_folder = self.input_folder_edit.text()
        output_folder = self.output_folder_edit.text()

        if not input_folder or not output_folder:
            QMessageBox.warning(self, "错误", "请选择输入和输出文件夹")
            return

        if not Path(input_folder).exists():
            QMessageBox.warning(self, "错误", "输入文件夹不存在")
            return

        # 获取所有TIFF文件
        tiff_files = list(Path(input_folder).glob("*.tif")) + list(
            Path(input_folder).glob("*.tiff")
        )
        if not tiff_files:
            QMessageBox.warning(self, "错误", "输入文件夹中没有TIFF文件")
            return

        # 更新配准参数
        self.registrator.smooth_sigma = self.smooth_sigma_spin.value()
        self.registrator.maxregshift = self.max_shift_spin.value()
        self.registrator.smooth_sigma_time = self.smooth_sigma_time_spin.value()

        # 禁用按钮
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setMaximum(len(tiff_files))
        self.progress_bar.setValue(0)

        # 开始批量处理
        self.batch_files = tiff_files
        self.current_file_index = 0
        self.process_next_file()

    def stop_batch_registration(self):
        """停止批量配准"""
        self.stop_btn.setEnabled(False)
        self.status_label.setText("已停止")
        if hasattr(self, "reg_thread"):
            self.reg_thread.terminate()

    def process_next_file(self):
        """处理下一个文件"""
        if self.current_file_index >= len(self.batch_files):
            self.batch_registration_finished()
            return

        input_file = self.batch_files[self.current_file_index]
        output_file = (
            Path(self.output_folder_edit.text())
            / f"{input_file.name}_reg.tif"
        )

        # 更新状态
        self.status_label.setText(f"处理: {input_file.name}")
        self.log_text.append(f"开始处理: {input_file.name}")

        # 创建配准线程
        ref_image_path = (
            self.ref_file_edit.text() if self.ref_file_edit.text() else None
        )
        self.reg_thread = RegistrationThread(
            str(input_file), str(output_file), self.registrator, ref_image_path
        )
        self.reg_thread.finished.connect(self.file_registration_finished)
        self.reg_thread.error.connect(self.file_registration_error)
        self.reg_thread.start()

    def file_registration_finished(self, info: dict):
        """单个文件配准完成"""
        self.log_text.append(
            f"完成 "
        )

        self.current_file_index += 1
        self.progress_bar.setValue(self.current_file_index)

        # 处理下一个文件
        QTimer.singleShot(100, self.process_next_file)

    def file_registration_error(self, error_msg: str):
        """单个文件配准错误"""
        current_file = self.batch_files[self.current_file_index]
        self.log_text.append(f"错误处理 {current_file.name}: {error_msg}")

        self.current_file_index += 1
        self.progress_bar.setValue(self.current_file_index)

        # 继续处理下一个文件
        QTimer.singleShot(100, self.process_next_file)

    def batch_registration_finished(self):
        """批量配准完成"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText(f"批量配准完成: {len(self.batch_files)} 个文件")
        self.log_text.append("批量配准完成!")
        QMessageBox.information(
            self, "完成", f"批量配准完成! 共处理 {len(self.batch_files)} 个文件"
        )


class CalciumAlignerMainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("钙成像配准软件")
        self.setGeometry(100, 100, 1200, 800)
        self.init_ui()

    def init_ui(self):
        """初始化UI"""
        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建布局
        layout = QVBoxLayout(central_widget)

        # 创建标签页
        self.tab_widget = QTabWidget()

        # 单个配准标签页
        self.single_tab = SingleRegistrationTab()
        self.tab_widget.addTab(self.single_tab, "单个配准")

        # 批量配准标签页
        self.batch_tab = BatchRegistrationTab()
        self.tab_widget.addTab(self.batch_tab, "批量配准")

        layout.addWidget(self.tab_widget)

        # 菜单栏
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu("文件")

        exit_action = file_menu.addAction("退出")
        exit_action.triggered.connect(self.close)

        # 帮助菜单
        help_menu = menubar.addMenu("帮助")

        about_action = help_menu.addAction("关于")
        about_action.triggered.connect(self.show_about)

    def show_about(self):
        """显示关于对话框"""
        QMessageBox.about(
            self,
            "关于钙成像配准软件",
            "钙成像配准软件 v1.0\n\n"
            "基于 Suite2p 配准算法\n"
            "使用 PySide6 构建图形界面\n\n"
            "功能特点:\n"
            "- 单个文件配准\n"
            "- 批量文件配准\n"
            "- 自定义参考图像\n"
            "- 实时预览和参数调整",
        )


def main():
    """主函数"""
    app = QApplication(sys.argv)

    # 设置应用程序样式
    app.setStyle("Fusion")

    # 创建并显示主窗口
    window = CalciumAlignerMainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
