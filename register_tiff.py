#!/usr/bin/env python3
"""
TIFF 配准工具

使用方法:
    python register_tiff.py input.tif output.tif [options]

示例:
    # 基本使用
    python register_tiff.py raw_data.tif registered_data.tif
    
    # 指定参考帧数
    python register_tiff.py raw_data.tif registered_data.tif --ref-frames 300
    
    # 调整配准参数
    python register_tiff.py raw_data.tif registered_data.tif --smooth-sigma 1.5 --max-shift 0.15
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from suite2p_registration import Suite2PRegistration

def setup_logging(verbose: bool = False):
    """设置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

def main():
    parser = argparse.ArgumentParser(
        description='Suite2p TIFF 配准工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('input', help='输入TIFF文件路径')
    parser.add_argument('output', help='输出TIFF文件路径')
    parser.add_argument(
        "--ref-frames",
        type=int,
        default=300,
        help="用于计算参考图像的帧数 (默认: 300帧)",
    )
    parser.add_argument('--smooth-sigma', type=float, default=1.125, help='空间平滑标准差 (默认: 1.125)')
    parser.add_argument('--max-shift', type=float, default=0.1, help='最大位移比例 (默认: 0.1)')
    parser.add_argument('--smooth-sigma-time', type=float, default=0, help='时间平滑标准差 (默认: 0)')
    parser.add_argument('--info-path', help='保存配准信息JSON文件路径')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细日志')

    args = parser.parse_args()
    setup_logging(args.verbose)

    # 检查输入文件
    if not Path(args.input).exists():
        print(f"错误: 输入文件不存在: {args.input}", file=sys.stderr)
        sys.exit(1)

    try:
        # 创建配准器
        registrator = Suite2PRegistration(
            smooth_sigma=args.smooth_sigma,
            maxregshift=args.max_shift,
            smooth_sigma_time=args.smooth_sigma_time
        )

        # 处理TIFF文件
        info = registrator.process_tiff_stack(
            args.input, 
            args.output, 
            ref_frames=args.ref_frames
        )

        # 保存配准信息
        info_path = args.info_path or args.output.replace('.tif', '_reg_info.json')
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)

        print(f"配准完成!")
        print(f"输出文件: {args.output}")
        print(f"参考图像: {args.output.replace('.tif', '_ref.tif')}")
        print(f"配准信息: {info_path}")
        print(f"平均位移: Y={info['mean_shift_y']:.2f}, X={info['mean_shift_x']:.2f}")
        print(f"平均相关性: {info['mean_correlation']:.3f}")

    except Exception as e:
        print(f"处理失败: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
