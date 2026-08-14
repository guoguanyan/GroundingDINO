import os
import cv2
import random
import numpy as np
import torch
from typing import List, Dict
from groundingdino.util.inference import load_model, load_image, predict
from PIL import Image, ExifTags


class GroundingDINOToYOLO:
    def __init__(self, config_path: str, ckpt_path: str, device: str = "cuda"):
        """
        初始化 GroundingDINO 模型及全局配置。
        Args:
            config_path: GroundingDINO 配置文件路径
            ckpt_path:   模型权重文件路径
            device:      推理设备，'cuda' 或 'cpu'
        """
        print(f"⏳ 正在加载 GroundingDINO 模型...")
        self.model = load_model(config_path, ckpt_path, device=device)
        self.device = device
        # 默认阈值，可在 interactive_preview 中通过滑动条动态调整
        self.box_threshold = 0.30
        self.text_threshold = 0.25
        # ⭐ 全局类别颜色映射，避免每帧重建导致同一类别颜色跳变
        self.class_color_map = {}
        print(f"✅ 模型加载完成 | 设备: {device}")

    def _build_color_map(self, class_names: List[str]):
        """
        根据类别名称列表构建固定的 HSV 颜色映射。
        内部对类名排序，保证跨运行、跨调用入口的颜色分配确定性。
        Args:
            class_names: 类别名称列表，如 ['person', 'drone']
        """
        self.class_color_map = {}
        unique_sorted = sorted(set(class_names))
        n = max(len(unique_sorted), 1)
        for i, name in enumerate(unique_sorted):
            # 在 HSV 色环上均匀分布色调，饱和度和明度固定以保证可读性
            hue = int((i * 180 / n) % 180)
            hsv_color = np.uint8([[[hue, 255, 200]]])
            bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
            self.class_color_map[name] = tuple(int(c) for c in bgr_color.tolist())

    def _predict(self, image_path: str, text_prompt: str):
        """
        对单张图片执行 GroundingDINO 推理，返回原图绝对坐标 xyxy 格式的检测结果。
        包含 EXIF 旋转修正、自定义 transform 管线、坐标空间还原三个关键步骤。
        Args:
            image_path:  图片文件路径
            text_prompt: GroundingDINO 文本提示，如 'person . drone'
        Returns:
            image_bgr:       BGR 格式原图（已修正旋转），用于后续绘制
            abs_boxes_tensor: 原图绝对坐标 xyxy 格式的 bbox tensor (N,4)
            logits:          各检测框的置信度 tensor (N,)
            phrases:         各检测框对应的类别名称列表
            orig_size:       原图尺寸元组 (width, height)
        """
        # ⭐ 1. PIL 读取并修正 EXIF 旋转
        #     手机/无人机拍摄的图片常含 Orientation 标记，
        #     OpenCV 直接读取会忽略该标记导致图像方向错误、标注框偏移
        pil_img = Image.open(image_path)
        try:
            exif = pil_img._getexif()
            if exif:
                for tag, value in exif.items():
                    if ExifTags.TAGS.get(tag) == 'Orientation':
                        if value == 3:
                            pil_img = pil_img.rotate(180, expand=True)
                        elif value == 6:
                            pil_img = pil_img.rotate(270, expand=True)
                        elif value == 8:
                            pil_img = pil_img.rotate(90, expand=True)
        except (AttributeError, KeyError):
            pass

        # ⭐ 2. 旋转后重新获取尺寸，避免使用旋转前的错误尺寸
        orig_w, orig_h = pil_img.size

        # ⭐ 3. 转 BGR 用于后续 OpenCV 绘制（基于已旋转的正确图像）
        image_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

        # ⭐ 4. 对修正 EXIF 后的 PIL 图手动执行 GroundingDINO 标准 transform
        #     不能直接用 load_image，因为它会重新读盘并丢失 EXIF 旋转修正
        #     这里复用 GroundingDINO 官方预处理管线保证输入分布一致
        from groundingdino.datasets.transforms import Compose, RandomResize, ToTensor, Normalize
        transform = Compose([
            RandomResize([800], max_size=1333),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_tensor, _ = transform(pil_img, None)  # 返回 [C,H,W]，target=None

        # ⭐ 5. 执行推理，predict 内部自动处理 3D tensor 的 batch 维度
        boxes, logits, phrases = predict(
            model=self.model,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device,
        )

        # ⭐ 6. 将模型输出空间 (0~1) 的 cxcywh 转换为原图绝对坐标 xyxy
        #     predict 输出的 boxes 是相对于模型输入 tensor 尺寸的归一化 cxcywh，
        #     必须乘以原图尺寸才能得到与原图像素对齐的绝对坐标，
        #     这是保证预览框与 YOLO 标签位置一致的核心步骤
        abs_boxes = []
        for box in boxes:
            cx_norm, cy_norm, bw_norm, bh_norm = box.tolist()
            x1 = (cx_norm - bw_norm / 2) * orig_w
            y1 = (cy_norm - bh_norm / 2) * orig_h
            x2 = (cx_norm + bw_norm / 2) * orig_w
            y2 = (cy_norm + bh_norm / 2) * orig_h
            abs_boxes.append([x1, y1, x2, y2])

        # ⭐ 7. 构造安全的空 tensor，避免无检测结果时下游维度/设备不匹配
        if abs_boxes:
            abs_boxes_tensor = torch.tensor(abs_boxes, dtype=torch.float32, device=self.device)
        else:
            abs_boxes_tensor = torch.empty((0, 4), dtype=torch.float32, device=self.device)

        return image_bgr, abs_boxes_tensor, logits, phrases, (orig_w, orig_h)

    def _annotate_frame(self, image_bgr, boxes, logits, phrases, orig_size):
        """
        在原图上绘制检测框、置信度和类别标签。
        使用全局颜色映射保证同一类别在所有帧中颜色一致。
        Args:
            image_bgr: BGR 格式原图
            boxes:     原图绝对坐标 xyxy tensor (N,4)
            logits:    置信度 tensor (N,)
            phrases:   类别名称列表
            orig_size: 原图尺寸 (width, height)
        Returns:
            result: 绘制完成的 BGR 图像副本
        """
        orig_w, orig_h = orig_size
        result = image_bgr.copy()

        for box, logit, phrase in zip(boxes, logits, phrases):
            # boxes 已是原图绝对坐标 xyxy，直接取整并做边界保护
            x1, y1, x2, y2 = box.tolist()
            x1, y1 = int(x1), int(y1)
            x2, y2 = int(x2), int(y2)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)

            conf = logit.item()
            label = f"{phrase} {conf:.2f}"

            # ⭐ 从全局固定映射取色，未知类别回退到红色 (0,0,255)
            color = self.class_color_map.get(phrase, (0, 0, 255))

            # 线宽和字号根据图像分辨率自适应，避免小图过粗或大图过细
            thickness = max(2, int(min(orig_w, orig_h) / 500))
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

            font_scale = max(0.4, min(orig_w, orig_h) / 1000)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            # 标签优先画在框上方，空间不足时画在框内顶部
            text_y = y1 - 4 if y1 - th - 8 >= 0 else y1 + th + 5
            bg_top = y1 - th - 8 if y1 - th - 8 >= 0 else y1
            bg_bottom = y1 if y1 - th - 8 >= 0 else y1 + th + 8

            cv2.rectangle(result, (x1, bg_top), (x1 + tw, bg_bottom), color, -1)
            cv2.putText(result, label, (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

        return result

    @staticmethod
    def _boxes_to_yolo(boxes: torch.Tensor, phrases: List[str],
                       class_map: Dict[str, int], img_w: int, img_h: int) -> List[str]:
        """
        将原图绝对坐标 xyxy 转换为 YOLO 归一化格式字符串。
        Args:
            boxes:     原图绝对坐标 xyxy tensor (N,4)
            phrases:   类别名称列表
            class_map: 类别名→ID 映射字典
            img_w:     原图宽度（像素）
            img_h:     原图高度（像素）
        Returns:
            lines: YOLO 格式字符串列表，每行格式为 '{cls_id} {cx} {cy} {bw} {bh}'
        """
        # ⭐ 安全校验：防止损坏图片尺寸为 0 导致除零异常
        if img_w <= 0 or img_h <= 0:
            return []

        lines = []
        for box, phrase in zip(boxes, phrases):
            cls_name = phrase.strip().lower()
            if cls_name not in class_map:
                continue
            cls_id = class_map[cls_name]
            x1, y1, x2, y2 = box.tolist()
            # 绝对坐标 → 归一化中心点 + 归一化宽高
            cx = ((x1 + x2) / 2.0) / img_w
            cy = ((y1 + y2) / 2.0) / img_h
            bw = (x2 - x1) / img_w
            bh = (y2 - y1) / img_h
            # 裁剪到 [0,1] 范围，防止浮点误差导致越界
            cx, cy = max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy))
            bw, bh = max(0.0, min(1.0, bw)), max(0.0, min(1.0, bh))
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        return lines

    def interactive_preview(self, image_paths: List[str], text_prompt: str, sample_size: int = 10):
        """
        交互式预览：随机抽样图片，通过滑动条实时调参，确认阈值后返回。
        支持 A/D 键切换图片、Enter 确认、ESC 取消，兼容 Windows/Linux 键码。
        Args:
            image_paths: 候选图片路径列表
            text_prompt: GroundingDINO 文本提示
            sample_size: 抽样数量上限
        Returns:
            confirmed: True 表示用户确认阈值，False 表示取消
        """
        # ⭐ 用 prompt 中的类别名初始化全局颜色映射
        prompt_classes = [p.strip() for p in text_prompt.split('.') if p.strip()]
        self._build_color_map(prompt_classes)

        sample_size = min(sample_size, len(image_paths))
        sampled = random.sample(image_paths, sample_size)
        print(f"\n🎯 交互预览 ({sample_size} 张) | Prompt: '{text_prompt}'")
        print(f"   拖动滑动条调参 | ← → 切图 | Enter 确认 | ESC 取消")

        win_name = "GDINO Preview (ENTER=confirm)"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 1280, 720)
        cv2.createTrackbar("Box x100", win_name, int(self.box_threshold * 100), 100, lambda x: None)
        cv2.createTrackbar("Text x100", win_name, int(self.text_threshold * 100), 100, lambda x: None)

        idx, confirmed = 0, False
        while True:
            # 实时读取滑动条值，变化时更新阈值触发重新推理
            new_box = cv2.getTrackbarPos("Box x100", win_name) / 100.0
            new_text = cv2.getTrackbarPos("Text x100", win_name) / 100.0
            if new_box != self.box_threshold or new_text != self.text_threshold:
                self.box_threshold, self.text_threshold = new_box, new_text

            img_bgr, boxes, logits, phrases, orig_size = self._predict(sampled[idx], text_prompt)
            frame = self._annotate_frame(img_bgr, boxes, logits, phrases, orig_size)

            info = f"[{idx+1}/{sample_size}] Box:{self.box_threshold:.2f} Text:{self.text_threshold:.2f} Dets:{len(boxes)}"
            cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "A/D:prev/next  ENTER:confirm  ESC:quit",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imshow(win_name, frame)

            # ⭐ 用 0xFFFF 保留完整键码，兼容 Linux 扩展键码
            key = cv2.waitKey(30) & 0xFFFF
            if key in (13, 10):             # Enter / Return
                confirmed = True
                break
            elif key == 27:                 # ESC
                break
            elif key in (ord('a'), ord('A')):  # A 键（全平台最可靠）
                idx = (idx - 1) % sample_size
            elif key in (ord('d'), ord('D')):  # D 键
                idx = (idx + 1) % sample_size
            elif key in (81, 65361, 2):     # Left Arrow: Windows(81) / Linux(65361) / 备用(2)
                idx = (idx - 1) % sample_size
            elif key in (83, 65363, 3):     # Right Arrow: Windows(83) / Linux(65363) / 备用(3)
                idx = (idx + 1) % sample_size

        cv2.destroyAllWindows()
        status = "✅ 阈值已确认" if confirmed else "⚠️ 预览取消"
        print(f"{status} → box={self.box_threshold:.2f}, text={self.text_threshold:.2f}")
        return confirmed

    def export_yolo(self, image_dir: str, output_dir: str, text_prompt: str,
                    class_map: Dict[str, int], extensions=None):
        """
        批量导出 YOLO 格式数据集：生成 images/、labels/ 目录及 classes.txt。
        同时输出带标注的可视化图片用于人工抽检。
        Args:
            image_dir:   原始图片目录
            output_dir:  YOLO 数据集输出根目录
            text_prompt: GroundingDINO 文本提示
            class_map:   类别名→ID 映射字典
            extensions:  支持的图片扩展名集合
        """
        if extensions is None:
            extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

        # ⭐ 用实际类别映射初始化全局颜色，保证导出可视化与预览颜色一致
        self._build_color_map(list(class_map.keys()))

        all_images = sorted([
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if os.path.splitext(f)[1].lower() in extensions
        ])

        out_img_dir = os.path.join(output_dir, "labeled_images")
        out_lbl_dir = os.path.join(output_dir, "labels")
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)

        # ⭐ 同步输出 classes.txt，按类别 ID 升序排列，行号即对应 YOLO 标签中的 cls_id
        classes_path = os.path.join(output_dir, "classes.txt")
        sorted_classes = sorted(class_map.items(), key=lambda x: x[1])
        with open(classes_path, "w", encoding="utf-8") as f:
            for class_name, _ in sorted_classes:
                f.write(f"{class_name}\n")
        print(f"✅ 已生成类别文件: {classes_path} ({len(sorted_classes)} 个类别)")

        total = len(all_images)
        total_dets = 0
        print(f"\n🚀 批量导出YOLO: {total} 张 | 类别映射: {class_map}")

        for i, img_path in enumerate(all_images, 1):
            img_bgr, boxes, logits, phrases, orig_size = self._predict(img_path, text_prompt)
            basename = os.path.splitext(os.path.basename(img_path))[0]
            img_h, img_w = img_bgr.shape[:2]

            # ⭐ 写 YOLO 标签文件，无检测结果时写入空文件而非空行
            yolo_lines = self._boxes_to_yolo(boxes, phrases, class_map, img_w, img_h)
            lbl_path = os.path.join(out_lbl_dir, f"{basename}.txt")
            with open(lbl_path, "w") as f:
                if yolo_lines:
                    f.write("\n".join(yolo_lines))

            # ⭐ 写可视化图片，BGR 格式由 cv2.imwrite 直接正确处理
            vis_frame = self._annotate_frame(img_bgr, boxes, logits, phrases, orig_size)
            cv2.imwrite(os.path.join(out_img_dir, os.path.basename(img_path)), vis_frame)

            total_dets += len(yolo_lines)
            if i % 10 == 0 or i == total:
                print(f"   进度: {i}/{total} | 累计标注: {total_dets}")

        print(f"\n✅ 导出完成! 总标注行数: {total_dets}")
        print(f"   图片: {os.path.abspath(out_img_dir)}")
        print(f"   标签: {os.path.abspath(out_lbl_dir)}")



# ==================== 主程序入口 ====================
if __name__ == "__main__":

    # # T模型（对小目标检测更好，但会有多余错误标注）
    # CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
    # CKPT_PATH   = "weights/groundingdino_swint_ogc.pth"
    
    # B模型
    CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinB_cfg.py"
    CKPT_PATH   = "weights/groundingdino_swinb_cogcoor.pth"

    # 需要处理的图片文件夹
    IMAGE_DIR   = "/home/cs/桌面/ggy_label_work/2472_实采_person/20260813/person_less/images"
    # 保存yolo标签位置 
    OUTPUT_DIR  = os.path.dirname(IMAGE_DIR)

    # 输入检测的类别
    # TEXT_PROMPT = 'person . drone'
    TEXT_PROMPT = 'person'

    # 输入类别标签映射，自动生成classes.txt
    CLASS_MAP = {
        "person": 0,
        "drone": 1,
        "balloon": 2,
        "target": 3
    }

    detector = GroundingDINOToYOLO(CONFIG_PATH, CKPT_PATH)
    
    img_list = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)
                if os.path.splitext(f)[1].lower() in {'.jpg','.jpeg','.png','.bmp','.webp'}]
    
    detector.interactive_preview(img_list, TEXT_PROMPT, sample_size=10)
    detector.export_yolo(IMAGE_DIR, OUTPUT_DIR, TEXT_PROMPT, CLASS_MAP)
