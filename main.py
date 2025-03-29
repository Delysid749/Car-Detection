import warnings

warnings.filterwarnings('ignore')

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import threading
import queue
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
from http import HTTPStatus
from functions import get_report

class FinalVehicleSpeedDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("车辆速度检测系统")

        # 模型初始化
        self.model = YOLO("runs/detect/train/weights/best.pt")  # 替换为你的模型路径
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            nms_max_overlap=1.0,
            max_cosine_distance=0.3,
            nn_budget=None,
            override_track_class=None,
            embedder="mobilenet",
            half=True,
            bgr=True,
            # embedder_gpu=True,
            embedder_model_name=None,
            embedder_wts=None,
            polygon=False,
            today=None
        )
        self.conf_threshold = 0.5

        # 视频处理
        self.cap = None
        self.video_path = ""
        self.is_playing = False
        self.frame_count = 0
        self.fps = 30

        # 透视变换
        self.perspective_points = []
        self.M = None
        self.Minv = None
        self.real_world_scale = 0.3  # 米/像素（需实际标定）

        # 双线计数系统
        self.up_line = []
        self.down_line = []
        self.counter_up = 0
        self.counter_down = 0
        self.line_offset = 6

        # 车辆跟踪
        self.track_history = {}
        self.speed_history = {}

        # GUI组件
        self.create_widgets()
        self.setup_bindings()
        self.image_queue = queue.Queue(maxsize=1)

    def create_widgets(self):
        # 控制面板
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        self.btn_open = ttk.Button(control_frame, text="打开视频", command=self.open_video)
        self.btn_open.pack(side=tk.LEFT, padx=5)

        self.btn_calibrate = ttk.Button(control_frame, text="校准透视变换",
                                        command=self.start_perspective_calibration)
        self.btn_calibrate.pack(side=tk.LEFT, padx=5)
        self.btn_calibrate.config(state=tk.DISABLED)

        self.btn_up_line = ttk.Button(control_frame, text="画上行线",
                                      command=self.start_drawing_up_line)
        self.btn_up_line.pack(side=tk.LEFT, padx=5)
        self.btn_up_line.config(state=tk.DISABLED)

        self.btn_down_line = ttk.Button(control_frame, text="画下行线",
                                        command=self.start_drawing_down_line)
        self.btn_down_line.pack(side=tk.LEFT, padx=5)
        self.btn_down_line.config(state=tk.DISABLED)

        # 置信度设置
        self.conf_frame = ttk.Frame(control_frame)
        self.conf_frame.pack(side=tk.LEFT, padx=10)
        ttk.Label(self.conf_frame, text="置信度:").pack(side=tk.LEFT)
        self.conf_entry = ttk.Entry(self.conf_frame, width=5)
        self.conf_entry.pack(side=tk.LEFT)
        self.conf_entry.insert(0, "0.5")
        ttk.Button(self.conf_frame, text="设置",
                   command=self.set_conf_threshold).pack(side=tk.LEFT, padx=5)

        self.btn_start = ttk.Button(control_frame, text="开始检测",
                                    command=self.toggle_detection)
        self.btn_start.pack(side=tk.LEFT, padx=5)
        self.btn_start.config(state=tk.DISABLED)

        # 视频画布
        self.canvas = tk.Canvas(self.root, bg='black', width=1020, height=500)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 状态栏
        self.status_bar = ttk.Label(self.root, text="就绪", anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def set_conf_threshold(self):
        try:
            conf = float(self.conf_entry.get())
            if 0 <= conf <= 1:
                self.conf_threshold = conf
                self.status_bar.config(text=f"置信度阈值已设置为: {conf:.2f}")
            else:
                self.status_bar.config(text="置信度需在0-1之间")
        except ValueError:
            self.status_bar.config(text="请输入有效数字")

    def setup_bindings(self):
        self.canvas.bind("<Button-1>", self.on_canvas_click)

    def open_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("视频文件", "*.mp4 *.avi")])
        if file_path:
            self.video_path = file_path
            self.btn_calibrate.config(state=tk.NORMAL)
            self.load_first_frame()

    def load_first_frame(self):
        cap = cv2.VideoCapture(self.video_path)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                self.fps = cap.get(cv2.CAP_PROP_FPS)
                frame = cv2.resize(frame, (1020, 500))
                self.first_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.show_frame(self.first_frame)
                self.status_bar.config(text="请先进行透视变换校准（点击4个点，顺时针从左上角开始）")
            cap.release()

    def start_perspective_calibration(self):
        self.perspective_points = []
        self.drawing_mode = 'perspective'
        self.btn_calibrate.config(state=tk.DISABLED)
        self.status_bar.config(text="请点击4个点定义道路区域（顺时针从左上角开始）")

    def start_drawing_up_line(self):
        self.drawing_mode = 'up_line'
        self.temp_line_points = []
        self.status_bar.config(text="正在绘制上行线（蓝色），请点击两个点")

    def start_drawing_down_line(self):
        self.drawing_mode = 'down_line'
        self.temp_line_points = []
        self.status_bar.config(text="正在绘制下行线（红色），请点击两个点")

    def on_canvas_click(self, event):
        if self.drawing_mode == 'perspective' and len(self.perspective_points) < 4:
            self.perspective_points.append((event.x, event.y))
            self.canvas.create_oval(event.x - 3, event.y - 3, event.x + 3, event.y + 3,
                                    fill='yellow', tags='calib_point')

            if len(self.perspective_points) == 4:
                self.calculate_perspective_matrix()
                self.btn_up_line.config(state=tk.NORMAL)
                self.btn_down_line.config(state=tk.NORMAL)
                self.btn_start.config(state=tk.NORMAL)
                self.drawing_mode = None

        elif self.drawing_mode in ('up_line', 'down_line'):
            self.temp_line_points.append((event.x, event.y))
            color = 'blue' if self.drawing_mode == 'up_line' else 'red'

            if len(self.temp_line_points) == 1:
                self.canvas.create_oval(event.x - 3, event.y - 3, event.x + 3, event.y + 3,
                                        fill=color, tags='temp_line')
            elif len(self.temp_line_points) == 2:
                if self.drawing_mode == 'up_line':
                    self.up_line = self.temp_line_points
                    self.canvas.create_line(self.temp_line_points, fill=color, width=2, tags='up_line')
                else:
                    self.down_line = self.temp_line_points
                    self.canvas.create_line(self.temp_line_points, fill=color, width=2, tags='down_line')

                self.drawing_mode = None
                self.check_lines_status()

    def check_lines_status(self):
        if self.up_line and self.down_line:
            self.status_bar.config(text="校准完成，可以开始检测")
        else:
            line_type = "上行" if not self.up_line else "下行"
            self.status_bar.config(text=f"请继续绘制{line_type}线")

    def calculate_perspective_matrix(self):
        src = np.array(self.perspective_points, dtype=np.float32)
        width = int(max(np.linalg.norm(src[0] - src[1]), np.linalg.norm(src[2] - src[3])))
        height = int(max(np.linalg.norm(src[1] - src[2]), np.linalg.norm(src[3] - src[0])))

        dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)
        self.status_bar.config(text=f"透视矩阵计算完成 区域尺寸：{width}x{height}像素")

    def transform_point(self, point):
        if self.M is not None:
            pt = np.array([[point]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(pt, self.M)
            return (transformed[0][0][0], transformed[0][0][1])
        return point

    def toggle_detection(self):
        if not self.is_playing:
            if not (self.up_line and self.down_line):
                self.status_bar.config(text="请先完成所有校准步骤")
                return
            self.is_playing = True
            self.btn_start.config(text="停止检测")
            self.start_detection()
        else:
            self.is_playing = False
            self.btn_start.config(text="开始检测")

    def start_detection(self):
        self.detection_thread = threading.Thread(target=self.process_video)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        self.update_frame()

    def process_video(self):
        self.cap = cv2.VideoCapture(self.video_path)
        while self.is_playing and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (1020, 500))
            results = self.model.track(
                frame,
                persist=True,
                verbose=False,
                conf=self.conf_threshold,
                classes=[0]  # 只检测车辆类别（根据COCO数据集）
            )

            # 获取检测结果
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []

            annotated_frame = frame.copy()

            # 绘制检测框和跟踪信息
            for box, conf, track_id in zip(boxes, confs, track_ids):
                x1, y1, x2, y2 = map(int, box)
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                # 绘制检测框
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame,
                            f"ID:{track_id} {conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2)

                # 更新轨迹历史
                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                self.track_history[track_id].append(center)

                # 保留最近10个位置
                if len(self.track_history[track_id]) > 10:
                    self.track_history[track_id].pop(0)

                # 鸟瞰图速度计算
                if len(self.track_history[track_id]) >= 2:
                    pt1 = self.transform_point(self.track_history[track_id][-2])
                    pt2 = self.transform_point(self.track_history[track_id][-1])

                    distance_px = np.linalg.norm(np.array(pt2) - np.array(pt1))
                    distance_m = distance_px * self.real_world_scale
                    time_diff = 1 / self.fps  # 秒每帧
                    speed = distance_m / time_diff * 3.6  # km/h
                    if speed > 140:
                        speed = 140
                    self.speed_history[track_id] = speed

                    cv2.putText(annotated_frame, f"{speed:.1f} km/h",
                                (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 255), 2)

                # 绘制运动轨迹
                for i in range(1, len(self.track_history[track_id])):
                    cv2.line(annotated_frame,
                             self.track_history[track_id][i - 1],
                             self.track_history[track_id][i],
                             (0, 255, 0), 2)

                # 检测线穿越
                self.check_line_crossing(track_id, center, annotated_frame)

            # 绘制检测线
            if self.up_line:
                cv2.line(annotated_frame, self.up_line[0], self.up_line[1], (255, 0, 0), 2)
            if self.down_line:
                cv2.line(annotated_frame, self.down_line[0], self.down_line[1], (0, 0, 255), 2)

            # 显示统计信息
            cv2.rectangle(annotated_frame, (0, 0), (250, 90), (0, 255, 255), -1)
            cv2.putText(annotated_frame, f"down: {self.counter_down}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            cv2.putText(annotated_frame, f"up: {self.counter_up}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            if self.image_queue.empty():
                try:
                    self.image_queue.put(img, block=False)
                except queue.Full:
                    pass
        self.cap.release()
        self.is_playing = False
        # 保存视频的最后一帧为图片
        cv2.imwrite("output.jpg", img)

        msg_window = self.show_generating_report_message()
        self.root.after(10000, msg_window.destroy)

        get_report("请分析道路交通情况并给出建议。","output.jpg")
        os.system("start road_report.md")
        self.btn_start.config(text="开始检测")

    def show_generating_report_message(self):
        # 创建一个新的顶级窗口用于显示消息
        msg_window = tk.Toplevel(root)
        msg_window.title("提示")
        msg_window.geometry("250x100")  # 设置窗口大小

        # 在窗口中添加一个标签，用来显示提示信息
        label = tk.Label(msg_window, text="正在生成报告，请稍候...")
        label.pack(expand=True)  # expand=True 使标签居中显示

        # 禁用窗口的关闭按钮，如果需要的话
        msg_window.protocol("WM_DELETE_WINDOW", lambda: None)

        # 返回这个窗口对象，以便在外边控制它（可选）
        return msg_window
    def check_line_crossing(self, track_id, center, frame):
        # 检测上行线穿越
        if self.up_line:
            y1, y2 = self.up_line[0][1], self.up_line[1][1]
            min_y = min(y1, y2) - self.line_offset
            max_y = max(y1, y2) + self.line_offset
            if min_y <= center[1] <= max_y:
                if track_id not in self.track_history or len(self.track_history[track_id]) < 2:
                    return
                prev_y = self.track_history[track_id][-2][1]
                if prev_y > max_y and center[1] <= max_y:
                    self.counter_up += 1
                    cv2.putText(frame, "上行计数+1", (center[0] + 10, center[1] + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # 检测下行线穿越
        if self.down_line:
            y1, y2 = self.down_line[0][1], self.down_line[1][1]
            min_y = min(y1, y2) - self.line_offset
            max_y = max(y1, y2) + self.line_offset
            if min_y <= center[1] <= max_y:
                if track_id not in self.track_history or len(self.track_history[track_id]) < 2:
                    return
                prev_y = self.track_history[track_id][-2][1]
                if prev_y < min_y and center[1] >= min_y:
                    self.counter_down += 1
                    cv2.putText(frame, "下行计数+1", (center[0] + 10, center[1] + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    def show_frame(self, frame):
        img = Image.fromarray(frame)
        self.photo = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def update_frame(self):
        try:
            img = self.image_queue.get_nowait()
            self.show_frame(img)
        except queue.Empty:
            pass

        if self.is_playing:
            self.root.after(30, self.update_frame)


if __name__ == "__main__":
    root = tk.Tk()
    app = FinalVehicleSpeedDetectionApp(root)
    root.geometry("1020x600")
    root.mainloop()