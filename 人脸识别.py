import cv2
import os
import numpy as np
import time
import pickle
from tkinter import (Tk, Frame, Label, Button, messagebox, simpledialog, Toplevel)
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading

class FaceRecognitionSystem:
    def __init__(self, root):
        # 主窗口配置
        self.root = root
        self.root.title("人脸识别系统")
        self.root.geometry("800x600")
        self.root.resizable(False, False)
        
        # 摄像头参数
        self.cam_width = 640
        self.cam_height = 480
        self.cap = None
        self.running = False  # 识别/采集状态标记
        
        # 核心数据
        self.confidence_threshold = 55
        self.unique_names = []
        self.all_face_samples = []
        self.all_labels = []
        
        # 加载人脸检测模型
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
        )
        
        # 初始化识别器
        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create(
                radius=1, neighbors=8
            )
        except AttributeError:
            messagebox.showerror("错误", "请安装opencv-contrib-python: pip3 install opencv-contrib-python")
            self.root.destroy()
            return
        
        # 先创建界面，再加载数据
        self._create_ui()
        self.load_trained_data()

    def _create_ui(self):
        """创建图形界面"""
        # 1. 顶部摄像头显示区
        self.cam_frame = Frame(self.root, width=self.cam_width, height=self.cam_height, bg="black")
        self.cam_frame.pack(pady=10)
        
        self.cam_label = Label(self.cam_frame, bg="black")
        self.cam_label.pack(fill="both", expand=True)
        
        # 2. 中间功能按钮区
        self.btn_frame = Frame(self.root)
        self.btn_frame.pack(pady=10)
        
        self.btn_add = Button(self.btn_frame, text="添加用户", width=15, command=self._add_user)
        self.btn_add.grid(row=0, column=0, padx=10)
        
        self.btn_start = Button(self.btn_frame, text="开始识别", width=15, command=self._start_recognition)
        self.btn_start.grid(row=0, column=1, padx=10)
        
        self.btn_list = Button(self.btn_frame, text="查看用户", width=15, command=self._list_users)
        self.btn_list.grid(row=0, column=2, padx=10)
        
        self.btn_delete = Button(self.btn_frame, text="删除用户", width=15, command=self._delete_user)
        self.btn_delete.grid(row=0, column=3, padx=10)
        
        self.btn_threshold = Button(self.btn_frame, text="设置阈值", width=15, command=self._set_threshold)
        self.btn_threshold.grid(row=0, column=4, padx=10)
        
        # 3. 底部状态提示区
        self.status_frame = Frame(self.root)
        self.status_frame.pack(fill="x", padx=20, pady=10)
        
        self.status_label = Label(self.status_frame, text="系统就绪", anchor="w")
        self.status_label.pack(fill="x")

    def _get_font(self, size):
        """获取中文字体"""
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", size)
        except:
            try:
                return ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", size)
            except:
                return ImageFont.load_default()

    def _draw_text(self, img, text, pos, size, color=(0, 255, 0)):
        """绘制中文文本"""
        if not text:
            return img
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text(pos, text, font=self._get_font(size), fill=(color[2], color[1], color[0]))
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def load_trained_data(self):
        """加载训练数据"""
        try:
            if os.path.exists('face_trained.yml') and os.path.exists('unique_names.txt'):
                self.recognizer.read('face_trained.yml')
                with open('unique_names.txt', 'r', encoding='utf-8') as f:
                    self.unique_names = [line.strip() for line in f.readlines() if line.strip()]
                
                if os.path.exists('all_face_samples.pkl') and os.path.exists('all_labels.pkl'):
                    with open('all_face_samples.pkl', 'rb') as f:
                        self.all_face_samples = pickle.load(f)
                    with open('all_labels.pkl', 'rb') as f:
                        self.all_labels = pickle.load(f)
                self.status_label.config(text=f"数据加载成功，共 {len(self.unique_names)} 个用户")
                return True
        except Exception as e:
            self.status_label.config(text=f"数据加载失败: {str(e)}")
        return False

    def _open_camera(self):
        """修复：使用更稳定的DirectShow后端（Windows专用）"""
        if self.cap is not None and self.cap.isOpened():
            return True
        
        # 强制使用DirectShow后端（解决Windows摄像头兼容性问题）
        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)
                # 测试读取一帧，确保正常
                ret, _ = self.cap.read()
                if ret:
                    self.status_label.config(text="摄像头已打开")
                    return True
            # 打开失败则释放
            self.cap.release()
            self.cap = None
        except Exception as e:
            self.status_label.config(f"摄像头打开失败: {str(e)}")
        
        messagebox.showerror("错误", "无法打开摄像头，请检查：\n1. 摄像头未被其他应用占用\n2. 摄像头驱动正常")
        return False

    def _add_user(self):
        """添加用户（增加异常捕获）"""
        if self.running:
            messagebox.showinfo("提示", "请先停止当前识别/采集任务")
            return
        
        name = simpledialog.askstring("输入", "请输入用户名:")
        if not name or name.strip() == "":
            messagebox.showwarning("警告", "用户名不能为空")
            return
        
        num_samples = 30
        self.status_label.config(text=f"正在收集 {name} 的人脸数据（{num_samples}个样本）")
        
        if not self._open_camera():
            return
        
        count = 0
        new_samples = []
        target_size = (150, 150)
        self.running = True
        
        def collect_thread():
            nonlocal count, new_samples
            try:  # 增加异常捕获
                while count < num_samples and self.running:
                    # 读取帧时捕获OpenCV异常
                    try:
                        ret, frame = self.cap.read()
                    except Exception as e:
                        self.status_label.config(text=f"摄像头读取错误: {str(e)}")
                        break
                    
                    if not ret:
                        self.status_label.config(text="获取视频帧失败")
                        break
                    
                    frame = cv2.resize(frame, (self.cam_width, self.cam_height))
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                    
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        frame = self._draw_text(frame, f"进度: {count}/{num_samples}", (10, 30), 20)
                        
                        face_roi = gray[y:y+h, x:x+w]
                        face_roi = cv2.resize(face_roi, target_size)
                        new_samples.append(face_roi)
                        count += 1
                    
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)
                    img_tk = ImageTk.PhotoImage(image=img_pil)
                    self.cam_label.config(image=img_tk)
                    self.cam_label.image = img_tk
                    
                    time.sleep(0.1)
            finally:  # 确保资源释放
                self.running = False
                if len(new_samples) > 0:
                    if name in self.unique_names:
                        user_label = self.unique_names.index(name)
                    else:
                        user_label = len(self.unique_names)
                        self.unique_names.append(name)
                    
                    self.all_face_samples.extend(new_samples)
                    self.all_labels.extend([user_label] * len(new_samples))
                    
                    self.status_label.config(text="正在训练模型...")
                    self.recognizer.train(self.all_face_samples, np.array(self.all_labels))
                    
                    self.recognizer.write('face_trained.yml')
                    with open('unique_names.txt', 'w', encoding='utf-8') as f:
                        for n in self.unique_names:
                            f.write(n + '\n')
                    with open('all_face_samples.pkl', 'wb') as f:
                        pickle.dump(self.all_face_samples, f)
                    with open('all_labels.pkl', 'wb') as f:
                        pickle.dump(self.all_labels, f)
                    
                    self.status_label.config(text=f"成功添加用户: {name}（{len(new_samples)}个样本）")
                else:
                    self.status_label.config(text="未收集到有效样本")
                
                if self.cap:
                    self.cap.release()
                    self.cap = None
                self.cam_label.config(image="")
        
        threading.Thread(target=collect_thread, daemon=True).start()

    def _start_recognition(self):
        """开始识别（重点修复异常问题）"""
        if self.running:
            self.running = False
            self.btn_start.config(text="开始识别")
            self.status_label.config(text="识别已停止")
            if self.cap:
                self.cap.release()
                self.cap = None
            self.cam_label.config(image="")
            return
        
        if not self.unique_names:
            messagebox.showinfo("提示", "请先添加用户数据")
            return
        
        if not self._open_camera():
            return
        
        self.running = True
        self.btn_start.config(text="停止识别")
        self.status_label.config(text="开始人脸识别...")
        frame_counter = 0
        cached_faces = []
        cache_valid_frames = 2
        current_cache_frame = 0

        def recognition_thread():
            nonlocal frame_counter, cached_faces, current_cache_frame
            try:  # 捕获线程中所有异常
                while self.running:
                    # 读取帧时捕获OpenCV底层异常
                    try:
                        ret, frame = self.cap.read()
                    except Exception as e:
                        self.status_label.config(text=f"摄像头读取错误: {str(e)}")
                        time.sleep(1)
                        break  # 出错后退出循环
                    
                    if not ret:
                        self.status_label.config(text="获取视频帧失败，重试...")
                        time.sleep(0.5)
                        continue
                    
                    frame = cv2.resize(frame, (self.cam_width, self.cam_height))
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame_counter += 1
                    current_faces = []

                    if frame_counter % 4 == 0:
                        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                        current_faces = [(x, y, w, h) for (x, y, w, h) in faces if w > 50]
                        cached_faces = current_faces
                        current_cache_frame = 0
                    else:
                        current_cache_frame += 1
                        if current_cache_frame <= cache_valid_frames:
                            current_faces = cached_faces
                        else:
                            current_faces = []

                    for (x, y, w, h) in current_faces:
                        face_roi = gray[y:y+h, x:x+w]
                        name = "未知"
                        confidence_text = "无数据"
                        
                        try:
                            label, confidence = self.recognizer.predict(face_roi)
                            similarity = 100 - confidence
                            
                            if 0 <= label < len(self.unique_names) and similarity >= self.confidence_threshold:
                                name = self.unique_names[label]
                                confidence_text = f"相似度: {int(similarity)}%"
                            else:
                                confidence_text = f"不匹配: {int(similarity)}%"
                        except Exception as e:
                            confidence_text = "识别出错"
                        
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        frame = self._draw_text(frame, name, (x, y-30), 25)
                        frame = self._draw_text(frame, confidence_text, (x, y-5), 18)
                    
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)
                    img_tk = ImageTk.PhotoImage(image=img_pil)
                    self.cam_label.config(image=img_tk)
                    self.cam_label.image = img_tk
                    
                    time.sleep(0.08)
            finally:  # 确保资源释放
                self.running = False
                self.btn_start.config(text="开始识别")
                if self.cap:
                    self.cap.release()
                    self.cap = None
                self.cam_label.config(image="")
                self.status_label.config(text="识别已停止（异常处理）")

        threading.Thread(target=recognition_thread, daemon=True).start()

    def _list_users(self):
        """查看用户列表"""
        if not self.unique_names:
            messagebox.showinfo("提示", "无存储的用户数据")
            return
        
        top = Toplevel(self.root)
        top.title("用户列表")
        top.geometry("400x300")
        top.transient(self.root)
        top.grab_set()
        
        tree = ttk.Treeview(top, columns=["id", "name", "samples"], show="headings")
        tree.heading("id", text="编号")
        tree.heading("name", text="用户名")
        tree.heading("samples", text="样本数")
        
        tree.column("id", width=50, anchor="center")
        tree.column("name", width=200, anchor="center")
        tree.column("samples", width=100, anchor="center")
        
        user_samples = {name: 0 for name in self.unique_names}
        for label in self.all_labels:
            user_samples[self.unique_names[label]] += 1
        
        for i, name in enumerate(self.unique_names, 1):
            tree.insert("", "end", values=[i, name, user_samples[name]])
        
        tree.pack(fill="both", expand=True, padx=10, pady=10)
        Button(top, text="关闭", command=top.destroy).pack(pady=10)

    def _delete_user(self):
        """删除用户"""
        if not self.unique_names:
            messagebox.showinfo("提示", "无存储的用户数据")
            return
        
        names = [f"{i+1}. {name}" for i, name in enumerate(self.unique_names)]
        name_str = "\n".join(names)
        choice = simpledialog.askstring("删除用户", f"请输入要删除的用户编号:\n{name_str}\n(输入0取消)")
        
        if not choice or choice == "0":
            return
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(self.unique_names):
                target_name = self.unique_names[idx]
                if messagebox.askyesno("确认", f"确定删除 {target_name} 吗?"):
                    target_label = idx
                    new_samples = []
                    new_labels = []
                    for sample, label in zip(self.all_face_samples, self.all_labels):
                        if label != target_label:
                            new_samples.append(sample)
                            new_labels.append(label if label < target_label else label - 1)
                    
                    self.unique_names = [n for n in self.unique_names if n != target_name]
                    self.all_face_samples = new_samples
                    self.all_labels = new_labels
                    
                    if self.unique_names:
                        self.recognizer.train(self.all_face_samples, np.array(self.all_labels))
                        self.recognizer.write('face_trained.yml')
                        with open('all_face_samples.pkl', 'wb') as f:
                            pickle.dump(self.all_face_samples, f)
                        with open('all_labels.pkl', 'wb') as f:
                            pickle.dump(self.all_labels, f)
                    else:
                        for f in ['face_trained.yml', 'unique_names.txt', 
                                 'all_face_samples.pkl', 'all_labels.pkl']:
                            if os.path.exists(f):
                                os.remove(f)
                    
                    with open('unique_names.txt', 'w', encoding='utf-8') as f:
                        for n in self.unique_names:
                            f.write(n + '\n')
                    
                    self.status_label.config(text=f"已删除用户: {target_name}")
            else:
                messagebox.showwarning("警告", "无效的编号")
        except ValueError:
            messagebox.showwarning("警告", "请输入数字")

    def _set_threshold(self):
        """设置置信度阈值"""
        current = self.confidence_threshold
        new_val = simpledialog.askinteger("设置阈值", 
                                         f"当前阈值: {current}%\n请输入新阈值(0-100):",
                                         minvalue=0, maxvalue=100)
        if new_val is not None:
            self.confidence_threshold = new_val
            self.status_label.config(text=f"阈值已设置为: {new_val}%")

    def on_close(self):
        """关闭窗口时强制释放资源"""
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None  # 确保彻底释放
        self.root.destroy()

if __name__ == "__main__":
    root = Tk()
    app = FaceRecognitionSystem(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()