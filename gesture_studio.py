"""커스텀 제스처 스튜디오 (GUI) - 수집 → 학습 → 실시간 인식을 한 창에서.

실행: python gesture_studio.py

탭 구성
    1. 데이터 수집: 제스처 추가/삭제, 녹화(Space), 제스처별 개수·진행률 표시
    2. 학습: 버튼 한 번으로 학습, 정확도·혼동 행렬 로그 표시
    3. 실시간 인식: 결과 + 클래스별 확률 막대, 임계값/스무딩 조절
"""
import csv
import queue
import threading
import time
import tkinter as tk
from collections import Counter, deque
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import cv2
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageFont, ImageTk

import train_gestures
from emoji_overlay import emoji_image
from gesture_features import CSV_COLUMNS, landmarks_to_row, row_to_feature

ROOT = Path(__file__).parent
HAND_MODEL = ROOT / "hand_landmarker.task"
DATA_CSV = ROOT / "data" / "gestures.csv"
LABELS_TXT = ROOT / "data" / "labels.txt"
MODEL_PATH = ROOT / "models" / "custom_gesture.joblib"
VIEW_W = 640

TAB_COLLECT, TAB_TRAIN, TAB_INFER = range(3)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def load_font(size):
    for path in ("C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/malgunbd.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def read_csv_rows():
    if not DATA_CSV.exists():
        return []
    with open(DATA_CSV, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))[1:]


class GestureStudio:
    def __init__(self, root):
        self.root = root
        root.title("커스텀 제스처 스튜디오")
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.font_label = load_font(22)
        self.font_big = load_font(140)
        self.font_info = load_font(20)

        # 수집 상태
        self.counts = Counter(row[0] for row in read_csv_rows())
        self.labels = self.load_labels()
        self.recording = False
        self.countdown_end = 0.0
        self.csv_file = None
        self.csv_writer = None
        self.frame_idx = 0

        # 학습 / 인식 상태
        self.train_queue = queue.Queue()
        self.training = False
        self.clf = None
        self.clf_labels = []
        self.history = {}
        self.prob_bars = []

        self.build_ui()
        self.refresh_label_list()

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.start_time = time.monotonic()
        self.last_ts = -1
        self.prev_time = self.start_time
        self.cap = None
        self.open_camera()

        if MODEL_PATH.exists():
            self.load_model(MODEL_PATH)

        root.bind("<space>", self.on_space)
        self.update_loop()

    # ------------------------------------------------------------------ UI
    def build_ui(self):
        style = ttk.Style()
        style.configure("Rec.TButton", font=("맑은 고딕", 12, "bold"), padding=8)
        style.configure("Big.TLabel", font=("맑은 고딕", 22, "bold"))
        style.configure("Hint.TLabel", foreground="#666")

        main = ttk.Frame(self.root, padding=8)
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True)
        self.video = tk.Label(left, bg="black", width=VIEW_W, height=480)
        self.video.pack()

        bar = ttk.Frame(left, padding=(0, 6))
        bar.pack(fill="x")
        ttk.Label(bar, text="카메라").pack(side="left")
        self.cam_var = tk.IntVar(value=0)
        ttk.Spinbox(bar, from_=0, to=9, width=4, textvariable=self.cam_var).pack(side="left", padx=4)
        ttk.Button(bar, text="카메라 전환", takefocus=False,
                   command=self.open_camera).pack(side="left")
        self.fps_var = tk.StringVar(value="")
        ttk.Label(bar, textvariable=self.fps_var).pack(side="right")

        self.nb = ttk.Notebook(main, width=440)
        self.nb.pack(side="right", fill="both", padx=(8, 0))
        self.build_collect_tab()
        self.build_train_tab()
        self.build_infer_tab()
        self.nb.bind("<<NotebookTabChanged>>", self.on_tab_changed)

    def build_collect_tab(self):
        tab = ttk.Frame(self.nb, padding=10)
        self.nb.add(tab, text=" 1. 데이터 수집 ")

        add = ttk.Frame(tab)
        add.pack(fill="x")
        self.new_label_var = tk.StringVar()
        entry = ttk.Entry(add, textvariable=self.new_label_var)
        entry.pack(side="left", fill="x", expand=True)
        entry.bind("<Return>", lambda e: self.add_label())
        ttk.Button(add, text="제스처 추가", takefocus=False,
                   command=self.add_label).pack(side="left", padx=(4, 0))

        self.listbox = tk.Listbox(tab, height=9, font=("맑은 고딕", 11),
                                  exportselection=False, activestyle="none")
        self.listbox.pack(fill="x", pady=6)
        self.listbox.bind("<<ListboxSelect>>", lambda e: self.on_label_selected())

        ttk.Button(tab, text="선택한 제스처와 데이터 삭제", takefocus=False,
                   command=self.delete_label).pack(anchor="e")

        opts = ttk.LabelFrame(tab, text="녹화 설정", padding=8)
        opts.pack(fill="x", pady=8)
        self.target_var = tk.IntVar(value=300)
        self.every_var = tk.IntVar(value=2)
        self.countdown_var = tk.IntVar(value=3)
        for row, (text, var, lo, hi) in enumerate([
                ("제스처당 목표 개수", self.target_var, 10, 5000),
                ("N프레임마다 저장", self.every_var, 1, 10),
                ("시작 카운트다운(초)", self.countdown_var, 0, 10)]):
            ttk.Label(opts, text=text).grid(row=row, column=0, sticky="w", pady=2)
            ttk.Spinbox(opts, from_=lo, to=hi, width=7, textvariable=var).grid(
                row=row, column=1, sticky="e", pady=2)
        opts.columnconfigure(0, weight=1)

        self.progress_var = tk.StringVar(value="")
        ttk.Label(tab, textvariable=self.progress_var).pack(anchor="w")
        self.progress = ttk.Progressbar(tab, maximum=100)
        self.progress.pack(fill="x", pady=(2, 8))

        self.rec_btn = ttk.Button(tab, text="● 녹화 시작 (Space)", style="Rec.TButton",
                                  takefocus=False, command=self.toggle_record)
        self.rec_btn.pack(fill="x")

        ttk.Label(tab, style="Hint.TLabel", wraplength=410, justify="left", text=(
            "팁: 'none' 제스처(아무 손 모양)를 꼭 함께 수집하세요.\n"
            "녹화 중 손을 앞뒤·좌우로 움직이고 살짝 기울이면 인식이 좋아집니다.\n"
            "수집할 때는 한 손만 화면에 보여주세요.")).pack(anchor="w", pady=(10, 0))

    def build_train_tab(self):
        tab = ttk.Frame(self.nb, padding=10)
        self.nb.add(tab, text=" 2. 학습 ")

        top = ttk.Frame(tab)
        top.pack(fill="x")
        self.train_btn = ttk.Button(top, text="학습 시작", style="Rec.TButton",
                                    takefocus=False, command=self.start_training)
        self.train_btn.pack(side="left")
        self.train_bar = ttk.Progressbar(top, mode="indeterminate")
        self.train_bar.pack(side="left", fill="x", expand=True, padx=(8, 0))

        self.train_summary = tk.StringVar(value="수집한 데이터로 분류 모델을 학습합니다.")
        ttk.Label(tab, textvariable=self.train_summary, wraplength=410).pack(anchor="w", pady=8)

        self.log = ScrolledText(tab, height=24, font=("Consolas", 9), wrap="none")
        self.log.pack(fill="both", expand=True)

    def build_infer_tab(self):
        tab = ttk.Frame(self.nb, padding=10)
        self.nb.add(tab, text=" 3. 실시간 인식 ")

        mrow = ttk.Frame(tab)
        mrow.pack(fill="x")
        self.model_var = tk.StringVar(value="모델 없음")
        ttk.Label(mrow, textvariable=self.model_var, wraplength=250).pack(side="left")
        ttk.Button(mrow, text="모델 열기…", takefocus=False,
                   command=self.choose_model).pack(side="right")

        self.emoji_label = ttk.Label(tab)  # rock 등 이모지가 지정된 제스처일 때 표시
        self.emoji_label.pack(pady=(12, 0))
        self.emoji_shown = None
        self.result_var = tk.StringVar(value="-")
        ttk.Label(tab, textvariable=self.result_var, style="Big.TLabel").pack(pady=(0, 12))

        sliders = ttk.LabelFrame(tab, text="설정", padding=8)
        sliders.pack(fill="x")
        self.threshold_var = tk.DoubleVar(value=0.7)
        self.smooth_var = tk.IntVar(value=5)
        self.threshold_txt = tk.StringVar()
        self.smooth_txt = tk.StringVar()
        for row, (text, var, txt, lo, hi) in enumerate([
                ("확신도 임계값", self.threshold_var, self.threshold_txt, 0.3, 0.99),
                ("스무딩 (프레임)", self.smooth_var, self.smooth_txt, 1, 15)]):
            ttk.Label(sliders, text=text).grid(row=row, column=0, sticky="w")
            ttk.Scale(sliders, from_=lo, to=hi, variable=var,
                      command=lambda _v: self.update_slider_text()).grid(
                row=row, column=1, sticky="ew", padx=6)
            ttk.Label(sliders, textvariable=txt, width=5).grid(row=row, column=2)
        sliders.columnconfigure(1, weight=1)
        self.update_slider_text()

        ttk.Label(tab, text="클래스별 확률 (첫 번째 손)").pack(anchor="w", pady=(12, 4))
        self.prob_frame = ttk.Frame(tab)
        self.prob_frame.pack(fill="x")
        ttk.Label(tab, style="Hint.TLabel", wraplength=410, text=(
            "확률이 임계값보다 낮으면 'Unknown'으로 표시됩니다.")).pack(anchor="w", pady=(10, 0))

    def update_slider_text(self):
        self.smooth_var.set(int(round(self.smooth_var.get())))
        self.threshold_txt.set(f"{self.threshold_var.get():.2f}")
        self.smooth_txt.set(str(self.smooth_var.get()))

    # -------------------------------------------------------------- 카메라
    def open_camera(self):
        if self.cap is not None:
            self.cap.release()
        idx = self.cam_var.get()
        self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            messagebox.showerror("카메라", f"카메라 {idx}번을 열 수 없습니다.")

    def update_loop(self):
        self.poll_training()
        ok, frame = (self.cap.read() if self.cap is not None else (False, None))
        if ok:
            self.process_frame(frame)
        self.root.after(5, self.update_loop)

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        ts = max(int((time.monotonic() - self.start_time) * 1000), self.last_ts + 1)
        self.last_ts = ts
        result = self.landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

        tab = self.nb.index(self.nb.select())
        texts = []  # (x, y, 문자열, 색) - PIL로 한글 그리기
        self.emojis = []  # (라벨, x, y, 크기) - 영상 위 이모지
        if tab == TAB_COLLECT:
            self.handle_collect(rgb, result, w, h, texts)
        elif tab == TAB_INFER:
            self.handle_infer(rgb, result, w, h, texts)
        else:
            for lm in result.hand_landmarks:
                self.draw_hand(rgb, lm, w, h, (0, 255, 0))

        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)
        for x, y, text, color, font in texts:
            box = draw.textbbox((x, y), text, font=font)
            draw.rectangle((box[0] - 6, box[1] - 4, box[2] + 6, box[3] + 4), fill=(30, 30, 30))
            draw.text((x, y), text, font=font, fill=color)
        for label, x, y, size in self.emojis:
            emoji = emoji_image(label, size)
            if emoji is not None:
                img.paste(emoji, (x, y), emoji)
        if img.width != VIEW_W:
            img = img.resize((VIEW_W, int(img.height * VIEW_W / img.width)))
        self.photo = ImageTk.PhotoImage(img)
        self.video.configure(image=self.photo, width=img.width, height=img.height)

        now = time.monotonic()
        self.fps_var.set(f"FPS {1.0 / max(now - self.prev_time, 1e-6):.1f}")
        self.prev_time = now

    @staticmethod
    def draw_hand(rgb, landmarks, w, h, color):
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(rgb, pts[a], pts[b], color, 2)
        for p in pts:
            cv2.circle(rgb, p, 4, (255, 255, 255), -1)
        return pts

    # ---------------------------------------------------------------- 수집
    def load_labels(self):
        labels = []
        if LABELS_TXT.exists():
            labels = [s.strip() for s in LABELS_TXT.read_text(encoding="utf-8").splitlines() if s.strip()]
        for label in self.counts:
            if label not in labels:
                labels.append(label)
        return labels or ["none"]

    def save_labels(self):
        LABELS_TXT.parent.mkdir(parents=True, exist_ok=True)
        LABELS_TXT.write_text("\n".join(self.labels) + "\n", encoding="utf-8")

    def current_label(self):
        sel = self.listbox.curselection()
        return self.labels[sel[0]] if sel else None

    def refresh_label_list(self, select=None):
        if select is None:
            select = self.current_label()
        self.listbox.delete(0, "end")
        for label in self.labels:
            self.listbox.insert("end", self.label_line(label))
        if self.labels:
            i = self.labels.index(select) if select in self.labels else 0
            self.listbox.selection_set(i)
            self.listbox.see(i)
        self.update_progress()

    def label_line(self, label):
        target = self.safe_int(self.target_var, 300)
        n = self.counts[label]
        mark = "✔" if n >= target else "  "
        return f"{mark} {label}  —  {n}개"

    def update_progress(self):
        label = self.current_label()
        if label is None:
            self.progress_var.set("")
            self.progress["value"] = 0
            return
        target = self.safe_int(self.target_var, 300)
        n = self.counts[label]
        self.progress_var.set(f"선택: {label}  ({n} / {target})")
        self.progress["value"] = min(100, n * 100 / max(target, 1))

    def add_label(self):
        name = self.new_label_var.get().strip().replace(",", "_")
        if not name:
            return
        if name not in self.labels:
            self.labels.append(name)
            self.save_labels()
        self.new_label_var.set("")
        self.refresh_label_list(select=name)
        self.root.focus_set()

    def delete_label(self):
        label = self.current_label()
        if label is None:
            return
        n = self.counts[label]
        if not messagebox.askyesno(
                "삭제 확인", f"'{label}' 제스처와 수집한 데이터 {n}개를 삭제할까요?\n되돌릴 수 없습니다."):
            return
        self.stop_record()
        if n:
            rows = [r for r in read_csv_rows() if r[0] != label]
            with open(DATA_CSV, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_COLUMNS)
                writer.writerows(rows)
            del self.counts[label]
        self.labels.remove(label)
        self.save_labels()
        self.refresh_label_list()

    def on_label_selected(self):
        self.stop_record()
        self.update_progress()

    def on_space(self, event):
        if isinstance(event.widget, (tk.Entry, ttk.Entry, ttk.Spinbox)):
            return
        if self.nb.index(self.nb.select()) == TAB_COLLECT:
            self.toggle_record()

    def toggle_record(self):
        if self.recording:
            self.stop_record()
        else:
            self.start_record()

    def start_record(self):
        label = self.current_label()
        if label is None:
            messagebox.showinfo("수집", "먼저 제스처를 추가하고 선택하세요.")
            return
        DATA_CSV.parent.mkdir(parents=True, exist_ok=True)
        is_new = not DATA_CSV.exists() or DATA_CSV.stat().st_size == 0
        self.csv_file = open(DATA_CSV, "a", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        if is_new:
            self.csv_writer.writerow(CSV_COLUMNS)
        self.recording = True
        self.countdown_end = time.monotonic() + self.safe_int(self.countdown_var, 3)
        self.rec_btn.configure(text="■ 녹화 정지 (Space)")
        self.listbox.configure(state="disabled")

    def stop_record(self):
        if self.csv_file:
            self.csv_file.close()
        self.csv_file = self.csv_writer = None
        self.recording = False
        self.rec_btn.configure(text="● 녹화 시작 (Space)")
        self.listbox.configure(state="normal")

    def handle_collect(self, rgb, result, w, h, texts):
        label = self.current_label()
        counting = self.recording and time.monotonic() < self.countdown_end
        saving = self.recording and not counting

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            color = (255, 60, 60) if saving else (0, 255, 0)
            self.draw_hand(rgb, landmarks, w, h, color)
            self.frame_idx += 1
            if saving and self.frame_idx % self.safe_int(self.every_var, 2) == 0:
                hand = result.handedness[0][0].category_name
                self.csv_writer.writerow([label, hand] + landmarks_to_row(landmarks, w, h))
                self.counts[label] += 1
                i = self.labels.index(label)
                self.listbox.configure(state="normal")
                self.listbox.delete(i)
                self.listbox.insert(i, self.label_line(label))
                self.listbox.selection_set(i)
                self.listbox.configure(state="disabled")
                self.update_progress()
                if self.counts[label] >= self.safe_int(self.target_var, 300):
                    self.stop_record()
                    self.csv_flushed_msg(label)

        if counting:
            remain = int(self.countdown_end - time.monotonic()) + 1
            texts.append((w // 2 - 40, h // 2 - 100, str(remain), (255, 220, 0), self.font_big))
        status = "● 녹화 중" if saving else ("준비…" if counting else "대기")
        texts.append((10, 10, f"{status}  |  {label or '-'}  {self.counts[label] if label else ''}",
                      (255, 80, 80) if saving else (255, 255, 255), self.font_info))
        if not result.hand_landmarks and saving:
            texts.append((10, 45, "손이 보이지 않습니다", (255, 200, 0), self.font_info))

    def csv_flushed_msg(self, label):
        self.progress_var.set(f"'{label}' 목표 개수 도달! 다른 제스처를 선택하세요.")

    # ---------------------------------------------------------------- 학습
    def start_training(self):
        if self.training:
            return
        self.stop_record()
        self.training = True
        self.train_btn.configure(state="disabled")
        self.train_bar.start(10)
        self.log.delete("1.0", "end")
        self.train_summary.set("학습 중…")

        def worker():
            try:
                res = train_gestures.train_model(
                    DATA_CSV, MODEL_PATH, log=lambda s: self.train_queue.put(("log", s)))
                self.train_queue.put(("done", res))
            except Exception as e:  # noqa: BLE001 - GUI에 오류 표시
                self.train_queue.put(("error", str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def poll_training(self):
        while not self.train_queue.empty():
            kind, payload = self.train_queue.get()
            if kind == "log":
                self.log.insert("end", payload + "\n")
                self.log.see("end")
                continue
            self.training = False
            self.train_btn.configure(state="normal")
            self.train_bar.stop()
            self.train_bar["value"] = 0
            if kind == "done":
                self.train_summary.set(
                    f"완료! 검증 정확도 {payload['accuracy'] * 100:.1f}%  →  "
                    "'3. 실시간 인식' 탭에서 바로 테스트하세요.")
                self.load_model(payload["path"])
            else:
                self.train_summary.set(f"오류: {payload}")
                self.log.insert("end", f"\n[오류] {payload}\n")

    # ---------------------------------------------------------------- 인식
    def choose_model(self):
        path = filedialog.askopenfilename(
            initialdir=MODEL_PATH.parent if MODEL_PATH.parent.exists() else ROOT,
            filetypes=[("joblib 모델", "*.joblib"), ("모든 파일", "*.*")])
        if path:
            self.load_model(Path(path))

    def load_model(self, path):
        try:
            bundle = joblib.load(path)
            self.clf, self.clf_labels = bundle["model"], list(bundle["labels"])
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("모델", f"모델을 불러올 수 없습니다:\n{e}")
            return
        self.history.clear()
        self.model_var.set(f"모델: {Path(path).name}  ({len(self.clf_labels)}개 클래스)")
        for child in self.prob_frame.winfo_children():
            child.destroy()
        self.prob_bars = []
        for i, label in enumerate(self.clf_labels):
            ttk.Label(self.prob_frame, text=label, width=12).grid(row=i, column=0, sticky="w")
            bar = ttk.Progressbar(self.prob_frame, maximum=100)
            bar.grid(row=i, column=1, sticky="ew", pady=2)
            val = tk.StringVar(value="0%")
            ttk.Label(self.prob_frame, textvariable=val, width=5).grid(row=i, column=2)
            self.prob_bars.append((bar, val))
        self.prob_frame.columnconfigure(1, weight=1)

    def handle_infer(self, rgb, result, w, h, texts):
        if self.clf is None:
            texts.append((10, 10, "모델이 없습니다. '2. 학습' 탭에서 먼저 학습하세요.",
                          (255, 200, 0), self.font_info))
            for lm in result.hand_landmarks:
                self.draw_hand(rgb, lm, w, h, (0, 255, 0))
            return

        threshold = self.threshold_var.get()
        smooth = max(1, self.smooth_var.get())
        seen, first = set(), None
        for landmarks, hd in zip(result.hand_landmarks, result.handedness):
            hand = hd[0].category_name
            seen.add(hand)
            feat = row_to_feature(landmarks_to_row(landmarks, w, h), hand)
            proba = self.clf.predict_proba(feat[None, :])[0]
            hist = self.history.get(hand)
            if hist is None or hist.maxlen != smooth:
                hist = self.history[hand] = deque(hist or [], maxlen=smooth)
            hist.append(proba)
            avg = np.mean(hist, axis=0)
            best = int(np.argmax(avg))
            name = self.clf_labels[best] if avg[best] >= threshold else "Unknown"
            if first is None:
                first = (name, avg)

            pts = self.draw_hand(rgb, landmarks, w, h,
                                 (0, 255, 0) if name != "Unknown" else (160, 160, 160))
            x0, y0 = min(p[0] for p in pts), min(p[1] for p in pts)
            x1 = max(p[0] for p in pts)
            self.emojis.append((name, min(x1 + 10, w - 110), max(0, y0), 110))
            texts.append((x0, max(0, y0 - 40), f"{hand}: {name} ({avg[best]:.2f})",
                          (255, 230, 0), self.font_label))

        for hand in list(self.history):
            if hand not in seen:
                del self.history[hand]

        self.show_panel_emoji(first[0] if first else None)
        if first is None:
            self.result_var.set("손 없음")
            for bar, val in self.prob_bars:
                bar["value"] = 0
                val.set("0%")
        else:
            name, avg = first
            self.result_var.set(name)
            for (bar, val), p in zip(self.prob_bars, avg):
                bar["value"] = p * 100
                val.set(f"{p * 100:.0f}%")

    def show_panel_emoji(self, name):
        if name == self.emoji_shown:
            return
        self.emoji_shown = name
        img = emoji_image(name, 120) if name else None
        self.emoji_photo = ImageTk.PhotoImage(img) if img is not None else None
        self.emoji_label.configure(image=self.emoji_photo or "")

    # ---------------------------------------------------------------- 기타
    def on_tab_changed(self, _event):
        if self.nb.index(self.nb.select()) != TAB_COLLECT:
            self.stop_record()

    @staticmethod
    def safe_int(var, default):
        try:
            return int(var.get())
        except (tk.TclError, ValueError):
            return default

    def on_close(self):
        self.stop_record()
        if self.cap is not None:
            self.cap.release()
        self.landmarker.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    GestureStudio(root)
    root.mainloop()


if __name__ == "__main__":
    main()
