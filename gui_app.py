"""Green Screen Studio - Tkinter GUI (uses keying.py + filters.py).

Run:  python gui_app.py [camera_index]
Ubuntu needs:  sudo apt install python3-tk     and   pip install pillow opencv-python numpy
"""
import sys, time, threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import filters, keying

W, H = 640, 480
BG, PANEL, CYAN, FG = "#3c3c3c", "#2b2b2b", "#29b6f6", "#ffffff"
FILTERS = ["Normal", "Greyscale", "Sepia", "Invert", "Posterize", "CMYK plates"]


class App:
    def __init__(self, root, cam_index=0):
        self.root, self.cam_index = root, cam_index
        root.title("Green Screen Studio")
        root.configure(bg=BG)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.cap = None
        self.running = False
        self.frame, self.frame_id, self.done_id = None, 0, 0
        self.raw = None
        self.key_bgr = np.array([40, 200, 40], np.uint8)
        self.key_hsv = cv2.cvtColor(self.key_bgr.reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0]
        self.bg_img, self.bg_vid = None, None
        self.writer = None
        self.fps, self.t_prev = 0.0, time.time()
        self.use_rgb = False
        self._build()
        self._set_bg(None)
        self._show(self.preview, np.zeros((H, W, 3), np.uint8))

    # ---------------- UI ----------------
    def _build(self):
        tk.Label(self.root, text="GREEN SCREEN STUDIO", bg=BG, fg=FG,
                 font=("Helvetica", 18, "bold")).pack(pady=6)
        body = tk.Frame(self.root, bg=BG)
        body.pack(padx=10, pady=4)

        left = tk.Frame(body, bg="#505050")
        left.grid(row=0, column=0, padx=6)
        self.preview = tk.Label(left, bg="black", width=W, height=H)
        self.preview.pack(padx=6, pady=6)
        self.preview.bind("<Button-1>", self.on_click)

        row = tk.Frame(left, bg="#505050")
        row.pack(padx=6, pady=6, anchor="w")
        self.hsv_lbl = self.rgb_lbl = self.hist_lbl = None
        for i, (attr, t) in enumerate([("hsv_lbl", "HSV mask"), ("rgb_lbl", "RGB mask"),
                                       ("hist_lbl", "Live hue histogram (key region)")]):
            f = tk.Frame(row, bg="#505050")
            f.grid(row=0, column=i, padx=4)
            tk.Label(f, text=t, bg="#505050", fg=FG, font=("Helvetica", 8)).pack()
            lbl = tk.Label(f, bg="black")
            lbl.pack()
            setattr(self, attr, lbl)

        right = tk.Frame(body, bg=PANEL, padx=12, pady=8)
        right.grid(row=0, column=1, sticky="ns", padx=6)
        tk.Label(right, text="FILTER", bg=PANEL, fg=FG, font=("Helvetica", 9, "bold")).pack()
        self.filter_var = tk.StringVar(value="Normal")
        ttk.Combobox(right, textvariable=self.filter_var, values=FILTERS,
                     state="readonly", width=22).pack(pady=4)
        self._btn(right, "Choose Image / Video Background", self.choose_bg)
        self._btn(right, "Start Camera", self.start_camera)
        self._btn(right, "Stop Camera", self.stop_camera)
        self._btn(right, "Start Recording", self.start_rec)
        self._btn(right, "Stop Recording", self.stop_rec)

        tk.Label(right, text="KEY COLOUR", bg=PANEL, fg=FG,
                 font=("Helvetica", 9, "bold")).pack(pady=(10, 0))
        tk.Label(right, text="Click the green area in the preview to pick it.",
                 bg=PANEL, fg=FG, font=("Helvetica", 8)).pack()
        self.swatch = tk.Label(right, width=10, height=1, bg="#28c828")
        self.swatch.pack(pady=3)
        mrow = tk.Frame(right, bg=PANEL)
        mrow.pack()
        self.b_hsv = tk.Button(mrow, text="HSV", width=6, command=lambda: self.set_mode(False))
        self.b_rgb = tk.Button(mrow, text="RGB", width=6, command=lambda: self.set_mode(True))
        self.b_hsv.pack(side="left"); self.b_rgb.pack(side="left")
        self.set_mode(False)

        self.s_h = self._slider(right, "H tolerance", 0, 90, 12)
        self.s_s = self._slider(right, "S tolerance", 0, 255, 90)
        self.s_v = self._slider(right, "V tolerance", 0, 255, 110)
        self.s_d = self._slider(right, "RGB distance", 0, 442, 90)
        self.s_f = self._slider(right, "Edge feather", 0, 15, 3)
        self.s_o = self._slider(right, "Speck removal (open)", 0, 15, 3)
        self.s_c = self._slider(right, "Hole filling (close)", 0, 31, 7)
        self.s_p = self._slider(right, "Despill %", 0, 100, 100)

        self.status = tk.Label(self.root, text="Camera stopped", bg=BG, fg=FG,
                               font=("Helvetica", 9, "bold"))
        self.status.pack(pady=4)

    def _btn(self, parent, text, cmd):
        tk.Button(parent, text=text, command=cmd, bg=CYAN, fg="black", relief="flat",
                  font=("Helvetica", 9, "bold"), width=30).pack(pady=3)

    def _slider(self, parent, text, lo, hi, init):
        var = tk.IntVar(value=init)
        lbl = tk.Label(parent, text=f"{text}: {init}", bg=PANEL, fg=FG, font=("Helvetica", 8, "bold"))
        lbl.pack(pady=(6, 0))
        tk.Scale(parent, from_=lo, to=hi, orient="horizontal", variable=var, showvalue=0,
                 length=210, bg=PANEL, fg=FG, troughcolor="#555", highlightthickness=0,
                 command=lambda v: lbl.config(text=f"{text}: {int(float(v))}")).pack()
        return var

    def set_mode(self, rgb):
        self.use_rgb = rgb
        self.b_hsv.config(bg="#bbb" if rgb else CYAN)
        self.b_rgb.config(bg=CYAN if rgb else "#bbb")

    # ---------------- background ----------------
    def _set_bg(self, path):
        self.bg_vid, self.bg_img = None, None
        if path and path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            self.bg_vid = cv2.VideoCapture(path)
        elif path:
            img = cv2.imread(path)
            if img is not None:
                self.bg_img = cv2.resize(img, (W, H))
        if self.bg_vid is None and self.bg_img is None:
            g = np.linspace(40, 200, H, dtype=np.uint8)[:, None]
            self.bg_img = np.dstack([np.repeat(g, W, 1)] * 3)[..., ::-1].copy()

    def choose_bg(self):
        p = filedialog.askopenfilename(filetypes=[("Image/Video", "*.png *.jpg *.jpeg *.bmp *.mp4 *.avi *.mov *.mkv")])
        if p:
            self._set_bg(p)

    def _next_bg(self):
        if self.bg_vid is None:
            return self.bg_img
        ok, f = self.bg_vid.read()
        if not ok:
            self.bg_vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, f = self.bg_vid.read()
        return cv2.resize(f, (W, H)) if ok else np.zeros((H, W, 3), np.uint8)

    # ---------------- camera ----------------
    def start_camera(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(self.cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        if not self.cap.isOpened():
            messagebox.showerror("Camera", "Cannot open camera")
            return
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()
        self.status.config(text="Camera running")
        self.tick()

    def _reader(self):
        while self.running:
            ok, f = self.cap.read()
            if ok:
                self.frame = cv2.resize(cv2.flip(f, 1), (W, H))
                self.frame_id += 1
            else:
                time.sleep(0.01)

    def stop_camera(self):
        self.stop_rec()
        self.running = False
        time.sleep(0.05)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.frame = None
        self.status.config(text="Camera stopped")
        self._show(self.preview, np.zeros((H, W, 3), np.uint8))

    # ---------------- recording ----------------
    def start_rec(self):
        if not self.running or self.writer is not None:
            return
        name = time.strftime("recording_%Y%m%d_%H%M%S.mp4")
        self.writer = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*"mp4v"),
                                      max(10.0, min(30.0, self.fps or 20.0)), (W, H))
        self.status.config(text=f"Recording -> {name}")

    def stop_rec(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            self.status.config(text="Camera running" if self.running else "Camera stopped")

    # ---------------- key pick ----------------
    def on_click(self, ev):
        if self.raw is None:
            return
        x = min(max(ev.x, 0), W - 1); y = min(max(ev.y, 0), H - 1)
        self.key_bgr, self.key_hsv = keying.pick_key(self.raw, x, y)
        b, g, r = (int(c) for c in self.key_bgr)
        self.swatch.config(bg=f"#{r:02x}{g:02x}{b:02x}")

    # ---------------- main loop ----------------
    def tick(self):
        if not self.running:
            return
        if self.frame is not None and self.frame_id != self.done_id:
            self.done_id = self.frame_id
            self.process(self.frame.copy())
        self.root.after(5, self.tick)

    def process(self, frame):
        self.raw = frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        m_hsv = keying.hsv_key_mask(hsv, self.key_hsv, self.s_h.get(), self.s_s.get(), self.s_v.get())
        m_rgb = keying.rgb_key_mask(frame, self.key_bgr, self.s_d.get())
        bg_mask = m_rgb if self.use_rgb else m_hsv
        alpha = keying.clean_alpha(bg_mask, self.s_o.get(), self.s_c.get(), self.s_f.get())
        fg = keying.despill(frame, self.key_bgr, self.s_p.get() / 100.0)
        out = keying.composite(fg, self._next_bg(), alpha)

        name = self.filter_var.get()
        if name == "Greyscale": out = filters.greyscale(out)
        elif name == "Sepia": out = filters.sepia(out)
        elif name == "Invert": out = filters.invert(out)
        elif name == "Posterize": out = filters.posterize(out, 4)
        elif name == "CMYK plates":
            out = cv2.resize(filters.cmyk_view(out), (W, H))

        if self.writer is not None:
            self.writer.write(out)

        now = time.time()
        self.fps = 0.9 * self.fps + 0.1 / max(now - self.t_prev, 1e-6) if self.fps else 15.0
        self.t_prev = now
        shown = out.copy()
        cv2.putText(shown, f"FPS: {self.fps:.1f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
        if self.writer is not None:
            cv2.circle(shown, (W - 20, 20), 8, (0, 0, 255), -1)
        self._show(self.preview, shown)
        small = lambda m: cv2.cvtColor(cv2.resize(m, (160, 120)), cv2.COLOR_GRAY2BGR)
        self._show(self.hsv_lbl, small(m_hsv))
        self._show(self.rgb_lbl, small(m_rgb))
        self._show(self.hist_lbl, self.hue_hist(hsv, bg_mask))

    def hue_hist(self, hsv, region):
        hues = hsv[..., 0][region > 0]
        counts = np.bincount(hues.ravel(), minlength=180) if hues.size else np.zeros(180, int)
        img = np.zeros((120, 360, 3), np.uint8)
        top = counts.max() or 1
        for h in range(180):
            col = cv2.cvtColor(np.uint8([[[h, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
            cv2.rectangle(img, (h * 2, 119), (h * 2 + 1, 119 - int(counts[h] / top * 115)), col, -1)
        return img

    @staticmethod
    def _show(label, bgr):
        im = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        label.configure(image=im, width=bgr.shape[1], height=bgr.shape[0])
        label.image = im

    def on_close(self):
        self.stop_camera()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    App(root, int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    root.mainloop()
