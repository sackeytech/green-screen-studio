import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import time
import os


class GreenScreenStudio:
    def __init__(self, root):
        self.root = root
        self.root.title("Green-Screen Studio")
        self.root.configure(bg="#1e2227")
        self.root.resizable(False, False)

        # --- Video Capture Setup (Task 1: >= 15 FPS) ---
        self.frame_w, self.frame_h = 640, 420
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_h)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # --- Default Chroma Key State (Task 2) ---
        # Default green in BGR, HSV, and RGB
        self.key_bgr = np.array([40, 200, 40], dtype=np.uint8)
        self.key_hsv = cv2.cvtColor(np.uint8([[self.key_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        self.key_rgb = np.array([40, 200, 40], dtype=np.float32)

        # View Mode ('HSV', 'RGB', 'Original') & Active Filter (Task 5)
        self.view_mode = "HSV"
        self.active_filter = None
        self.cmyk_quad_mode = True  # Shows 4-channel split view first when C/M/Y/K is clicked

        # --- Background State (Task 4) ---
        self.bg_image = self._create_default_virtual_set()
        self.bg_video_cap = None

        # --- MP4 Recording State (Task 4) ---
        self.is_recording = False
        self.video_writer = None

        # --- FPS Tracking (Task 1) ---
        self.prev_time = time.time()
        self.fps = 0.0
        self.current_raw_frame = None
        self.current_clean_mask = None
        self.saved_eval_count = 0

        self._build_ui()
        self.root.bind("<s>", self._save_evaluation_pair)
        self._update_loop()

    def _create_default_virtual_set(self):
        """Creates a default studio background plate until an image/video is imported."""
        y = np.linspace(0, 1, self.frame_h)[:, None]
        x = np.linspace(0, 1, self.frame_w)[None, :]
        bg = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)
        bg[:, :, 0] = (40 + 80 * (1 - y)).astype(np.uint8)   # Blue gradient
        bg[:, :, 1] = (25 + 45 * x).astype(np.uint8)         # Green gradient
        bg[:, :, 2] = (20 + 60 * y).astype(np.uint8)         # Red gradient
        return bg

    # =========================================================================
    # UI LAYOUT (Matches Wireframe & Generated Mockup)
    # =========================================================================
    def _build_ui(self):
        main_frame = tk.Frame(self.root, bg="#1e2227", padx=12, pady=12)
        main_frame.pack(fill=tk.BOTH, expand=True)

        top_section = tk.Frame(main_frame, bg="#1e2227")
        top_section.pack(fill=tk.BOTH, expand=True)

        # --- LEFT PANEL: Video Feed + Mode Tabs ---
        left_panel = tk.Frame(top_section, bg="#1e2227")
        left_panel.pack(side=tk.LEFT, padx=(0, 12))

        video_border = tk.Frame(
            left_panel, bg="#333842", bd=2, relief=tk.FLAT, padx=2, pady=2
        )
        video_border.pack()

        self.video_label = tk.Label(
            video_border, bg="#121417", width=self.frame_w, height=self.frame_h,
            cursor="crosshair"
        )
        self.video_label.pack()
        # Task 2: Click on live video to pick chroma key color
        self.video_label.bind("<Button-1>", self._on_video_click)

        # Middle Row: Original | RGB | HSV Tabs
        mode_frame = tk.Frame(left_panel, bg="#252a30", pady=2)
        mode_frame.pack(fill=tk.X, pady=(10, 0))

        self.mode_buttons = {}
        for mode in ["Original", "RGB", "HSV"]:
            btn = tk.Button(
                mode_frame, text=mode, font=("Segoe UI", 11, "bold"),
                bg="#3b424c" if mode == self.view_mode else "#252a30",
                fg="#ffffff" if mode == self.view_mode else "#aab2bd",
                activebackground="#48505c", activeforeground="#ffffff",
                bd=1, relief=tk.SOLID, height=2,
                command=lambda m=mode: self._set_view_mode(m)
            )
            btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
            self.mode_buttons[mode] = btn

        # --- RIGHT PANEL: Import, Record, Sliders, Live Hue Histogram ---
        right_panel = tk.Frame(
            top_section, bg="#252a30", width=270, padx=14, pady=14,
            highlightbackground="#3a4049", highlightthickness=1
        )
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        right_panel.pack_propagate(False)

        self.import_btn = tk.Button(
            right_panel, text="Import", font=("Segoe UI", 11, "bold"),
            bg="#2b4560", fg="#ffffff", activebackground="#36587a",
            activeforeground="#ffffff", relief=tk.FLAT, height=2,
            highlightbackground="#4a76a8", highlightthickness=1,
            command=self._import_background
        )
        self.import_btn.pack(fill=tk.X, pady=(0, 8))

        self.record_btn = tk.Button(
            right_panel, text="Record MP4", font=("Segoe UI", 11, "bold"),
            bg="#2d5240", fg="#ffffff", activebackground="#3b6e55",
            activeforeground="#ffffff", relief=tk.FLAT, height=2,
            highlightbackground="#4ca977", highlightthickness=1,
            command=self._toggle_recording
        )
        self.record_btn.pack(fill=tk.X, pady=(0, 14))

        # Task 2: H, S, V Tolerance Sliders
        self.h_tol = self._create_slider(right_panel, "Hue", 1, 90, 22)
        self.s_tol = self._create_slider(right_panel, "Saturation", 5, 255, 85)
        self.v_tol = self._create_slider(right_panel, "Value", 5, 255, 85)

        # Task 6: Live Hue Histogram Canvas
        hist_label = tk.Label(
            right_panel, text="Key Region Hue Histogram",
            font=("Segoe UI", 9), bg="#252a30", fg="#9aa2ad"
        )
        hist_label.pack(anchor="w", pady=(8, 2))

        self.hist_w, self.hist_h = 240, 110
        self.hist_canvas = tk.Label(
            right_panel, bg="#181b20", width=self.hist_w, height=self.hist_h,
            highlightbackground="#3a4049", highlightthickness=1
        )
        self.hist_canvas.pack(pady=(2, 0))

        # --- BOTTOM ROW: Custom NumPy Filter Bank (Task 5) ---
        bottom_frame = tk.Frame(main_frame, bg="#1e2227", pady=10)
        bottom_frame.pack(fill=tk.X)

        filters = ["Grayscale", "Sepia", "Invert", "Posterize", "C", "M", "Y", "K"]
        self.filter_buttons = {}
        for f_name in filters:
            btn = tk.Button(
                bottom_frame, text=f_name, font=("Segoe UI", 11),
                bg="#282d34", fg="#e0e6ed",
                activebackground="#3b424c", activeforeground="#ffffff",
                relief=tk.SOLID, bd=1, height=2, width=10,
                command=lambda fn=f_name: self._toggle_filter(fn)
            )
            btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
            self.filter_buttons[f_name] = btn

    def _create_slider(self, parent, title, min_v, max_v, init_v):
        lbl = tk.Label(
            parent, text=title, font=("Segoe UI", 10),
            bg="#252a30", fg="#dcdfe4"
        )
        lbl.pack(anchor="w", pady=(2, 0))
        slider = tk.Scale(
            parent, from_=min_v, to=max_v, orient=tk.HORIZONTAL,
            bg="#252a30", fg="#dcdfe4", troughcolor="#3b6e5c",
            highlightthickness=0, bd=0, sliderrelief=tk.FLAT,
            activebackground="#56b895", showvalue=True
        )
        slider.set(init_v)
        slider.pack(fill=tk.X, pady=(0, 6))
        return slider

    # =========================================================================
    # USER INTERACTION HANDLERS
    # =========================================================================
    def _on_video_click(self, event):
        """Task 2: Click on the video feed to pick the chroma key color."""
        if self.current_raw_frame is None:
            return
        x = int(np.clip(event.x, 0, self.frame_w - 1))
        y = int(np.clip(event.y, 0, self.frame_h - 1))

        # Sample a 5x5 patch around the click to avoid single-pixel camera noise
        y0, y1 = max(0, y - 2), min(self.frame_h, y + 3)
        x0, x1 = max(0, x - 2), min(self.frame_w, x + 3)
        patch_bgr = self.current_raw_frame[y0:y1, x0:x1]

        mean_bgr = patch_bgr.mean(axis=(0, 1)).astype(np.uint8)
        self.key_bgr = mean_bgr
        self.key_hsv = cv2.cvtColor(np.uint8([[mean_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        self.key_rgb = mean_bgr[::-1].astype(np.float32)

    def _set_view_mode(self, mode):
        self.view_mode = mode
        for m, btn in self.mode_buttons.items():
            btn.configure(
                bg="#3b424c" if m == mode else "#252a30",
                fg="#ffffff" if m == mode else "#aab2bd"
            )

    def _toggle_filter(self, f_name):
        """Task 5: Toggle custom NumPy filters and CMYK 4-channel / single-channel views."""
        if self.active_filter == f_name:
            if f_name in ("C", "M", "Y", "K") and self.cmyk_quad_mode:
                # Second click switches from 4-channel quad view to single channel
                self.cmyk_quad_mode = False
            else:
                self.active_filter = None
                self.cmyk_quad_mode = True
        else:
            self.active_filter = f_name
            self.cmyk_quad_mode = True

        for name, btn in self.filter_buttons.items():
            btn.configure(bg="#2b4560" if name == self.active_filter else "#282d34")

    def _import_background(self):
        """Task 4: Import a virtual set background image or video."""
        path = filedialog.askopenfilename(
            title="Select Virtual Set Image or Video",
            filetypes=[
                ("Media Files", "*.png *.jpg *.jpeg *.bmp *.mp4 *.avi *.mov"),
                ("All Files", "*.*")
            ]
        )
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        if ext in (".mp4", ".avi", ".mov", ".mkv"):
            if self.bg_video_cap is not None:
                self.bg_video_cap.release()
            self.bg_video_cap = cv2.VideoCapture(path)
        else:
            img = cv2.imread(path)
            if img is not None:
                if self.bg_video_cap is not None:
                    self.bg_video_cap.release()
                    self.bg_video_cap = None
                self.bg_image = cv2.resize(img, (self.frame_w, self.frame_h))

    def _toggle_recording(self):
        """Task 4: Record composited output to MP4."""
        if not self.is_recording:
            filename = f"studio_output_{int(time.time())}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(
                filename, fourcc, 20.0, (self.frame_w, self.frame_h)
            )
            self.is_recording = True
            self.record_btn.configure(text="Stop Recording", bg="#8c2d2d")
        else:
            self.is_recording = False
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            self.record_btn.configure(text="Record MP4", bg="#2d5240")

    def _save_evaluation_pair(self, event=None):
        """Task 7 Helper: Press 'S' to save frame + algorithm mask for evaluation."""
        if self.current_raw_frame is None or self.current_clean_mask is None:
            return
        os.makedirs("eval_frames", exist_ok=True)
        self.saved_eval_count += 1
        idx = self.saved_eval_count
        cv2.imwrite(f"eval_frames/frame_{idx}.png", self.current_raw_frame)
        # Save foreground mask (person = 255, background = 0)
        fg_mask = 255 - self.current_clean_mask
        cv2.imwrite(f"eval_frames/algo_mask_{idx}.png", fg_mask)
        print(f"[Task 7] Saved eval_frames/frame_{idx}.png and algo_mask_{idx}.png")

    # =========================================================================
    # CORE COMPUTER VISION & NUMPY ALGORITHMS (Tasks 2, 3, 4, 5, 6)
    # =========================================================================
    def _compute_masks(self, frame_bgr):
        """Task 2: Segment in HSV with sliders AND compute RGB-distance version."""
        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        h_t = self.h_tol.get()
        s_t = self.s_tol.get()
        v_t = self.v_tol.get()

        kh, ks, kv = int(self.key_hsv[0]), int(self.key_hsv[1]), int(self.key_hsv[2])

        # HSV Mask (255 = background key region, 0 = foreground person)
        low_s, high_s = max(0, ks - s_t), min(255, ks + s_t)
        low_v, high_v = max(0, kv - v_t), min(255, kv + v_t)

        # Handle Hue wrap-around in [0, 179]
        if kh - h_t < 0:
            m1 = cv2.inRange(frame_hsv, np.array([0, low_s, low_v]), np.array([kh + h_t, high_s, high_v]))
            m2 = cv2.inRange(frame_hsv, np.array([180 + (kh - h_t), low_s, low_v]), np.array([179, high_s, high_v]))
            hsv_bg_mask = cv2.bitwise_or(m1, m2)
        elif kh + h_t > 179:
            m1 = cv2.inRange(frame_hsv, np.array([kh - h_t, low_s, low_v]), np.array([179, high_s, high_v]))
            m2 = cv2.inRange(frame_hsv, np.array([0, low_s, low_v]), np.array([(kh + h_t) - 180, high_s, high_v]))
            hsv_bg_mask = cv2.bitwise_or(m1, m2)
        else:
            lower = np.array([kh - h_t, low_s, low_v], dtype=np.uint8)
            upper = np.array([kh + h_t, high_s, high_v], dtype=np.uint8)
            hsv_bg_mask = cv2.inRange(frame_hsv, lower, upper)

        # RGB-Distance Mask using 3D Euclidean Distance in NumPy
        frame_rgb = frame_bgr[:, :, ::-1].astype(np.float32)
        dist_sq = np.sum((frame_rgb - self.key_rgb) ** 2, axis=2)
        rgb_threshold = (h_t * 1.5 + (s_t + v_t) * 0.35)
        rgb_bg_mask = np.where(dist_sq <= (rgb_threshold ** 2), 255, 0).astype(np.uint8)

        return frame_hsv, hsv_bg_mask, rgb_bg_mask

    def _clean_mask_and_despill(self, frame_bgr, bg_mask):
        """
        Task 3:
        1. Remove specks with morphological open & close
        2. Feather the edge for soft alpha
        3. Suppress green spill on hair and edges
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned_bg = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned_bg = cv2.morphologyEx(cleaned_bg, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Foreground alpha: 1.0 = subject, 0.0 = background, intermediate at feathered edges
        fg_mask = 255 - cleaned_bg
        feathered = cv2.GaussianBlur(fg_mask, (7, 7), 0)
        alpha = (feathered.astype(np.float32) / 255.0)[:, :, None]

        # Green Spill Suppression in NumPy: cap green excess relative to Red & Blue
        b = frame_bgr[:, :, 0].astype(np.float32)
        g = frame_bgr[:, :, 1].astype(np.float32)
        r = frame_bgr[:, :, 2].astype(np.float32)

        rb_avg = 0.5 * (r + b)
        spill_mask = g > rb_avg
        # Neutralize green bounce especially on semi-transparent hair/edges
        g_suppressed = np.where(spill_mask, rb_avg * 0.85 + g * 0.15, g)

        despilled_bgr = np.stack([b, g_suppressed, r], axis=2).astype(np.uint8)
        return cleaned_bg, alpha, despilled_bgr

    def _alpha_composite(self, fg_bgr, bg_bgr, alpha):
        """Task 4: Per-pixel alpha compositing in NumPy."""
        comp = alpha * fg_bgr.astype(np.float32) + (1.0 - alpha) * bg_bgr.astype(np.float32)
        return np.clip(comp, 0, 255).astype(np.uint8)

    def _apply_numpy_filter(self, img_bgr):
        """
        Task 5: Filters written from scratch with NumPy (no built-in effects).
        Includes Grayscale (luminance weights), Sepia, Invert, Posterize, and CMYK separation.
        """
        if self.active_filter is None:
            return img_bgr

        b = img_bgr[:, :, 0].astype(np.float32)
        g = img_bgr[:, :, 1].astype(np.float32)
        r = img_bgr[:, :, 2].astype(np.float32)

        if self.active_filter == "Grayscale":
            # Luminance weights: Y = 0.299*R + 0.587*G + 0.114*B
            lum = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
            return np.stack([lum, lum, lum], axis=2)

        elif self.active_filter == "Sepia":
            tr = np.clip(0.393 * r + 0.769 * g + 0.189 * b, 0, 255).astype(np.uint8)
            tg = np.clip(0.349 * r + 0.686 * g + 0.168 * b, 0, 255).astype(np.uint8)
            tb = np.clip(0.272 * r + 0.534 * g + 0.131 * b, 0, 255).astype(np.uint8)
            return np.stack([tb, tg, tr], axis=2)

        elif self.active_filter == "Invert":
            return 255 - img_bgr

        elif self.active_filter == "Posterize":
            # Quantize into 4 discrete color levels per channel
            levels = 4
            step = 256 // levels
            return ((img_bgr // step) * step).astype(np.uint8)

        elif self.active_filter in ("C", "M", "Y", "K"):
            # Pure NumPy RGB -> CMYK conversion
            r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0
            k = 1.0 - np.maximum(np.maximum(r_norm, g_norm), b_norm)
            denom = np.where((1.0 - k) == 0, 1e-6, 1.0 - k)

            c = np.clip((1.0 - r_norm - k) / denom, 0.0, 1.0)
            m = np.clip((1.0 - g_norm - k) / denom, 0.0, 1.0)
            y = np.clip((1.0 - b_norm - k) / denom, 0.0, 1.0)

            zeros = np.zeros_like(c, dtype=np.uint8)
            full = np.full_like(c, 255, dtype=np.uint8)

            # Tinted plates (white paper minus ink) so each channel is visually clear
            c_plate = np.stack([full, full, (255 * (1.0 - c)).astype(np.uint8)], axis=2)
            m_plate = np.stack([full, (255 * (1.0 - m)).astype(np.uint8), full], axis=2)
            y_plate = np.stack([(255 * (1.0 - y)).astype(np.uint8), full, full], axis=2)
            k_val = (255 * (1.0 - k)).astype(np.uint8)
            k_plate = np.stack([k_val, k_val, k_val], axis=2)

            if self.cmyk_quad_mode:
                # 4-Channel CMYK Separation Grid View (Task 5 requirement)
                hw, hh = self.frame_w // 2, self.frame_h // 2
                c_sm = cv2.resize(c_plate, (hw, hh))
                m_sm = cv2.resize(m_plate, (hw, hh))
                y_sm = cv2.resize(y_plate, (hw, hh))
                k_sm = cv2.resize(k_plate, (hw, hh))

                cv2.putText(c_sm, "Cyan (C)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
                cv2.putText(m_sm, "Magenta (M)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
                cv2.putText(y_sm, "Yellow (Y)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
                cv2.putText(k_sm, "Key / Black (K)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 0), 2)

                top_row = np.hstack([c_sm, m_sm])
                bot_row = np.hstack([y_sm, k_sm])
                return np.vstack([top_row, bot_row])
            else:
                plates = {"C": c_plate, "M": m_plate, "Y": y_plate, "K": k_plate}
                return plates[self.active_filter]

        return img_bgr

    def _render_hue_histogram(self, frame_hsv, bg_mask):
        """Task 6: Live hue histogram of the keyed region."""
        hist_img = np.full((self.hist_h, self.hist_w, 3), 24, dtype=np.uint8)
        keyed_hues = frame_hsv[:, :, 0][bg_mask > 0]

        if keyed_hues.size == 0:
            keyed_hues = frame_hsv[:, :, 0].ravel()

        counts = np.bincount(keyed_hues, minlength=180)[:180].astype(np.float32)
        if counts.max() > 0:
            counts = (counts / counts.max()) * (self.hist_h - 14)

        for h_val in range(180):
            bar_h = int(counts[h_val])
            if bar_h <= 0:
                continue
            x0 = int(h_val * self.hist_w / 180)
            x1 = max(x0 + 1, int((h_val + 1) * self.hist_w / 180))
            color_bgr = cv2.cvtColor(
                np.uint8([[[h_val, 230, 230]]]), cv2.COLOR_HSV2BGR
            )[0][0].tolist()
            cv2.rectangle(
                hist_img, (x0, self.hist_h - bar_h), (x1, self.hist_h - 1), color_bgr, -1
            )

        # Draw marker line for selected key Hue
        kh_x = int(int(self.key_hsv[0]) * self.hist_w / 180)
        cv2.line(hist_img, (kh_x, 0), (kh_x, self.hist_h), (255, 255, 255), 1)

        hist_rgb = cv2.cvtColor(hist_img, cv2.COLOR_BGR2RGB)
        img_tk = ImageTk.PhotoImage(Image.fromarray(hist_rgb))
        self.hist_canvas.img_tk = img_tk
        self.hist_canvas.configure(image=img_tk)

    # =========================================================================
    # MAIN REAL-TIME LOOP
    # =========================================================================
    def _get_background_frame(self):
        if self.bg_video_cap is not None:
            ret, bg_frame = self.bg_video_cap.read()
            if not ret:
                self.bg_video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, bg_frame = self.bg_video_cap.read()
            if ret:
                self.bg_image = cv2.resize(bg_frame, (self.frame_w, self.frame_h))
        return self.bg_image

    def _update_loop(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (self.frame_w, self.frame_h))
            self.current_raw_frame = frame.copy()

            # Compute FPS (Task 1)
            now = time.time()
            dt = max(now - self.prev_time, 1e-6)
            self.fps = 0.85 * self.fps + 0.15 * (1.0 / dt)
            self.prev_time = now

            # Task 2 & 3: Masks, Morphological Cleanup, Feathering, Spill Suppression
            frame_hsv, hsv_bg_mask, rgb_bg_mask = self._compute_masks(frame)
            clean_hsv_bg, alpha_hsv, despilled_fg = self._clean_mask_and_despill(frame, hsv_bg_mask)
            self.current_clean_mask = clean_hsv_bg

            bg_plate = self._get_background_frame()

            # Select Display View Mode
            if self.view_mode == "Original":
                output_bgr = frame.copy()
            elif self.view_mode == "RGB":
                # Task 2 Requirement: Show HSV vs RGB-distance version side by side
                clean_rgb_bg, alpha_rgb, _ = self._clean_mask_and_despill(frame, rgb_bg_mask)
                comp_hsv = self._alpha_composite(despilled_fg, bg_plate, alpha_hsv)
                comp_rgb = self._alpha_composite(despilled_fg, bg_plate, alpha_rgb)

                hw = self.frame_w // 2
                left_half = cv2.resize(comp_hsv, (hw, self.frame_h))
                right_half = cv2.resize(comp_rgb, (hw, self.frame_h))
                cv2.putText(left_half, "HSV Keying", (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 255, 120), 2)
                cv2.putText(right_half, "RGB-Distance Keying", (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 200, 255), 2)
                output_bgr = np.hstack([left_half, right_half])
                cv2.line(output_bgr, (hw, 0), (hw, self.frame_h), (255, 255, 255), 2)
            else:
                # Default HSV Chroma Key Composite
                output_bgr = self._alpha_composite(despilled_fg, bg_plate, alpha_hsv)

            # Task 5: Apply Custom NumPy Filter
            output_bgr = self._apply_numpy_filter(output_bgr)

            # Task 4: Write to MP4 if recording
            if self.is_recording and self.video_writer is not None:
                self.video_writer.write(output_bgr)

            # HUD Overlay (FPS + Sampled Key Swatch)
            cv2.putText(
                output_bgr, f"FPS: {self.fps:.1f} | Mode: {self.view_mode}",
                (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 240, 245), 2, cv2.LINE_AA
            )
            key_c = tuple(int(c) for c in self.key_bgr)
            cv2.rectangle(output_bgr, (self.frame_w - 42, 10), (self.frame_w - 12, 36), key_c, -1)
            cv2.rectangle(output_bgr, (self.frame_w - 42, 10), (self.frame_w - 12, 36), (255, 255, 255), 1)

            # Update Main Video Feed
            out_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
            img_tk = ImageTk.PhotoImage(Image.fromarray(out_rgb))
            self.video_label.img_tk = img_tk
            self.video_label.configure(image=img_tk)

            # Task 6: Update Live Hue Histogram
            self._render_hue_histogram(frame_hsv, clean_hsv_bg)

        self.root.after(15, self._update_loop)

    def on_close(self):
        if self.video_writer is not None:
            self.video_writer.release()
        if self.bg_video_cap is not None:
            self.bg_video_cap.release()
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = GreenScreenStudio(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()