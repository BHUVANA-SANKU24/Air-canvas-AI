import cv2
import numpy as np
import textwrap
import time
import re

from src.hand_tracking import HandTracker
from src.canvas import DrawingCanvas
from src.recognition.object_classifier import ObjectClassifier
from src.recognition.math_solver import MathSolver
from src.recognition.web_qa import WebQA


class AirCanvasApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        ret, frame = self.cap.read()
        if ret:
            self.h, self.w, _ = frame.shape
        else:
            self.w, self.h = 640, 480

        self.hand_tracker = HandTracker(
            model_path="models/hand_landmarker.task")
        self.canvas = DrawingCanvas(width=self.w, height=self.h)

        self.object_model = ObjectClassifier()
        self.math_solver = MathSolver()
        self.webqa = WebQA()

        self.mode = "Detect Object"
        self.dropdown_open = False

        self.header_height = 135
        self.window_name = "Air Canvas AI"
        self.buttons = {}

        # results
        self.result_text = ""
        self.result_lines = []

        # submit
        self.submit_frames = 0
        self.submit_threshold = 8

        # colors
        self.colors = [
            ("WHITE", (255, 255, 255)),
            ("RED", (0, 0, 255)),
            ("GREEN", (0, 255, 0)),
            ("BLUE", (255, 0, 0)),
            ("YELLOW", (0, 255, 255)),
        ]

        # OCR preview
        self.last_ocr_time = 0
        self.ocr_preview = ""

        # Q&A state
        self.generated_question = None

    # ---------------- text cleaning ----------------
    def clean_text(self, s: str):
        """Prevent garbage in cv2.putText. Keep only ASCII printable."""
        if not s:
            return ""
        s = s.strip()
        s = s.replace("\n", " ").replace("\t", " ")
        s = re.sub(r"\s+", " ", s)
        s = s.encode("ascii", errors="ignore").decode("ascii")
        s = re.sub(r"[?]{2,}", "", s)
        return s.strip()

    def wrap_lines(self, text, width=65, max_lines=5):
        text = self.clean_text(text)
        return textwrap.wrap(text, width=width)[:max_lines]

    def in_box(self, x, y, box):
        x1, y1, x2, y2 = box
        return x1 <= x <= x2 and y1 <= y <= y2

    def set_result(self, text):
        self.result_text = self.clean_text(text)
        self.result_lines = self.wrap_lines(
            self.result_text, width=70, max_lines=5)

    # ---------------- mouse callback ----------------
    def mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # dropdown click
        if self.in_box(x, y, self.buttons["dropdown"]):
            self.dropdown_open = not self.dropdown_open
            return

        # dropdown options
        if self.dropdown_open:
            for name, box in self.buttons["dropdown_options"]:
                if self.in_box(x, y, box):
                    self.mode = name
                    self.dropdown_open = False
                    self.canvas.clear()
                    self.generated_question = None
                    self.set_result(f"Mode selected: {name}")
                    return
            self.dropdown_open = False

        # colors
        for cname, col, box in self.buttons["colors"]:
            if self.in_box(x, y, box):
                self.canvas.set_color(col)
                self.set_result(f"Color selected: {cname}")
                return

        # tools
        if self.in_box(x, y, self.buttons["brush"]):
            self.canvas.set_tool("brush")
            self.set_result("Tool: BRUSH")
            return

        if self.in_box(x, y, self.buttons["eraser"]):
            self.canvas.set_tool("eraser")
            self.set_result("Tool: ERASER")
            return

        if self.in_box(x, y, self.buttons["clear"]):
            self.canvas.clear()
            self.generated_question = None
            self.set_result("Canvas cleared")
            return

    # ---------------- UI ----------------
    def draw_button(self, frame, box, text, fill=(60, 60, 60)):
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), fill, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
        cv2.putText(frame, text, (x1 + 8, y2 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def draw_dropdown_menu(self, frame, dropdown_box):
        """Draw dropdown options AFTER panel -> always on top."""
        options = ["Detect Object", "Detect Math", "Detect Q&A"]
        option_boxes = []

        if not self.dropdown_open:
            self.buttons["dropdown_options"] = []
            return

        option_h = 32
        total_h = option_h * len(options)

        drop_x1, _, drop_x2, drop_y2 = dropdown_box

        down_y0 = drop_y2 + 5
        up_y0 = dropdown_box[1] - total_h - 5

        if down_y0 + total_h < self.h:
            start_y = down_y0
        else:
            start_y = max(5, up_y0)

        for i, opt in enumerate(options):
            y1 = start_y + i * option_h
            y2 = y1 + (option_h - 2)
            box = (drop_x1, y1, drop_x2, y2)

            cv2.rectangle(frame, (box[0], box[1]),
                          (box[2], box[3]), (20, 20, 20), -1)
            cv2.rectangle(frame, (box[0], box[1]),
                          (box[2], box[3]), (255, 255, 255), 1)
            cv2.putText(frame, opt, (box[0] + 8, box[3] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            option_boxes.append((opt, box))

        self.buttons["dropdown_options"] = option_boxes

    def draw_ui(self, frame):
        cv2.rectangle(frame, (0, 0), (self.w, self.header_height),
                      (35, 35, 35), -1)

        # MODE label
        cv2.putText(frame, "MODE:", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

        # dropdown
        drop_x1 = 110
        drop_x2 = min(self.w - 20, 360)
        dropdown_box = (drop_x1, 15, drop_x2, 45)
        self.draw_button(frame, dropdown_box, self.mode, fill=(50, 50, 50))
        self.buttons["dropdown"] = dropdown_box
        cv2.putText(frame, "v", (dropdown_box[2] - 18, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # COLOR label
        cv2.putText(frame, "COLOR:", (380, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)

        # colors row
        color_boxes = []
        x0, y0 = 380, 15
        bw, bh, gap = 50, 25, 8
        for i, (cname, col) in enumerate(self.colors):
            x1 = x0 + i * (bw + gap)
            if x1 + bw > self.w - 10:
                break
            box = (x1, y0, x1 + bw, y0 + bh)
            cv2.rectangle(frame, box[:2], box[2:], col, -1)
            cv2.rectangle(frame, box[:2], box[2:], (255, 255, 255), 1)
            color_boxes.append((cname, col, box))
        self.buttons["colors"] = color_boxes

        # tools
        brush_box = (380, 50, 460, 75)
        eraser_box = (470, 50, 560, 75)
        clear_box = (570, 50, 640, 75)

        self.draw_button(frame, brush_box, "BRUSH", fill=(70, 70, 70))
        self.draw_button(frame, eraser_box, "ERASER", fill=(70, 70, 70))
        self.draw_button(frame, clear_box, "CLEAR", fill=(0, 0, 150))

        self.buttons["brush"] = brush_box
        self.buttons["eraser"] = eraser_box
        self.buttons["clear"] = clear_box

        # OCR preview or Q&A question
        if self.mode == "Detect Q&A" and self.generated_question:
            prev = self.clean_text(self.generated_question)
            cv2.putText(frame, f"Q: {prev[:55]}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.58, (100, 255, 255), 2)
        elif self.mode in ["Detect Math", "Detect Q&A"]:
            prev = self.clean_text(self.ocr_preview)
            cv2.putText(frame, f"OCR: {prev[:55]}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.58, (255, 255, 255), 1)
        else:
            self.ocr_preview = ""

        # result panel
        panel_x1, panel_y1 = 20, 95
        panel_x2, panel_y2 = self.w - 20, self.header_height - 10
        cv2.rectangle(frame, (panel_x1, panel_y1),
                      (panel_x2, panel_y2), (18, 18, 18), -1)
        cv2.rectangle(frame, (panel_x1, panel_y1),
                      (panel_x2, panel_y2), (100, 100, 100), 1)

        y = panel_y1 + 22
        for line in self.result_lines:
            cv2.putText(frame, line, (panel_x1 + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2)
            y += 22

        # draw dropdown OVER panel
        self.draw_dropdown_menu(frame, dropdown_box)

        cv2.putText(frame, "FIST = SUBMIT | Q: Quit",
                    (20, self.h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2)

        return frame

    # ---------------- prediction ----------------
    def run_prediction(self):
        cropped = self.canvas.get_cropped_drawing()

        if self.mode == "Detect Object":
            label, conf = self.object_model.predict(cropped)
            if label is None:
                self.set_result("Draw an object first!")
            elif conf == 0.0:
                self.set_result(label)  # Error message from classifier
            else:
                confidence_pct = int(conf * 100)
                self.set_result(f"It's a {label} ({confidence_pct}%)")

        elif self.mode == "Detect Math":
            raw, expr, ans = self.math_solver.solve(cropped)
            if raw is None:
                self.set_result("Write math like: 2+2=")
            elif ans is None:
                self.set_result(f"OCR: {raw} | Could not solve")
            else:
                self.set_result(f"{expr} = {ans}")

        elif self.mode == "Detect Q&A":
            raw = self.math_solver.ocr(cropped, fast=False)
            raw = self.clean_text(raw if raw else "")

            if not raw:
                self.set_result("Write any keyword/question and show FIST.")
                self.generated_question = None
            else:
                # Generate question and search
                self.set_result("Searching web...")
                cv2.waitKey(100)  # Show searching message briefly

                question, answer = self.webqa.answer(raw)
                self.generated_question = question

                if answer:
                    # Combine question and answer for display
                    full_text = f"Q: {question}\nA: {answer}"
                    self.set_result(full_text)
                else:
                    self.set_result(f"Q: {question}\nNo answer found.")

    # ---------------- run loop ----------------
    def run(self):
        print("=" * 50)
        print("AIR CANVAS AI - ENHANCED")
        print("=" * 50)
        print("Features:")
        print("  ✅ Object Detection (expanded)")
        print("  ✅ Math Solver")
        print("  ✅ Q&A with smart question generation")
        print("\nGestures:")
        print("  - Index finger: Draw")
        print("  - Fist: Submit")
        print("  - Press Q: Quit")
        print("=" * 50)

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        frame_count = 0

        while True:
            ok, frame = self.cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)

            frame = self.hand_tracker.find_hands(
                frame, True, int(frame_count * 33))
            frame_count += 1

            self.hand_tracker.get_position(frame)
            gesture = self.hand_tracker.get_gesture()
            pos = self.hand_tracker.get_fingertip_position(8, smooth=True)

            # draw
            if gesture == "draw" and pos:
                x, y = pos
                if y > self.header_height:
                    if not self.canvas.drawing:
                        self.canvas.start_stroke(x, y)
                    else:
                        self.canvas.add_point(x, y)
            else:
                if self.canvas.drawing:
                    self.canvas.end_stroke()

            # live OCR preview every 1 sec
            now = time.time()
            if self.mode in ["Detect Math", "Detect Q&A"]:
                if now - self.last_ocr_time > 1.0:
                    self.last_ocr_time = now
                    cropped = self.canvas.get_cropped_drawing()
                    txt = self.math_solver.ocr(cropped, fast=True)
                    self.ocr_preview = txt if txt else ""

            # submit
            if gesture == "submit":
                self.submit_frames += 1
                if self.submit_frames == self.submit_threshold:
                    self.run_prediction()
            else:
                self.submit_frames = 0

            # overlay drawing
            canvas_img = self.canvas.get_canvas()
            mask = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2GRAY)
            mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]

            alpha = 0.6
            for c in range(3):
                frame[:, :, c] = np.where(
                    mask == 255,
                    canvas_img[:, :, c] * alpha + frame[:, :, c] * (1 - alpha),
                    frame[:, :, c]
                )

            frame = self.draw_ui(frame)
            cv2.imshow(self.window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Air Canvas AI closed. Thank you!")


if __name__ == "__main__":
    AirCanvasApp().run()
