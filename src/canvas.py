import numpy as np
import cv2


class DrawingCanvas:
    def __init__(self, width=640, height=480, brush_color=(255, 255, 255), brush_thickness=8):
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)

        self.brush_color = brush_color
        self.brush_thickness = brush_thickness

        self.prev_x, self.prev_y = None, None
        self.drawing = False

        self.stroke_history = []
        self.current_stroke = []

        self.tool = "brush"  # brush or eraser

    def set_color(self, color):
        self.brush_color = color

    def set_tool(self, tool_name):
        if tool_name in ["brush", "eraser"]:
            self.tool = tool_name

    def start_stroke(self, x, y):
        self.drawing = True
        self.prev_x, self.prev_y = x, y
        self.current_stroke = [(x, y)]

    def add_point(self, x, y):
        if not self.drawing:
            return

        if self.prev_x is None or self.prev_y is None:
            self.prev_x, self.prev_y = x, y
            return

        if self.tool == "eraser":
            color = (0, 0, 0)
            thickness = self.brush_thickness * 3
        else:
            color = self.brush_color
            thickness = self.brush_thickness

        cv2.line(self.canvas, (self.prev_x, self.prev_y),
                 (x, y), color, thickness)

        self.current_stroke.append((x, y))
        self.prev_x, self.prev_y = x, y

    def end_stroke(self):
        if self.drawing and len(self.current_stroke) > 0:
            self.stroke_history.append({
                'points': self.current_stroke.copy(),
                'color': self.brush_color,
                'thickness': self.brush_thickness
            })
        self.drawing = False
        self.current_stroke = []
        self.prev_x, self.prev_y = None, None

    def clear(self):
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.stroke_history = []
        self.current_stroke = []
        self.prev_x, self.prev_y = None, None

    def get_canvas(self):
        return self.canvas

    def get_canvas_grayscale(self):
        return cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)

    def get_bounding_box(self):
        gray = self.get_canvas_grayscale()
        coords = cv2.findNonZero(gray)

        if coords is not None and len(coords) > 0:
            x, y, w, h = cv2.boundingRect(coords)
            return x, y, w, h
        return None

    def get_cropped_drawing(self, padding=20):
        bbox = self.get_bounding_box()
        if bbox is None:
            return None

        x, y, w, h = bbox

        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(self.width - x, w + 2 * padding)
        h = min(self.height - y, h + 2 * padding)

        gray = self.get_canvas_grayscale()
        cropped = gray[y:y + h, x:x + w]
        return cropped
