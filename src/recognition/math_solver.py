import re
import ast
import operator
import easyocr
import cv2


class MathSolver:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=False)

    def preprocess(self, img, w=400, h=200):
        img = cv2.resize(img, (w, h))
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)[1]
        return img

    def ocr(self, cropped_img, fast=False):
        if cropped_img is None:
            return None

        if fast:
            img = self.preprocess(cropped_img, 300, 160)
        else:
            img = self.preprocess(cropped_img, 420, 220)

        results = self.reader.readtext(img, detail=0)
        if not results:
            return None

        txt = " ".join(results).strip()
        return txt if txt else None

    def clean_expr(self, text: str):
        text = text.lower()
        text = text.replace("×", "*").replace("x", "*")
        text = text.replace("÷", "/")
        text = text.replace("=", "")
        text = re.sub(r"[^0-9\+\-\*\/\(\)\.]", "", text)
        return text.strip()

    def safe_eval(self, expr: str):
        ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
        }

        def _eval(node):
            if isinstance(node, ast.Num):
                return node.n
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                return -_eval(node.operand)
            if isinstance(node, ast.BinOp):
                if type(node.op) not in ops:
                    raise ValueError("bad operator")
                return ops[type(node.op)](_eval(node.left), _eval(node.right))
            raise ValueError("bad expr")

        tree = ast.parse(expr, mode="eval")
        return _eval(tree.body)

    def solve(self, cropped_img):
        raw = self.ocr(cropped_img, fast=False)
        if raw is None:
            return None, None, None

        expr = self.clean_expr(raw)
        if not expr:
            return raw, None, None

        try:
            ans = self.safe_eval(expr)
            return raw, expr, ans
        except Exception:
            return raw, expr, None
