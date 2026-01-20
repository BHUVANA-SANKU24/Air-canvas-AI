import os
import numpy as np
import cv2


class ObjectClassifier:
    def __init__(self, model_path="models/quickdraw_model.h5", confidence_threshold=0.55):
        # Expanded label list
        self.labels = [
            "car", "tree", "house", "cat", "bicycle", "dog", "flower",
            "bird", "fish", "airplane", "boat", "train", "circle",
            "square", "triangle", "star", "heart", "apple", "banana",
            "chair", "table", "cup", "book", "computer", "phone"
        ]

        # ✅ FIX 2: Increased confidence threshold to 0.55 (55%)
        # This prevents low-confidence misclassifications like triangle->cat
        self.confidence_threshold = confidence_threshold
        self.model = None

        if os.path.exists(model_path):
            try:
                from tensorflow.keras.models import load_model
                self.model = load_model(model_path)
                print(f"✅ QuickDraw model loaded: {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading model: {e}")
                self.model = None
        else:
            print(f"⚠️ Model not found at {model_path}")
            print("   Using geometric shape detection")

    def detect_basic_shapes(self, img):
        """
        ✅ FIX 2: IMPROVED shape detection - more accurate
        Returns: (shape_name, confidence)
        """
        # Find contours
        contours, _ = cv2.findContours(
            img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, 0.0

        # Get largest contour
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)

        if area < 150:  # Too small
            return None, 0.0

        # Approximate the contour
        peri = cv2.arcLength(cnt, True)
        # More precise approximation
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        vertices = len(approx)

        # Calculate circularity
        circularity = 4 * np.pi * area / (peri * peri) if peri > 0 else 0

        # Get bounding box for aspect ratio
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h if h > 0 else 0

        # ✅ FIX 2: More robust shape detection with stricter rules

        # Triangle - 3 vertices
        if vertices == 3:
            return "triangle", 0.85

        # Rectangle/Square - 4 vertices
        elif vertices == 4:
            if 0.90 <= aspect_ratio <= 1.10:
                return "square", 0.85
            else:
                return "rectangle", 0.80

        # Pentagon - 5 vertices
        elif vertices == 5:
            if circularity > 0.65:
                return "pentagon", 0.75
            else:
                return "star", 0.70

        # Hexagon - 6 vertices
        elif vertices == 6:
            return "hexagon", 0.75

        # Circle - high circularity and many vertices
        elif vertices > 8 and circularity > 0.75:
            return "circle", 0.90

        # Oval/Ellipse
        elif vertices > 6 and 0.55 < circularity < 0.75:
            if 0.5 < aspect_ratio < 2.0:
                return "oval", 0.70

        # Unknown shape
        return "geometric shape", 0.50

    def analyze_drawing_complexity(self, img):
        """
        Check if drawing is valid for classification
        """
        non_zero = cv2.countNonZero(img)
        contours, _ = cv2.findContours(
            img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        total_area = img.shape[0] * img.shape[1]
        fill_ratio = non_zero / total_area if total_area > 0 else 0
        num_contours = len(contours)

        # Too simple
        if fill_ratio < 0.015 or non_zero < 180:
            return False, "Drawing too simple. Draw larger."

        # Too messy
        if fill_ratio > 0.65 or num_contours > 25:
            return False, "Drawing unclear. Redraw cleaner."

        return True, None

    def preprocess(self, img):
        """Preprocess image for model"""
        img = cv2.resize(img, (28, 28))
        img = img.astype("float32") / 255.0
        img = img.reshape(1, 28, 28, 1)
        return img

    def predict(self, cropped_img):
        """
        ✅ FIX 2: IMPROVED prediction logic - shape detection takes priority
        """
        if cropped_img is None:
            return None, 0.0

        # Ensure clean binary image
        cropped_img = cv2.threshold(cropped_img, 30, 255, cv2.THRESH_BINARY)[1]

        # Validate drawing
        is_valid, error_msg = self.analyze_drawing_complexity(cropped_img)
        if not is_valid:
            return error_msg, 0.0

        # ✅ FIX 2: Try SHAPE detection FIRST (more reliable)
        shape, shape_conf = self.detect_basic_shapes(cropped_img)

        # If shape detection is confident, use it
        if shape and shape_conf >= 0.70:
            # Check if it's a basic geometric shape
            basic_shapes = ["triangle", "square", "rectangle", "circle",
                            "pentagon", "hexagon", "oval", "star"]

            if any(s in shape.lower() for s in basic_shapes):
                # For geometric shapes, trust shape detection more
                return shape, shape_conf

        # Try model prediction for complex objects
        if self.model is not None:
            try:
                x = self.preprocess(cropped_img)
                pred = self.model.predict(x, verbose=0)[0]
                idx = int(np.argmax(pred))
                conf = float(pred[idx])
                model_label = self.labels[idx]

                # ✅ FIX 2: Only trust model if confidence > threshold
                if conf >= self.confidence_threshold:
                    # If model says it's a complex object (not basic shape), use model
                    basic_shapes = ["triangle",
                                    "square", "rectangle", "circle"]
                    if model_label not in basic_shapes:
                        return model_label, conf
                    # If model says basic shape, compare with shape detection
                    elif shape and shape_conf > conf:
                        return shape, shape_conf
                    else:
                        return model_label, conf

                # Low confidence from model - use shape detection if available
                if shape and shape_conf >= 0.65:
                    return shape, shape_conf

                # Very low confidence - warn user
                return f"{model_label} (uncertain - {int(conf*100)}%)", conf

            except Exception as e:
                print(f"⚠️ Model error: {e}")
                # Fall through to shape detection

        # Final fallback - use shape detection
        if shape:
            return shape, shape_conf
        else:
            return "Cannot recognize. Try redrawing.", 0.0
