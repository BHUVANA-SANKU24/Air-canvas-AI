# Air Canvas AI

AI-powered air canvas with object recognition and math equation solving.

## Features
- ✅ Hand tracking with MediaPipe
- ✅ Draw in the air using gestures
- 🚧 Object recognition from sketches
- 🚧 Math equation solver

## Setup

1. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
python app.py
```

## Gestures
- **Index finger**: Draw
- **Index + Middle finger**: Switch mode (Object/Math)
- **3 fingers up**: Clear canvas
- **Closed fist**: Erase
- **Press 'h'**: Toggle help
- **Press 'q'**: Quit

## Project Status
- [x] Phase 1: Air Canvas with hand tracking
- [ ] Phase 2: Object recognition
- [ ] Phase 3: Math solver

## Author
Built with ❤️ using Python, OpenCV, and MediaPipe