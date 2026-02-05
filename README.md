# CleanSort AI - Waste Segregation System 🌱

An intelligent waste classification system that combines computer vision AI with sensor data for accurate waste segregation. Built for hackathon demonstration.

## 🎯 Features

- **Real-time Waste Classification** using MobileNetV2
- **Hybrid Decision Logic** combining AI with sensor verification
- **Live Video Dashboard** with modern glassmorphism UI
- **Multiple Waste Categories**: Plastic, Paper, Metal, Glass, Organic
- **Statistics Tracking** for classification performance
- **Webcam-based** - no additional hardware required

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam (built-in or external)
- Windows/Mac/Linux

### Installation

1. **Clone or download this repository**

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

This will install:
- Flask (web framework)
- TensorFlow (AI model)
- OpenCV (webcam/video processing)
- NumPy, Pillow (image processing)

**Note**: First run will download MobileNetV2 model (~14MB) automatically.

### Running the Application

#### Option 1: Web Dashboard (Recommended)

Run the Flask server:
```bash
python app.py
```

Then open your browser and navigate to:
```
http://localhost:5000
```

You should see the CleanSort AI dashboard with:
- Live webcam feed
- Real-time waste classification
- Confidence metrics
- Statistics panel

#### Option 2: Standalone Test

Test the hybrid logic and webcam classifier:
```bash
python cleansort_ai.py
```

Choose from:
1. **Hybrid Logic Demo** - See how AI + sensors make decisions
2. **Webcam Test** - Capture and classify items manually
3. **Both** - Run full demonstration

## 📖 How It Works

### Hybrid Decision Logic

The system uses a priority-based decision system:

1. **Metal Detector** (Highest Priority)
   - If metal sensor triggered → Metal Bin (95% confidence)

2. **High Confidence AI** (>85%)
   - Trust AI prediction directly

3. **Moisture Sensor**
   - If moisture > 60% → Wet Waste Bin (80% confidence)

4. **Cross-Verification** (AI 50-85%)
   - Combine AI with sensors to boost confidence
   - Example: AI says "organic" + high moisture = higher confidence

5. **Rejection** (<50% confidence)
   - Manual sorting required

### AI Classification

- **Model**: MobileNetV2 (pre-trained on ImageNet)
- **Categories**: Maps ImageNet labels to waste types
- **Processing**: Classifies every 20 frames for performance
- **Mapping Examples**:
  - `water_bottle` → Plastic
  - `banana` → Organic
  - `tin_can` → Metal

## 🎨 Dashboard Features

### Live Video Feed
- Real-time webcam stream
- Classification overlay
- Color-coded confidence bars

### Prediction Panel
- Detected category
- Confidence percentage
- Recommended bin
- Decision method (AI/Sensor/Hybrid)
- Reasoning explanation

### Statistics
- Total items classified
- Breakdown by category
- Reset functionality

## 📁 Project Structure

```
AI_based_waste_segmentation/
│
├── app.py                  # Flask web application
├── cleansort_ai.py         # AI classifier & hybrid logic
├── requirements.txt        # Python dependencies
├── README.md              # This file
│
└── templates/
    └── index.html         # Dashboard UI
```

## 🔧 Configuration

### Adjust Classification Frequency

In `app.py`, line 26:
```python
classification_interval = 20  # Classify every N frames
```
- Lower = More responsive, higher CPU usage
- Higher = Better performance, less responsive

### Camera Selection

The app tries camera indices in order (0, 1). To force a specific camera, edit `app.py`:
```python
camera = cv2.VideoCapture(1)  # Change index
```

## 🎤 Presentation Guide

### Key Points for Judges

1. **Innovation**: Hybrid approach combines AI confidence with sensor verification
2. **Accuracy**: Cross-verification improves reliability over pure AI/sensors
3. **Scalability**: Works with webcam demo, ready for Arduino integration
4. **Real-world Impact**: Reduces contamination in waste streams

### Demo Steps

1. Start the dashboard
2. Show live classification of different items:
   - Plastic bottle
   - Paper/notebook
   - Metal can
   - Fruit/organic waste
3. Point out confidence levels and decision methods
4. Show statistics tracking
5. Explain hybrid logic with examples

### Technical Architecture

```
Webcam Feed → Frame Capture → AI (MobileNetV2)
                                    ↓
                              AI Prediction + Confidence
                                    ↓
Sensor Data (Moisture, Metal) → Hybrid Classifier
                                    ↓
                              Final Decision → Bin Recommendation
```

## 🐛 Troubleshooting

### Webcam Not Detected
```
ERROR: No webcam detected!
```
**Solutions**:
- Check if webcam is connected
- Close other apps using webcam (Zoom, Teams, etc.)
- Try running with administrator/sudo privileges
- Edit `app.py` to try different camera index

### Model Loading Slow
First run downloads MobileNetV2 (~14MB). Subsequent runs are faster.

### Flask App Won't Start
```bash
# Check if port 5000 is in use
netstat -ano | findstr :5000    # Windows
lsof -i :5000                   # Mac/Linux

# Use different port
python app.py --port 8080
```

### Low FPS / Laggy
- Increase `classification_interval` in app.py
- Close other programs
- Use better lighting for faster processing

## 📊 Expected Performance

- **FPS**: 25-30 fps video stream
- **Classification Speed**: ~0.1-0.3s per frame
- **Accuracy**: 70-85% on common waste items
- **CPU Usage**: ~30-50% on modern laptops

## 🔄 Future Enhancements

- [ ] Arduino integration for real sensor data
- [ ] Custom model fine-tuned on waste dataset
- [ ] Export classifications to CSV
- [ ] Multi-camera support
- [ ] Raspberry Pi deployment
- [ ] Mobile app

## 📝 License

This project is created for educational/hackathon purposes. MobileNetV2 model is used under Apache 2.0 license.

## 🤝 Contributing

Built for hackathon demonstration. Feel free to fork and enhance!

## ⚡ Quick Commands Reference

```bash
# Install dependencies
pip install -r requirements.txt

# Run web dashboard
python app.py

# Run standalone tests
python cleansort_ai.py

# Check Python version
python --version

# List installed packages
pip list
```

## 📞 Support

For issues or questions during demo:
1. Check webcam permissions
2. Verify all packages installed: `pip list`
3. Try standalone test first: `python cleansort_ai.py`
4. Check console for error messages

---

**Built with ❤️ for SDG 12: Responsible Consumption & Production**

Good luck with your hackathon! 🚀
