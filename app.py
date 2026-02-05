"""
CleanSort AI - Flask Web Application
Real-time waste classification dashboard with webcam streaming
"""

from flask import Flask, render_template, Response, jsonify
import cv2
import threading
import time
import random
from cleansort_ai import WasteClassifier, HybridWasteClassifier

app = Flask(__name__)

# Global variables
classifier = None
hybrid_classifier = HybridWasteClassifier()
camera = None
current_prediction = {
    'category': 'unknown',
    'confidence': 0,
    'bin': 'Waiting for item...',
    'method': 'N/A',
    'reasoning': 'No item detected yet'
}
stats = {
    'total': 0,
    'plastic': 0,
    'paper': 0,
    'metal': 0,
    'glass': 0,
    'organic': 0
}
frame_count = 0
classification_interval = 20  # Classify every 20 frames for performance
lock = threading.Lock()


def load_model():
    """Load MobileNetV2 model (runs in background)"""
    global classifier
    print("Initializing AI model...")
    classifier = WasteClassifier()
    print("Model ready!")


def init_camera():
    """Initialize webcam"""
    global camera
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        print("Warning: Primary camera not found, trying alternative...")
        camera = cv2.VideoCapture(1)
        
        if not camera.isOpened():
            print("ERROR: No webcam detected!")
            return False
    
    # Set camera properties for better performance
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)
    
    print("Webcam initialized successfully!")
    return True


def simulate_sensor_data(category, confidence):
    """
    Simulate sensor readings based on waste type
    In real implementation, this would read from Arduino sensors
    """
    # Simulate moisture sensor (0-100%)
    if category == 'organic':
        moisture = random.uniform(50, 90)  # High moisture for organic
    elif category == 'paper':
        moisture = random.uniform(5, 25)   # Low moisture for paper
    else:
        moisture = random.uniform(0, 20)   # Very low for plastic/metal/glass
    
    # Simulate metal detector
    metal_detected = (category == 'metal' and confidence > 60)
    
    return moisture, metal_detected


def classify_frame(frame):
    """Classify waste item in frame using hybrid logic"""
    global current_prediction, stats
    
    if classifier is None:
        return
    
    try:
        # Get AI prediction
        category, confidence, raw_label = classifier.classify_image(frame)
        
        # Simulate sensor data (in real setup, read from Arduino)
        moisture, metal_detected = simulate_sensor_data(category, confidence)
        
        # Use hybrid classifier for final decision
        hybrid_classifier.set_ai_prediction(category, confidence)
        hybrid_classifier.set_sensor_data(moisture, metal_detected)
        decision = hybrid_classifier.decide()
        
        # Update current prediction
        with lock:
            # Only update if this is a real detection (not unknown with very low confidence)
            if decision['confidence'] > 30 or decision['category'] != 'unknown':
                current_prediction = {
                    'category': decision['category'],
                    'confidence': decision['confidence'],
                    'bin': decision['bin'],
                    'method': decision['method'],
                    'reasoning': decision['reasoning'],
                    'raw_label': raw_label,
                    'ai_confidence': confidence,
                    'moisture': moisture,
                    'metal_detected': metal_detected
                }
                
                # Update stats only for successful classifications
                if decision['confidence'] > 50 and decision['category'] != 'unknown':
                    stats['total'] += 1
                    if decision['category'] in stats:
                        stats[decision['category']] += 1
    
    except Exception as e:
        print(f"Classification error: {e}")


def generate_frames():
    """Generate video frames with classification overlay"""
    global frame_count
    
    while True:
        if camera is None or not camera.isOpened():
            break
        
        success, frame = camera.read()
        if not success:
            break
        
        frame_count += 1
        
        # Classify every N frames to optimize performance
        if classifier is not None and frame_count % classification_interval == 0:
            classify_frame(frame.copy())
        
        # Draw overlay on frame
        with lock:
            pred = current_prediction.copy()
        
        # Create overlay
        overlay = frame.copy()
        
        # Semi-transparent background for text
        cv2.rectangle(overlay, (10, 10), (630, 140), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Add text information
        category_text = pred['category'].upper() if pred['category'] != 'unknown' else 'SCANNING...'
        confidence = pred['confidence']
        
        # Color code based on confidence
        if confidence > 75:
            color = (0, 255, 0)  # Green
        elif confidence > 50:
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 0, 255)  # Red
        
        # Draw text
        cv2.putText(frame, f"Category: {category_text}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Confidence: {confidence:.1f}%", 
                   (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Bin: {pred['bin']}", 
                   (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add live indicator
        if frame_count % 30 < 15:  # Blinking effect
            cv2.circle(frame, (610, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "LIVE", (550, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        
        # Yield frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.033)  # ~30 fps


@app.route('/')
def index():
    """Render dashboard"""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/prediction')
def get_prediction():
    """Get current prediction as JSON"""
    with lock:
        return jsonify(current_prediction)


@app.route('/stats')
def get_stats():
    """Get classification statistics as JSON"""
    with lock:
        return jsonify(stats)


@app.route('/reset_stats')
def reset_stats():
    """Reset statistics"""
    global stats
    with lock:
        stats = {
            'total': 0,
            'plastic': 0,
            'paper': 0,
            'metal': 0,
            'glass': 0,
            'organic': 0
        }
    return jsonify({'status': 'success'})


def cleanup():
    """Cleanup resources"""
    global camera
    if camera is not None:
        camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print("\n" + "="*70)
    print("CLEANSORT AI - WASTE SEGREGATION DASHBOARD")
    print("="*70 + "\n")
    
    # Initialize camera
    print("Step 1: Initializing webcam...")
    if not init_camera():
        print("FATAL ERROR: Cannot start without webcam!")
        exit(1)
    
    # Load model in background thread
    print("Step 2: Loading AI model (this may take a minute)...")
    model_thread = threading.Thread(target=load_model, daemon=True)
    model_thread.start()
    
    print("\nStarting Flask server...")
    print("\n" + "="*70)
    print("Dashboard ready! Open your browser and navigate to:")
    print("    http://localhost:5000")
    print("="*70 + "\n")
    print("Press Ctrl+C to stop the server\n")
    
    try:
        # Run Flask app
        app.run(debug=False, threaded=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        cleanup()
        print("CleanSort AI stopped. Goodbye!")
