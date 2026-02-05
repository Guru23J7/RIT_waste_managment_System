"""
CleanSort AI - Hybrid Waste Classification System
Combines computer vision AI with sensor data for intelligent waste segregation
"""

import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import cv2


class HybridWasteClassifier:
    """
    Hybrid decision-making system that combines AI predictions with sensor data.
    
    Priority Logic:
    1. Metal detector triggered → Metal Bin (95% confidence)
    2. AI confidence > 85% → Use AI prediction
    3. Moisture > 60% → Wet Waste Bin (80% confidence)
    4. AI confidence 50-85% → Cross-verify with sensors and boost confidence
    5. AI confidence < 50% → Reject for manual sorting
    """
    
    def __init__(self):
        self.ai_category = None
        self.ai_confidence = 0.0
        self.moisture_level = 0.0  # 0-100%
        self.metal_detected = False
        
        # Waste categories
        self.CATEGORIES = {
            'plastic': 'Recyclable - Plastic',
            'paper': 'Recyclable - Paper',
            'metal': 'Recyclable - Metal',
            'glass': 'Recyclable - Glass',
            'organic': 'Organic - Wet Waste'
        }
        
        # Bin mappings
        self.BIN_MAP = {
            'plastic': 'RECYCLABLE BIN (Plastic)',
            'paper': 'RECYCLABLE BIN (Paper)',
            'metal': 'RECYCLABLE BIN (Metal)',
            'glass': 'RECYCLABLE BIN (Glass)',
            'organic': 'WET WASTE BIN'
        }
    
    def set_ai_prediction(self, category, confidence):
        """Set AI prediction results"""
        self.ai_category = category
        self.ai_confidence = confidence
    
    def set_sensor_data(self, moisture_level=0.0, metal_detected=False):
        """Set sensor data readings"""
        self.moisture_level = moisture_level
        self.metal_detected = metal_detected
    
    def decide(self):
        """
        Make final decision based on hybrid logic
        
        Returns:
            dict: {
                'category': waste category,
                'bin': recommended bin,
                'method': decision method used,
                'confidence': final confidence,
                'reasoning': explanation of decision
            }
        """
        
        # Priority 1: Metal detector has highest priority
        if self.metal_detected:
            return {
                'category': 'metal',
                'bin': self.BIN_MAP['metal'],
                'method': 'SENSOR (Metal Detector)',
                'confidence': 95,
                'reasoning': 'Metal detector triggered - direct to metal bin'
            }
        
        # Priority 2: High confidence AI (>85%)
        if self.ai_confidence > 85:
            return {
                'category': self.ai_category,
                'bin': self.BIN_MAP.get(self.ai_category, 'MANUAL SORTING'),
                'method': 'AI (High Confidence)',
                'confidence': self.ai_confidence,
                'reasoning': f'AI confidence {self.ai_confidence:.1f}% - reliable prediction'
            }
        
        # Priority 3: High moisture indicates organic waste
        if self.moisture_level > 60:
            return {
                'category': 'organic',
                'bin': self.BIN_MAP['organic'],
                'method': 'SENSOR (Moisture)',
                'confidence': 80,
                'reasoning': f'High moisture detected ({self.moisture_level:.1f}%) - likely organic waste'
            }
        
        # Priority 4: Medium confidence AI (50-85%) - cross-verify with sensors
        if 50 <= self.ai_confidence <= 85:
            # Boost confidence if sensors support AI prediction
            boosted_confidence = self.ai_confidence
            reasoning_parts = [f'AI prediction: {self.ai_confidence:.1f}%']
            
            # Check if moisture supports organic classification
            if self.ai_category == 'organic' and self.moisture_level > 30:
                boosted_confidence = min(90, self.ai_confidence + 15)
                reasoning_parts.append(f'moisture sensor confirms ({self.moisture_level:.1f}%)')
            
            # Check if low moisture supports non-organic classification
            elif self.ai_category != 'organic' and self.moisture_level < 30:
                boosted_confidence = min(90, self.ai_confidence + 10)
                reasoning_parts.append('low moisture confirms dry waste')
            
            return {
                'category': self.ai_category,
                'bin': self.BIN_MAP.get(self.ai_category, 'MANUAL SORTING'),
                'method': 'HYBRID (AI + Sensors)',
                'confidence': boosted_confidence,
                'reasoning': ' + '.join(reasoning_parts)
            }
        
        # Priority 5: Low confidence - reject for manual sorting
        return {
            'category': 'unknown',
            'bin': 'MANUAL SORTING REQUIRED',
            'method': 'REJECTED',
            'confidence': self.ai_confidence,
            'reasoning': f'Confidence too low ({self.ai_confidence:.1f}%) - needs human verification'
        }


class WasteClassifier:
    """AI-based waste classifier using MobileNetV2"""
    
    def __init__(self):
        print("Loading MobileNetV2 model...")
        self.model = MobileNetV2(weights='imagenet', include_top=True)
        print("Model loaded successfully!")
        
        # ImageNet label mappings to waste categories
        self.LABEL_MAPPINGS = {
            # Plastic
            'water_bottle': 'plastic',
            'pop_bottle': 'plastic',
            'pill_bottle': 'plastic',
            'soap_dispenser': 'plastic',
            'plastic_bag': 'plastic',
            'bottle': 'plastic',
            
            # Glass
            'beer_bottle': 'glass',
            'wine_bottle': 'glass',
            'vase': 'glass',
            'jar': 'glass',
            
            # Paper
            'envelope': 'paper',
            'notebook': 'paper',
            'carton': 'paper',
            'book': 'paper',
            'menu': 'paper',
            'newspaper': 'paper',
            'tissue': 'paper',
            
            # Metal
            'tin_can': 'metal',
            'beer_can': 'metal',
            'can_opener': 'metal',
            'soup_can': 'metal',
            
            # Organic
            'banana': 'organic',
            'orange': 'organic',
            'apple': 'organic',
            'lemon': 'organic',
            'strawberry': 'organic',
            'pineapple': 'organic',
            'pomegranate': 'organic',
            'custard_apple': 'organic',
            'mango': 'organic',
            'guacamole': 'organic',
            'salad': 'organic',
            'pizza': 'organic',
            'bagel': 'organic',
        }
    
    def preprocess_image(self, img):
        """Preprocess image for MobileNetV2"""
        # Resize to 224x224
        img_resized = cv2.resize(img, (224, 224))
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        # Expand dimensions
        img_array = np.expand_dims(img_rgb, axis=0)
        # Preprocess for MobileNetV2
        img_preprocessed = preprocess_input(img_array)
        return img_preprocessed
    
    def classify_image(self, img):
        """
        Classify waste item in image
        
        Args:
            img: OpenCV image (BGR format)
        
        Returns:
            tuple: (category, confidence, raw_label)
        """
        # Preprocess
        processed_img = self.preprocess_image(img)
        
        # Predict
        predictions = self.model.predict(processed_img, verbose=0)
        
        # Decode top 3 predictions
        decoded = decode_predictions(predictions, top=3)[0]
        
        # Try to map top predictions to waste categories
        for _, label, confidence in decoded:
            label_lower = label.lower().replace('_', ' ')
            
            # Check exact matches
            for key, category in self.LABEL_MAPPINGS.items():
                if key.replace('_', ' ') in label_lower or label_lower in key.replace('_', ' '):
                    return category, float(confidence * 100), label
        
        # No match found - return top prediction with low confidence marker
        top_label = decoded[0][1]
        top_conf = float(decoded[0][2] * 100)
        
        # Make educated guesses based on keywords
        top_label_lower = top_label.lower()
        if any(word in top_label_lower for word in ['bottle', 'container', 'jug']):
            return 'plastic', top_conf * 0.6, top_label  # Reduced confidence
        elif any(word in top_label_lower for word in ['can', 'tin']):
            return 'metal', top_conf * 0.6, top_label
        elif any(word in top_label_lower for word in ['food', 'fruit', 'vegetable']):
            return 'organic', top_conf * 0.6, top_label
        elif any(word in top_label_lower for word in ['paper', 'book', 'card']):
            return 'paper', top_conf * 0.6, top_label
        
        return 'unknown', top_conf * 0.3, top_label  # Very low confidence for unknown


def demo_hybrid_logic():
    """Demonstrate hybrid decision logic with test cases"""
    
    print("\n" + "="*70)
    print("CLEANSORT AI - HYBRID DECISION LOGIC DEMO")
    print("="*70)
    
    classifier = HybridWasteClassifier()
    
    test_cases = [
        {
            'name': 'Test 1: Metal Detector Triggered',
            'ai_category': 'plastic',
            'ai_confidence': 75,
            'moisture': 10,
            'metal': True,
            'description': 'Metal detector overrides AI prediction'
        },
        {
            'name': 'Test 2: High Confidence AI',
            'ai_category': 'plastic',
            'ai_confidence': 92,
            'moisture': 5,
            'metal': False,
            'description': 'AI confidence >85% - trust the AI'
        },
        {
            'name': 'Test 3: High Moisture Sensor',
            'ai_category': 'paper',
            'ai_confidence': 60,
            'moisture': 75,
            'metal': False,
            'description': 'High moisture overrides medium AI confidence'
        },
        {
            'name': 'Test 4: Hybrid Cross-Verification',
            'ai_category': 'organic',
            'ai_confidence': 70,
            'moisture': 45,
            'metal': False,
            'description': 'AI + moisture sensor boost confidence'
        },
        {
            'name': 'Test 5: Low Confidence Rejection',
            'ai_category': 'plastic',
            'ai_confidence': 35,
            'moisture': 15,
            'metal': False,
            'description': 'Too uncertain - manual sorting required'
        },
    ]
    
    for test in test_cases:
        print(f"\n{test['name']}")
        print("-" * 70)
        print(f"Description: {test['description']}")
        print(f"\nInputs:")
        print(f"  AI Prediction: {test['ai_category']} ({test['ai_confidence']}% confidence)")
        print(f"  Moisture Level: {test['moisture']}%")
        print(f"  Metal Detected: {'YES' if test['metal'] else 'NO'}")
        
        classifier.set_ai_prediction(test['ai_category'], test['ai_confidence'])
        classifier.set_sensor_data(test['moisture'], test['metal'])
        
        decision = classifier.decide()
        
        print(f"\nDecision:")
        print(f"  Category: {decision['category'].upper()}")
        print(f"  Bin: {decision['bin']}")
        print(f"  Method: {decision['method']}")
        print(f"  Final Confidence: {decision['confidence']:.1f}%")
        print(f"  Reasoning: {decision['reasoning']}")
    
    print("\n" + "="*70 + "\n")


def webcam_test():
    """Simple webcam test - capture and classify on keypress"""
    
    print("\n" + "="*70)
    print("CLEANSORT AI - WEBCAM CLASSIFIER TEST")
    print("="*70)
    print("\nInstructions:")
    print("  - Press SPACE to capture and classify")
    print("  - Press 'q' to quit")
    print("  - Hold item in front of camera for best results")
    print("\nStarting webcam...")
    
    # Initialize classifier
    classifier = WasteClassifier()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Cannot access webcam!")
        print("Trying alternative camera index...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("ERROR: No webcam detected!")
            return
    
    print("Webcam ready! Show waste item and press SPACE\n")
    
    classification_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("ERROR: Failed to capture frame")
            break
        
        # Display frame
        display_frame = frame.copy()
        cv2.putText(display_frame, "Press SPACE to classify | Q to quit", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('CleanSort AI - Webcam Test', display_frame)
        
        # Wait for key
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space key
            print(f"\n--- Classification #{classification_count + 1} ---")
            print("Analyzing image...")
            
            # Classify
            category, confidence, raw_label = classifier.classify_image(frame)
            
            classification_count += 1
            
            print(f"Raw Prediction: {raw_label}")
            print(f"Waste Category: {category.upper()}")
            print(f"Confidence: {confidence:.2f}%")
            
            if confidence > 70:
                print("Status: ✓ High confidence")
            elif confidence > 50:
                print("Status: ~ Medium confidence (sensor verification recommended)")
            else:
                print("Status: ✗ Low confidence (manual sorting suggested)")
            
            # Show result on frame
            result_frame = frame.copy()
            cv2.putText(result_frame, f"Category: {category.upper()}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(result_frame, f"Confidence: {confidence:.1f}%", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Classification Result', result_frame)
            cv2.waitKey(2000)  # Show for 2 seconds
            cv2.destroyWindow('Classification Result')
        
        elif key == ord('q'):  # Quit
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n{classification_count} items classified")
    print("Webcam test completed!\n")


if __name__ == '__main__':
    print("\nCleanSort AI - Waste Segregation System")
    print("=======================================\n")
    print("Choose an option:")
    print("1. Run Hybrid Logic Demo")
    print("2. Run Webcam Test")
    print("3. Both")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == '1':
        demo_hybrid_logic()
    elif choice == '2':
        webcam_test()
    elif choice == '3':
        demo_hybrid_logic()
        webcam_test()
    else:
        print("Invalid choice!")
