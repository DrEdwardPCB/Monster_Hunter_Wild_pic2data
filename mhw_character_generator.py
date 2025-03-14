import cv2
import numpy as np
import dlib
import argparse
from PIL import Image
import json
import os


class MHWCharacterGenerator:
    def __init__(self):
        # Load face detection and landmark prediction models
        self.face_detector = dlib.get_frontal_face_detector()

        # You'll need to download this file from dlib's website
        # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        model_path = "shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(model_path):
            print(f"Error: {model_path} not found.")
            print(
                "Please download it from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            )
            print("Extract it and place it in the same directory as this script.")
            exit(1)

        self.landmark_predictor = dlib.shape_predictor(model_path)

        # Define feature mappings (these are approximations, adjust based on the actual game)
        self.face_shapes = ["Round", "Oval", "Square", "Heart", "Diamond"]
        self.eye_colors = ["Brown", "Blue", "Green", "Gray", "Amber", "Hazel"]
        self.skin_tones = ["Very Light", "Light", "Medium", "Tan", "Dark", "Very Dark"]
        self.hair_colors = [
            "Black",
            "Dark Brown",
            "Brown",
            "Light Brown",
            "Blonde",
            "Red",
        ]

    def _load_image(self, image_path):
        """Load image with robust error handling and format support"""
        # First try with PIL/Pillow (supports more formats)
        try:
            # Open with PIL and force conversion to RGB
            pil_img = Image.open(image_path)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            
            # Convert to numpy array
            img_np = np.array(pil_img)
            
            # Ensure it's 8-bit
            if img_np.dtype != np.uint8:
                img_np = (img_np / img_np.max() * 255).astype(np.uint8)
                
            # Convert RGB to BGR (OpenCV format)
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # Verify image dimensions and channels
            if len(img_cv.shape) != 3 or img_cv.shape[2] != 3:
                raise ValueError("Image must have 3 channels (BGR)")
            
            # Print image properties for debugging
            print(f"Loaded image via PIL: shape={img_cv.shape}, dtype={img_cv.dtype}, min={img_cv.min()}, max={img_cv.max()}")
                
            return img_cv
            
        except Exception as pil_error:
            print(f"PIL could not load image properly, trying OpenCV: {pil_error}")
            
            # Fallback to OpenCV
            try:
                # Read with OpenCV directly
                img = cv2.imread(image_path, cv2.IMREAD_COLOR)
                
                if img is None:
                    # Try alternative decoding
                    with open(image_path, 'rb') as f:
                        img_data = f.read()
                    img_np = np.frombuffer(img_data, np.uint8)
                    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                    
                    if img is None:
                        raise ValueError("Failed to decode image data")
                
                # Ensure we have the right format
                if len(img.shape) != 3 or img.shape[2] != 3:
                    # If image is grayscale, convert to BGR
                    if len(img.shape) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    else:
                        raise ValueError("Image has an unsupported number of channels")
                
                # Ensure it's 8-bit
                if img.dtype != np.uint8:
                    img = (img / img.max() * 255).astype(np.uint8)
                    
                # Check image dimensions
                if img.shape[0] <= 0 or img.shape[1] <= 0:
                    raise ValueError("Image has invalid dimensions")
                
                # Print image properties for debugging
                print(f"Loaded image via OpenCV: shape={img.shape}, dtype={img.dtype}, min={img.min()}, max={img.max()}")
                    
                return img
                
            except Exception as cv_error:
                print(f"All image loading methods failed: {cv_error}")
                raise ValueError(f"Could not load image in a format compatible with dlib")

       
    def extract_features(self, image_path):
        """Extract facial features from an image"""
        # Load image with enhanced image loading
        img = self._load_image(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")

        # Print image details for debugging
        print(f"Image shape: {img.shape}, dtype: {img.dtype}")

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Save the grayscale image for verification
        cv2.imwrite(f"{os.path.splitext(image_path)[0]}_gray.jpg", gray)
        print(f"Saved grayscale image for debugging")

        # Initialize face variable
        face = None

        # Detect faces using dlib
        try:
            # First try with the image as is
            faces = self.face_detector(gray)

            if len(faces) == 0:
                # If no faces detected, try upsampling the image
                print("No faces detected with default settings, trying upsampling...")
                faces = self.face_detector(gray, 1)  # 1 = upsample once

            if len(faces) == 0:
                raise ValueError("No faces detected in the image")

            # Use the first face detected
            face = faces[0]

        except Exception as e:
            print(f"Error during face detection: {e}")
            print("Trying alternative approach with OpenCV cascade...")

            # Fallback to OpenCV's face detector if dlib fails
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            opencv_faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(opencv_faces) == 0:
                raise ValueError("No faces detected with any method")

            # Convert OpenCV face to dlib rectangle format
            x, y, w, h = opencv_faces[0]
            face = dlib.rectangle(left=x, top=y, right=x + w, bottom=y + h)
            print("Successfully detected face using OpenCV cascade")

        # Ensure we have a face
        if face is None:
            raise ValueError("No face was detected after all attempts")

        # Save a debug image with the face rectangle
        debug_path = f"{os.path.splitext(image_path)[0]}_face_debug.jpg"
        debug_img = img.copy()
        cv2.rectangle(
            debug_img,
            (face.left(), face.top()),
            (face.right(), face.bottom()),
            (0, 255, 0),
            2,
        )
        cv2.imwrite(debug_path, debug_img)
        print(f"Saved debug image to {debug_path}")

        # Get facial landmarks
        landmarks = self.landmark_predictor(gray, face)

        # Extract features
        features = {}
        print(1)
        # Face shape analysis
        features["face_shape"] = self._analyze_face_shape(landmarks)
        print(2)
        # Face proportions
        features["face_proportions"] = self._analyze_face_proportions(landmarks)
        print(3)
        # Eye analysis
        features["eye_details"] = self._analyze_eyes(img, landmarks)
        print(4)
        # Nose analysis
        features["nose_details"] = self._analyze_nose(landmarks)
        print(5)
        # Mouth analysis
        features["mouth_details"] = self._analyze_mouth(landmarks)
        print(6)
        # Skin tone
        features["skin_tone"] = self._analyze_skin_tone(img, landmarks)
        print(7)
        # Hair color
        features["hair_color"] = self._analyze_hair_color(img, landmarks)

        return features

    def _analyze_face_shape(self, landmarks):
        """Determine face shape from landmarks"""
        print("Analyzing face shape...")
        # Convert landmarks to numpy array for easier processing
        points = np.zeros((68, 2), dtype=np.float32)
        for i in range(68):
            points[i] = (landmarks.part(i).x, landmarks.part(i).y)

        # Measure face width and height
        face_width = np.linalg.norm(points[0] - points[16])
        face_height = np.linalg.norm(points[8] - points[27])
        jaw_width = np.linalg.norm(points[2] - points[14])
        forehead_width = np.linalg.norm(points[0] - points[16])

        # Determine face shape based on proportions
        ratio = face_width / face_height
        jaw_to_forehead_ratio = jaw_width / forehead_width

        # Simplified face shape classification
        if ratio > 0.9 and ratio < 1.1:
            if jaw_to_forehead_ratio < 0.9:
                shape_index = 2  # "Square"
            else:
                shape_index = 0  # "Round"
        elif ratio < 0.85:
            if jaw_to_forehead_ratio < 0.8:
                shape_index = 3  # "Heart"
            else:
                shape_index = 4  # "Diamond"
        else:
            shape_index = 1  # "Oval"
        print("finish face shape")
        return {
            "shape_name": self.face_shapes[shape_index],
            "shape_index": shape_index,
            "width_height_ratio": float(ratio),
            "jaw_forehead_ratio": float(jaw_to_forehead_ratio),
        }

    def _analyze_face_proportions(self, landmarks):
        """Analyze facial feature proportions"""
        print("Analyzing facial proportions...")
        # Convert landmarks to numpy array
        points = np.zeros((68, 2), dtype=np.float32)
        for i in range(68):
            points[i] = (landmarks.part(i).x, landmarks.part(i).y)

        # Calculate various facial proportions
        # These can be mapped to sliders in the character creator

        # Eye distance to face width ratio
        eye_distance = np.linalg.norm(points[39] - points[42])
        face_width = np.linalg.norm(points[0] - points[16])
        eye_ratio = eye_distance / face_width

        # Nose width to face width ratio
        nose_width = np.linalg.norm(points[31] - points[35])
        nose_ratio = nose_width / face_width

        # Mouth width to face width ratio
        mouth_width = np.linalg.norm(points[48] - points[54])
        mouth_ratio = mouth_width / face_width

        # Chin length
        chin_length = np.linalg.norm(points[8] - points[57])
        chin_to_face_height = chin_length / np.linalg.norm(points[8] - points[27])
        print("finish facial proportions")
        return {
            "eye_distance": float(eye_ratio * 100),  # Scale to 0-100 for game sliders
            "nose_width": float(nose_ratio * 100),
            "mouth_width": float(mouth_ratio * 100),
            "chin_length": float(chin_to_face_height * 100),
        }

    def _analyze_eyes(self, img, landmarks):
        """Analyze eye features"""
        print("Analyzing eye features...")
        # Convert landmarks to numpy array
        points = np.zeros((68, 2), dtype=np.float32)
        for i in range(68):
            points[i] = (landmarks.part(i).x, landmarks.part(i).y)

        # Eye color detection
        # This uses a very simplified approach - for more accuracy,
        # a dedicated eye color classifier would be better
        left_eye_center = (
            (points[37] + points[38] + points[40] + points[41]) / 4
        ).astype(int)
        right_eye_center = (
            (points[43] + points[44] + points[46] + points[47]) / 4
        ).astype(int)

        # Sample pixels around eye centers
        def get_eye_color_sample(center):
            x, y = center
            # Make sure we don't go out of bounds
            h, w = img.shape[:2]
            if y < 3 or y >= h - 3 or x < 3 or x >= w - 3:
                return np.array([0, 0, 0])

            roi = img[y - 3 : y + 3, x - 3 : x + 3]
            if roi.size == 0:
                return np.array([0, 0, 0])
            # Get average color in BGR
            return np.mean(roi, axis=(0, 1))

        left_color = get_eye_color_sample(left_eye_center)
        right_color = get_eye_color_sample(right_eye_center)

        # Average the colors from both eyes
        avg_color = (left_color + right_color) / 2
        b, g, r = avg_color

        # Simple rule-based eye color classification
        # This is a very rough approximation
        if r > 60 and g > 60 and b > 60:
            if b > r and b > g:
                color_index = 1  # "Blue"
            elif g > r and g > b:
                color_index = 2  # "Green"
            elif max(r, g, b) - min(r, g, b) < 20:
                color_index = 3  # "Gray"
            elif r > g and r > b and g > b:
                color_index = 4  # "Amber"
            elif r > g and r > b and g > 100:
                color_index = 5  # "Hazel"
            else:
                color_index = 0  # "Brown"
        else:
            color_index = 0  # "Brown"

        # Eye shape
        left_eye_width = np.linalg.norm(points[36] - points[39])
        left_eye_height = np.linalg.norm(
            (points[37] + points[38]) / 2 - (points[40] + points[41]) / 2
        )
        right_eye_width = np.linalg.norm(points[42] - points[45])
        right_eye_height = np.linalg.norm(
            (points[43] + points[44]) / 2 - (points[46] + points[47]) / 2
        )

        # Average the measurements
        eye_width = (left_eye_width + right_eye_width) / 2
        eye_height = (left_eye_height + right_eye_height) / 2
        eye_ratio = eye_height / eye_width
        print("finish eye features")
        return {
            "eye_color": self.eye_colors[color_index],
            "eye_color_index": color_index,
            "eye_rgb": [int(r), int(g), int(b)],
            "eye_roundness": float(eye_ratio * 100),
            "eye_size": float(eye_width * 50),  # Scale appropriately
        }

    def _analyze_nose(self, landmarks):
        """Analyze nose features"""
        print("Analyzing nose features...")
        # Convert landmarks to numpy array
        points = np.zeros((68, 2), dtype=np.float32)
        for i in range(68):
            points[i] = (landmarks.part(i).x, landmarks.part(i).y)

        # Nose measurements
        nose_width = np.linalg.norm(points[31] - points[35])
        nose_length = np.linalg.norm(points[27] - points[30])
        nose_bridge_width = np.linalg.norm(points[21] - points[22])

        # Calculate nose proportions relative to face
        face_width = np.linalg.norm(points[0] - points[16])
        face_height = np.linalg.norm(points[8] - points[27])

        # Nose tip position (how pointy or flat)
        nose_tip_angle = (
            np.arctan2(points[30][1] - points[27][1], points[30][0] - points[27][0])
            * 180
            / np.pi
        )
        print("finish nose features")
        return {
            "nose_width": float((nose_width / face_width) * 100),
            "nose_length": float((nose_length / face_height) * 100),
            "nose_bridge_width": float((nose_bridge_width / face_width) * 100),
            "nose_tip_angle": float(nose_tip_angle),
        }

    def _analyze_mouth(self, landmarks):
        """Analyze mouth features"""
        print("Analyzing mouth features...")
        # Convert landmarks to numpy array
        points = np.zeros((68, 2), dtype=np.float32)
        for i in range(68):
            points[i] = (landmarks.part(i).x, landmarks.part(i).y)

        # Mouth measurements
        mouth_width = np.linalg.norm(points[48] - points[54])
        mouth_height = np.linalg.norm(
            (points[50] + points[52]) / 2 - (points[56] + points[58]) / 2
        )

        # Calculate mouth proportions relative to face
        face_width = np.linalg.norm(points[0] - points[16])

        # Lip thickness
        upper_lip_height = np.linalg.norm(points[50] - points[52])
        lower_lip_height = np.linalg.norm(points[56] - points[58])

        # Mouth curve (smile line)
        center_upper = (points[51][1] + points[62][1]) / 2
        corner_avg = (points[48][1] + points[54][1]) / 2
        curve = center_upper - corner_avg

        print("finish mouth features")
        return {
            "mouth_width": float((mouth_width / face_width) * 100),
            "mouth_height": float(mouth_height * 10),
            "upper_lip_thickness": float(upper_lip_height * 20),
            "lower_lip_thickness": float(lower_lip_height * 20),
            "mouth_curve": float(curve),
        }

    def _analyze_skin_tone(self, img, landmarks):
        """Analyze skin tone"""
        # Convert landmarks to numpy array
        print("Analyzing skin tone...")
        points = np.zeros((68, 2), dtype=np.float32)
        for i in range(68):
            points[i] = (landmarks.part(i).x, landmarks.part(i).y)

        # Sample skin tone from a few points on the face
        # (forehead, cheeks, chin)
        forehead = (
            int((points[19][0] + points[24][0]) / 2),
            int((points[19][1] + points[24][1]) / 2) - 10,
        )
        left_cheek = (int(points[1][0]), int(points[1][1]))
        right_cheek = (int(points[15][0]), int(points[15][1]))
        chin = (int(points[8][0]), int(points[8][1]))

        # Get average BGR values at these points
        def get_sample(point):
            x, y = point
            # Make sure we don't go out of bounds
            h, w = img.shape[:2]
            if y < 0 or y >= h or x < 0 or x >= w:
                return np.array([0, 0, 0])

            return img[y, x]

        samples = [
            get_sample(point) for point in [forehead, left_cheek, right_cheek, chin]
        ]
        samples = [s for s in samples if not np.array_equal(s, [0, 0, 0])]

        if not samples:  # If all samples were out of bounds
            avg_color = np.array([128, 128, 128])  # Default to gray
        else:
            avg_color = np.mean(samples, axis=0)

        # Convert BGR to HSV for better skin tone analysis
        hsv = cv2.cvtColor(np.uint8([[avg_color]]), cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = hsv

        # Classify skin tone (simplified)
        # This is a very rough approximation
        if v < 100:
            tone_index = 5  # "Very Dark"
        elif v < 130:
            tone_index = 4  # "Dark"
        elif v < 160:
            tone_index = 3  # "Tan"
        elif v < 190:
            tone_index = 2  # "Medium"
        elif v < 220:
            tone_index = 1  # "Light"
        else:
            tone_index = 0  # "Very Light"
        print("finish skin tone")
        return {
            "skin_tone": self.skin_tones[tone_index],
            "skin_tone_index": tone_index,
            "skin_rgb": [int(avg_color[2]), int(avg_color[1]), int(avg_color[0])],
        }

    def _analyze_hair_color(self, img, landmarks):
        """Analyze hair color"""
        print("Analyzing hair color...")
        # This is challenging without specific hair segmentation
        # Simplified approach: sample pixels above the forehead

        # Convert landmarks to numpy array
        points = np.zeros((68, 2), dtype=np.float32)
        for i in range(68):
            points[i] = (landmarks.part(i).x, landmarks.part(i).y)

        # Get forehead region
        forehead_center = ((points[19] + points[24]) / 2).astype(int)
        # Sample area above forehead
        hair_sample_y = max(0, forehead_center[1] - 30)
        hair_sample_x = forehead_center[0]

        # Make sure we don't go out of bounds
        h, w = img.shape[:2]
        if hair_sample_y >= h or hair_sample_x >= w:
            return {
                "hair_color": "Brown",  # Default
                "hair_color_index": 2,
                "hair_rgb": [100, 70, 40],
            }

        # Sample a small region
        hair_region_start_y = max(0, hair_sample_y)
        hair_region_end_y = min(h, hair_sample_y + 20)
        hair_region_start_x = max(0, hair_sample_x - 20)
        hair_region_end_x = min(w, hair_sample_x + 20)

        hair_region = img[
            hair_region_start_y:hair_region_end_y, hair_region_start_x:hair_region_end_x
        ]

        # If we can't sample hair (e.g., image is cropped too tight)
        if hair_region.size == 0:
            return {
                "hair_color": "Brown",  # Default
                "hair_color_index": 2,
                "hair_rgb": [100, 70, 40],
            }

        # Get average color
        avg_color = np.mean(hair_region, axis=(0, 1))
        b, g, r = avg_color

        # Classify hair color using simple rules
        # This is a rough approximation
        if r < 50 and g < 50 and b < 50:
            color_index = 0  # "Black"
        elif r < 80 and g < 60 and b < 60:
            color_index = 1  # "Dark Brown"
        elif r < 120 and g < 100 and b < 100:
            color_index = 2  # "Brown"
        elif r < 150 and g < 120 and b < 100:
            color_index = 3  # "Light Brown"
        elif r > 150 and g > 120 and b < 100:
            color_index = 4  # "Blonde"
        elif r > 100 and g < 80 and b < 80:
            color_index = 5  # "Red"
        else:
            color_index = 2  # Default to "Brown"
        print("finish hair")
        return {
            "hair_color": self.hair_colors[color_index],
            "hair_color_index": color_index,
            "hair_rgb": [int(r), int(g), int(b)],
        }

    def generate_mhw_parameters(self, features):
        """
        Convert extracted features to Monster Hunter Wild character creation parameters
        Note: These mappings are approximations and may need adjustments
        based on the actual game parameters
        """
        # Map features to game parameters
        mhw_params = {
            # Base parameters
            "face_type": features["face_shape"]["shape_index"],
            "skin_color": features["skin_tone"]["skin_tone_index"],
            "skin_color_rgb": features["skin_tone"]["skin_rgb"],
            # Face proportions
            "face_width": min(
                100, max(0, int(features["face_shape"]["width_height_ratio"] * 50))
            ),
            "jaw_width": min(
                100, max(0, int(features["face_shape"]["jaw_forehead_ratio"] * 50))
            ),
            "chin_height": min(
                100, max(0, int(features["face_proportions"]["chin_length"]))
            ),
            # Eyes
            "eye_type": min(
                10, max(0, int(features["eye_details"]["eye_roundness"] / 10))
            ),
            "eye_position_vertical": 50,  # Default middle
            "eye_position_horizontal": min(
                100, max(0, int(features["face_proportions"]["eye_distance"]))
            ),
            "eye_size": min(100, max(0, int(features["eye_details"]["eye_size"]))),
            "eye_color": features["eye_details"]["eye_color_index"],
            "eye_color_rgb": features["eye_details"]["eye_rgb"],
            # Nose
            "nose_type": min(
                10, max(0, int(abs(features["nose_details"]["nose_tip_angle"]) / 10))
            ),
            "nose_height": 50,  # Default middle
            "nose_width": min(100, max(0, int(features["nose_details"]["nose_width"]))),
            "nose_length": min(
                100, max(0, int(features["nose_details"]["nose_length"]))
            ),
            # Mouth
            "mouth_type": min(
                10, max(0, int(abs(features["mouth_details"]["mouth_curve"] * 5) + 5))
            ),
            "mouth_width": min(
                100, max(0, int(features["mouth_details"]["mouth_width"]))
            ),
            "mouth_height": min(
                100, max(0, int(features["mouth_details"]["mouth_height"]))
            ),
            "upper_lip_thickness": min(
                100, max(0, int(features["mouth_details"]["upper_lip_thickness"]))
            ),
            "lower_lip_thickness": min(
                100, max(0, int(features["mouth_details"]["lower_lip_thickness"]))
            ),
            # Hair
            "hair_color": features["hair_color"]["hair_color_index"],
            "hair_color_rgb": features["hair_color"]["hair_rgb"],
            # Default values for parameters we can't easily determine
            "hair_style": 0,  # Would need a more advanced hair classifier
            "eyebrow_type": 0,
            "makeup": 0,
            "facial_hair": 0,
            "scars": 0,
            "age": 0,
        }

        return mhw_params

    def process_image(self, image_path, output_path=None):
        """
        Main method to process an image and generate character parameters

        Args:
            image_path: Path to the input image
            output_path: Path to save the output JSON (if None, will use image_path + ".json")

        Returns:
            Dictionary of MHW character parameters
        """
        try:
            # Extract features
            features = self.extract_features(image_path)

            # Convert to game parameters
            mhw_params = self.generate_mhw_parameters(features)

            # Save to JSON
            if output_path is None:
                output_path = f"{os.path.splitext(image_path)[0]}_mhw_params.json"

            with open(output_path, "w") as f:
                json.dump(mhw_params, f, indent=2)

            print(f"Character parameters saved to {output_path}")

            # Also return the parameters
            return mhw_params

        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Generate Monster Hunter Wild character parameters from a face image"
    )
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Path to save the output JSON"
    )

    # Parse arguments
    args = parser.parse_args()

    # Create generator and process image
    generator = MHWCharacterGenerator()
    generator.process_image(args.image_path, args.output)


if __name__ == "__main__":
    main()
