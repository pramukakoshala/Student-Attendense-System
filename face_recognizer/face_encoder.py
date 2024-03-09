# import the necessary packages
from face_recognizer.detect_faces import face_detection
import face_recognition
import pandas as pd
import pickle
import cv2
import os
from imutils import paths

class FaceEncoder():
    def __init__(self, facesPath, encodings, attendance, prototxt, model):
        self.facesPath = facesPath
        self.encodings = encodings
        self.attendance = attendance
        self.prototxt = prototxt
        self.model = model

        self.knowEncodings = []
        self.KnowNames = []
        
    def encode_faces(self):
        # Extract image paths and initialize empty encoding and names array
        image_paths = list(paths.list_images(self.facesPath))

        # Loop over the image paths
        for (i, image_path) in enumerate(image_paths):
            print(f"[INFO] processing image {i + 1}/{len(image_paths)}")
            name = image_path.split(os.path.sep)[-2]

            image = cv2.imread(image_path)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            (boxes, _) = face_detection(image, self.prototxt, self.model)
            boxes = [(box[1], box[2], box[3], box[0]) for box in boxes]
            encodings = face_recognition.face_encodings(rgb, boxes)

            for encoding in encodings:
                self.knowEncodings.append(encoding)
                self.KnowNames.append(name)

    def save_face_encodings(self):
        # Check if there are registered faces
        if os.path.exists(self.attendance):
            # Append new encodings to the existing one
            with open(self.encodings, "rb") as f:
                print("[INFO] loading encodings...")
                data = pickle.loads(f.read())

            data["names"].extend(self.KnowNames)
            data["encodings"].extend(self.knowEncodings)

            with open(self.encodings, "wb") as f:
                print("[INFO] serialize encodings to disk...")
                f.write(pickle.dumps(data))

            # Load existing attendance DataFrame
            df = pd.read_csv(self.attendance, index_col=0)
            new_df = pd.DataFrame({"names": list(set(self.KnowNames))})
            new_df = new_df.assign(**{col: 0 for col in df.columns if col != 'names'})

            # Combine the existing and new DataFrames while avoiding duplicates
            df_combined = pd.concat([df, new_df]).drop_duplicates(subset=['names']).reset_index(drop=True)
            df_combined = df_combined.sort_values(by='names')

            print("[INFO] storing additional student names in a dataframe...")
            df_combined.to_csv(self.attendance)
        else:
            # Serialize encodings to disk for the first time
            print("[INFO] serialize encodings to disk...")
            data = {"names": self.KnowNames, "encodings": self.knowEncodings}
            with open(self.encodings, "wb") as f:
                f.write(pickle.dumps(data))

            # Storing the names in a DataFrame for the first time
            print("[INFO] storing student names in a dataframe...")
            df_new = pd.DataFrame({"names": sorted(list(set(self.KnowNames)))})
            df_new.to_csv(self.attendance)

if __name__ == "__main__":
    # Example usage:
    facesPath = "/path/to/faces"
    encodingsPath = "/path/to/encodings.pickle"
    attendancePath = "/path/to/attendance.csv"
    prototxtPath = "/path/to/deploy.prototxt"
    modelPath = "/path/to/res10_300x300_ssd_iter_140000.caffemodel"

    encoder = FaceEncoder(facesPath, encodingsPath, attendancePath, prototxtPath, modelPath)
    encoder.encode_faces()
    encoder.save_face_encodings()
