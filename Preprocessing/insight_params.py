from pathlib import Path

import cv2
import insightface
import matplotlib.pyplot as plt
import numpy as np
from insightface.app import FaceAnalysis

# Initialize face analysis model
app = FaceAnalysis(
    name="buffalo_l", providers=["CPUExecutionProvider"]
)  # 'CUDAExecutionProvider' for GPU
app.prepare(ctx_id=-1)  # ctx_id=-1 for CPU, 0 for GPU


def get_face_embedding(image_path):
    """Extract face embedding from an image"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    faces = app.get(img)

    if len(faces) < 1:
        raise ValueError("No faces detected in the image")
    if len(faces) > 1:
        print("Warning: Multiple faces detected. Using first detected face")

    face = faces[0]
    print("Detected bbox:", face.bbox)  # [x1, y1, x2, y2]
    print("2D landmarks (106):", face.landmark_2d_106)  # list of (x,y) coords
    # Optional features
    print("Age:", face.age, "Gender:", face.gender)  # attribute estimation
    return face.embedding  # 512-dim numpy array


# def compare_faces(
#     emb1, emb2, threshold=0.65
# ):  # Adjust this threshold according to your usecase.
#     """Compare two embeddings using cosine similarity"""
#     similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
#     return similarity, similarity > threshold


# PATH = "video_assets/video_frame/Braley_21_147_306/Braley_21_147_306_0000.bmp"
# image1_path = PATH

if __name__ == "main":

    INPUT_ROOT = Path("video_assets/video_frame")
    OUTPUT_ROOT = Path("output")  # e.g. ./output

    for bmp_path in INPUT_ROOT.rglob("*.bmp"):
        # Compute where to save this file’s result
        # e.g. video_assets/video_frame/A/B/C.bmp → output/A/B/C_face.bmp
        rel = bmp_path.relative_to(INPUT_ROOT)
        out_dir = OUTPUT_ROOT / rel.parent  # preserve subfolder
        out_file = out_dir / f"{rel.stem}_face.bmp"
        try:
            out_dir.mkdir(parents=True, exist_ok=True)

            img = cv2.imread(str(bmp_path))
            faces = app.get(img)

            for face in faces:
                # Unpack bbox & score
                x1, y1, x2, y2 = face.bbox.astype(int)
                score = face.det_score

                # Access landmarks, embedding, attributes
                # landmarks = face.landmark_2d_106  # 106 points
                # embedding = face.embedding  # 512-dim vector
                # gender = face.gender
                # age = face.age
                #
                # # Visualize
                # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # for x, y in landmarks:
                #     cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), -1)
                # print(f"BBox: {(x1,y1,x2,y2)}, Score: {score:.2f}")
                # print(f"Age: {age}, Gender: {gender}")
                # print(f"Embedding (first 5 dims): {embedding[:5]}")

            # 4. Show result
            # plt.imshow(img)
            # plt.show()
            face_roi = img[y1:y2, x1:x2]

            cv2.imwrite(str(out_file), face_roi)
            print(f"Saved cropped face to {out_file}")
            # cv2.imshow("Features", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        except Exception as e:
            print(f"Error: {str(e)}")
