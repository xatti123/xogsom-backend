from pathlib import Path
from deepface import DeepFace

from config import FACE_MODEL_NAME, FACE_DETECTOR_BACKEND, FACE_DISTANCE_THRESHOLD


def register_face(face_path: Path) -> dict:
    embedding_objs = DeepFace.represent(
        img_path=str(face_path),
        model_name=FACE_MODEL_NAME,
        detector_backend=FACE_DETECTOR_BACKEND,
        enforce_detection=True,
    )

    if not embedding_objs:
        raise ValueError("No face detected in registration image")

    embedding = embedding_objs[0]["embedding"]
    return {
        "embedding": embedding,
    }


def verify_face(registered_face_path: Path, probe_face_path: Path) -> dict:
    result = DeepFace.verify(
        img1_path=str(registered_face_path),
        img2_path=str(probe_face_path),
        model_name=FACE_MODEL_NAME,
        detector_backend=FACE_DETECTOR_BACKEND,
        enforce_detection=True,
    )

    distance = float(result.get("distance", 1.0))
    verified = bool(result.get("verified", False))
    final_verified = verified or distance <= FACE_DISTANCE_THRESHOLD

    return {
        "verified": final_verified,
        "distance": distance,
    }