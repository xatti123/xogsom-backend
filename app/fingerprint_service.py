from pathlib import Path

import cv2
import numpy as np

RAW_MATCH_MIN_SCORE = 0.68
RAW_MATCH_MIN_GOOD_MATCHES = 10
RAW_MIN_BRIGHTNESS = 20
RAW_MAX_BRIGHTNESS = 245
RAW_MIN_SHARPNESS = 8.0


def _read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Invalid fingerprint image: {path}")
    return image


def preprocess_fingerprint_image(input_path: Path, output_path: Path) -> None:
    image = _read_bgr(input_path)
    cv2.imwrite(str(output_path.with_suffix(".png")), image)


def evaluate_fingerprint_quality(processed_path: Path) -> dict:
    image = _read_bgr(processed_path.with_suffix(".png"))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    problems = []

    if brightness < RAW_MIN_BRIGHTNESS:
        problems.append("Image is too dark")
    if brightness > RAW_MAX_BRIGHTNESS:
        problems.append("Image is too bright")
    if sharpness < RAW_MIN_SHARPNESS:
        problems.append("Image is too blurry")

    return {
        "ok": len(problems) == 0,
        "brightness": brightness,
        "sharpness": sharpness,
        "foreground_ratio": 1.0,
        "problems": problems,
    }


def _crop_center_fingerprint_area(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    x1 = int(w * 0.45)
    x2 = int(w * 0.92)
    y1 = int(h * 0.28)
    y2 = int(h * 0.80)

    cropped = img[y1:y2, x1:x2]
    if cropped.size == 0:
        return img
    return cropped


def _orb_features_bgr(img: np.ndarray):
    orb = cv2.ORB_create(
        nfeatures=3000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=10,
        patchSize=31,
        fastThreshold=10,
    )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors


def _orb_score(img1: np.ndarray, img2: np.ndarray) -> tuple[float, int]:
    kp1, des1 = _orb_features_bgr(img1)
    kp2, des2 = _orb_features_bgr(img2)

    if des1 is None or des2 is None or len(kp1) < 20 or len(kp2) < 20:
        return 0.0, 0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = matcher.knnMatch(des1, des2, k=2)

    good_matches = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    score = min(1.0, len(good_matches) / 40.0)
    return float(score), len(good_matches)


def _same_size(img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    return img1[:h, :w], img2[:h, :w]


def _template_score_color(img1: np.ndarray, img2: np.ndarray) -> float:
    img1, img2 = _same_size(img1, img2)
    result = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
    return float(result[0][0])


def _shift_variants(img: np.ndarray):
    variants = []
    shifts_x = [-8, -4, 0, 4, 8]
    shifts_y = [-4, 0, 4]

    for dx in shifts_x:
        for dy in shifts_y:
            matrix = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted = cv2.warpAffine(
                img,
                matrix,
                (img.shape[1], img.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            variants.append(shifted)

    return variants


def compare_fingerprint_images(
    registered_processed_path: Path,
    probe_processed_path: Path,
) -> dict:
    registered_path = registered_processed_path.with_suffix(".png")
    probe_path = probe_processed_path.with_suffix(".png")

    registered = _read_bgr(registered_path)
    probe = _read_bgr(probe_path)

    registered = _crop_center_fingerprint_area(registered)
    probe = _crop_center_fingerprint_area(probe)

    best_score = 0.0
    best_matches = 0

    for probe_variant in _shift_variants(probe):
        template_score = _template_score_color(registered, probe_variant)
        orb_score, good_matches = _orb_score(registered, probe_variant)

        combined = (
            (template_score * 0.30)
            + (orb_score * 0.70)
        )

        print(
            "template_score=", template_score,
            "orb_score=", orb_score,
            "good_matches=", good_matches,
            "combined=", combined,
        )

        if combined > best_score:
            best_score = combined
            best_matches = good_matches

    verified = (
        best_score >= RAW_MATCH_MIN_SCORE
        and best_matches >= RAW_MATCH_MIN_GOOD_MATCHES
    )

    return {
        "verified": verified,
        "score": float(best_score),
        "poor_quality": False,
        "message": (
            "Fingerprint verified successfully"
            if verified
            else "Fingerprint does not match"
        ),
        "good_matches": best_matches,
    }


def register_fingerprint(raw_input_path: Path, processed_output_path: Path) -> dict:
    preprocess_fingerprint_image(raw_input_path, processed_output_path)
    quality = evaluate_fingerprint_quality(processed_path=processed_output_path)

    return {
        "processed_path": str(processed_output_path.with_suffix(".png")),
        "quality": quality,
    }


def verify_fingerprint(
    raw_probe_path: Path,
    processed_probe_path: Path,
    registered_processed_path: Path,
) -> dict:
    preprocess_fingerprint_image(raw_probe_path, processed_probe_path)
    quality = evaluate_fingerprint_quality(processed_path=processed_probe_path)

    severe_problems = []
    if quality["brightness"] < 20:
        severe_problems.append("Image is extremely dark")
    if quality["sharpness"] < 8:
        severe_problems.append("Image is extremely blurry")

    if severe_problems:
        return {
            "verified": False,
            "score": 0.0,
            "poor_quality": True,
            "message": " ; ".join(severe_problems),
        }

    return compare_fingerprint_images(
        registered_processed_path=registered_processed_path,
        probe_processed_path=processed_probe_path,
    )