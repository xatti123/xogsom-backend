from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.models import RegisterEmployeeRequest, VerifyEmployeeRequest, ApiResponse
from app.storage import employee_key, save_base64_to_file
from app.config import (
    FACE_DIR,
    FINGERPRINT_RAW_DIR,
    FINGERPRINT_PROCESSED_DIR,
)
from app.face_service import register_face, verify_face
from app.fingerprint_service import register_fingerprint, verify_fingerprint

app = FastAPI(title="Employee Biometric API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://beamish-lily-46f96e.netlify.app",
        "http://localhost:5000",
        "http://127.0.0.1:5000",
        "http://192.168.10.7:5000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"ok": True, "message": "Biometric API running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/register-employee", response_model=ApiResponse)
def register_employee(payload: RegisterEmployeeRequest):
    key = employee_key(
        payload.company_uid,
        payload.employee_id,
        payload.national_id_number,
    )

    try:
        if payload.face_image_base64:
            raw_face_path = FACE_DIR / f"{key}_registered.jpg"
            save_base64_to_file(payload.face_image_base64, raw_face_path)
            register_face(raw_face_path)

        if payload.fingerprint_image_base64:
            raw_fp_path = FINGERPRINT_RAW_DIR / f"{key}_registered_raw.jpg"
            processed_fp_path = (
                FINGERPRINT_PROCESSED_DIR / f"{key}_registered_processed.png"
            )

            save_base64_to_file(payload.fingerprint_image_base64, raw_fp_path)

            fp_registration = register_fingerprint(
                raw_input_path=raw_fp_path,
                processed_output_path=processed_fp_path,
            )

            if not fp_registration["quality"]["ok"]:
                raise HTTPException(
                    status_code=400,
                    detail="Fingerprint registration failed: "
                    + ", ".join(fp_registration["quality"]["problems"]),
                )

        return ApiResponse(
            success=True,
            result="registered",
            message="Employee biometric registration saved successfully",
            employee_key=key,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/verify-employee", response_model=ApiResponse)
def verify_employee(payload: VerifyEmployeeRequest):
    key = employee_key(
        payload.company_uid,
        payload.employee_id,
        payload.national_id_number,
    )

    try:
        if payload.biometric_type == "face":
            registered_face_path = FACE_DIR / f"{key}_registered.jpg"

            if not registered_face_path.exists():
                return ApiResponse(
                    success=True,
                    result="record_not_found",
                    message="Face record not found",
                    employee_key=key,
                )

            probe_face_path = FACE_DIR / f"{key}_probe.jpg"
            save_base64_to_file(payload.image_base64, probe_face_path)

            face_result = verify_face(
                registered_face_path=registered_face_path,
                probe_face_path=probe_face_path,
            )

            score = max(0.0, 1.0 - face_result["distance"])

            if face_result["verified"]:
                return ApiResponse(
                    success=True,
                    result="clear",
                    message="Employee verified successfully",
                    score=score,
                    employee_key=key,
                )

            return ApiResponse(
                success=True,
                result="flagged",
                message="Face does not match",
                score=score,
                employee_key=key,
            )

        if payload.biometric_type == "fingerprint":
            registered_processed_path = (
                FINGERPRINT_PROCESSED_DIR / f"{key}_registered_processed.png"
            )

            if not registered_processed_path.exists():
                return ApiResponse(
                    success=True,
                    result="record_not_found",
                    message="Fingerprint record not found",
                    employee_key=key,
                )

            raw_probe_path = FINGERPRINT_RAW_DIR / f"{key}_probe_raw.jpg"
            processed_probe_path = (
                FINGERPRINT_PROCESSED_DIR / f"{key}_probe_processed.png"
            )

            save_base64_to_file(payload.image_base64, raw_probe_path)

            fp_result = verify_fingerprint(
                raw_probe_path=raw_probe_path,
                processed_probe_path=processed_probe_path,
                registered_processed_path=registered_processed_path,
            )

            if fp_result["poor_quality"]:
                return ApiResponse(
                    success=True,
                    result="poor_quality",
                    message=fp_result["message"],
                    score=fp_result["score"],
                    employee_key=key,
                )

            if fp_result["verified"]:
                return ApiResponse(
                    success=True,
                    result="clear",
                    message="Employee verified successfully",
                    score=fp_result["score"],
                    employee_key=key,
                )

            return ApiResponse(
                success=True,
                result="flagged",
                message=fp_result["message"],
                score=fp_result["score"],
                employee_key=key,
            )

        raise HTTPException(status_code=400, detail="Invalid biometric type")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))