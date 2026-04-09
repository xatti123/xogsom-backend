from pydantic import BaseModel
from typing import Optional


class RegisterEmployeeRequest(BaseModel):
    company_uid: str
    employee_name: str
    employee_id: str
    national_id_number: str
    employee_email: Optional[str] = None
    phone_number: Optional[str] = None
    department: Optional[str] = None
    designation: Optional[str] = None
    background_status: str
    crime_status: str
    background_notes: Optional[str] = None
    face_image_base64: Optional[str] = None
    fingerprint_image_base64: Optional[str] = None


class VerifyEmployeeRequest(BaseModel):
    company_uid: str
    employee_name: str
    employee_id: str
    national_id_number: str
    biometric_type: str
    image_base64: str


class ApiResponse(BaseModel):
    success: bool
    result: str
    message: str
    score: Optional[float] = None
    employee_key: Optional[str] = None