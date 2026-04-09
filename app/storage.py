import base64
from pathlib import Path


def normalize_id(value: str) -> str:
    value = (value or "").strip().lower()
    return "".join(ch for ch in value if ch.isalnum())


def employee_key(company_uid: str, employee_id: str, national_id_number: str) -> str:
    return f"{normalize_id(company_uid)}__{normalize_id(employee_id)}__{normalize_id(national_id_number)}"


def save_base64_to_file(base64_string: str, path: Path) -> None:
    binary = base64.b64decode(base64_string)
    path.write_bytes(binary)