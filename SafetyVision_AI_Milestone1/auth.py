import re
from datetime import datetime, timezone
from typing import Tuple, Optional, Dict, Any
import bcrypt
import streamlit as st
from database import (
    get_user_by_email, get_user_by_id, create_user,
    record_failed_login, reset_failed_login, update_user_last_login,
    change_user_password, log_audit_action
)

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")

def validate_email(email: str) -> Tuple[bool, str]:
    """Validates email format."""
    if not email or not isinstance(email, str):
        return False, "Email address is required."
    clean_email = email.strip()
    if len(clean_email) > 254:
        return False, "Email address is too long."
    if not EMAIL_REGEX.match(clean_email):
        return False, "Please enter a valid email address (e.g., user@company.com)."
    return True, ""

def validate_password_strength(password: str) -> Tuple[bool, str]:
    """
    Enforces password complexity:
    - At least 8 characters
    - Contains at least one letter and at least one digit or symbol
    """
    if not password or not isinstance(password, str):
        return False, "Password is required."
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if len(password) > 128:
        return False, "Password must be under 128 characters."
    has_letter = any(c.isalpha() for c in password)
    has_digit_or_special = any(c.isdigit() or not c.isalnum() for c in password)
    if not (has_letter and has_digit_or_special):
        return False, "Password must contain at least one letter and one number or special character."
    return True, ""

def hash_password(plain_password: str) -> str:
    """Hashes a password with bcrypt and unique salt."""
    salt = bcrypt.gensalt(rounds=11)
    return bcrypt.hashpw(plain_password.encode("utf-8"), salt).decode("utf-8")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plaintext password against a stored bcrypt hash."""
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False

def authenticate_user(email: str, plain_password: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Authenticates a user safely:
    - Prevents account enumeration with constant-time dummy verification if user not found.
    - Enforces account lockout for repeated failed attempts.
    - Validates account active status.
    - Updates last login timestamp and resets failed attempts upon success.
    """
    clean_email = email.strip().lower()
    user = get_user_by_email(clean_email)

    # Constant-time dummy verification hash to prevent timing attacks if email is not found
    dummy_hash = "$2b$12$e80yqV4G8bFf.7nN8CgW4e1E8kZl1F7uI5b3f2.3f4c5e6g7h8i9j"

    if not user:
        # Perform dummy comparison
        verify_password(plain_password, dummy_hash)
        return False, "Invalid email or password.", None

    # Check lockout
    lockout_until = user.get("lockout_until")
    if lockout_until:
        try:
            lock_dt = datetime.fromisoformat(lockout_until.replace("Z", "+00:00"))
            if datetime.now(timezone.utc) < lock_dt:
                minutes_left = max(1, int((lock_dt - datetime.now(timezone.utc)).total_seconds() / 60))
                return False, f"Account is temporarily locked due to multiple failed attempts. Please retry in {minutes_left} minutes.", None
        except Exception:
            pass

    # Check active status
    if user.get("status") != "active":
        return False, "Your account has been deactivated. Please contact an administrator.", None

    # Check password
    stored_hash = user.get("password_hash", "")
    if not verify_password(plain_password, stored_hash):
        record_failed_login(clean_email)
        return False, "Invalid email or password.", None

    # Successful login
    update_user_last_login(user["id"])
    reset_failed_login(clean_email)
    log_audit_action(user["id"], "user_login", target_type="user", target_id=user["id"], details="Successful login")

    # Return safe user profile (without password hash)
    safe_profile = {
        "id": user["id"],
        "email": user["email"],
        "full_name": user["full_name"],
        "workplace": user.get("workplace", ""),
        "role": user.get("role", "standard"),
        "status": user.get("status", "active"),
        "created_at": user.get("created_at"),
        "last_login_at": user.get("last_login_at"),
    }
    return True, "Login successful.", safe_profile

def register_user(full_name: str, email: str, password: str, confirm_password: str, workplace: str = "") -> Tuple[bool, str, Optional[int]]:
    """Registers a new standard user account with validation."""
    clean_name = full_name.strip()
    clean_email = email.strip().lower()
    clean_workplace = workplace.strip()

    if not clean_name:
        return False, "Full name is required.", None

    is_valid_email, err_email = validate_email(clean_email)
    if not is_valid_email:
        return False, err_email, None

    if password != confirm_password:
        return False, "Passwords do not match.", None

    is_valid_pw, err_pw = validate_password_strength(password)
    if not is_valid_pw:
        return False, err_pw, None

    # Check duplicate email
    if get_user_by_email(clean_email):
        return False, "An account with this email address already exists.", None

    # Hash and save with standard role
    hashed_pw = hash_password(password)
    try:
        user_id = create_user(
            email=clean_email,
            password_hash=hashed_pw,
            full_name=clean_name,
            workplace=clean_workplace,
            role="standard"
        )
        return True, "Account created successfully! You can now log in.", user_id
    except Exception as exc:
        return False, f"Could not create account: {exc}", None

# ==========================================
# STREAMLIT SESSION STATE MANAGEMENT
# ==========================================

def init_session():
    """Initializes authentication state in Streamlit session."""
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "user" not in st.session_state:
        st.session_state["user"] = None
    if "auth_view" not in st.session_state:
        st.session_state["auth_view"] = "login"

def set_login_session(user_profile: Dict[str, Any]):
    """Stores user profile in session state."""
    st.session_state["authenticated"] = True
    st.session_state["user"] = user_profile

def logout_session():
    """Cleans up session state on logout."""
    user = st.session_state.get("user")
    if user:
        log_audit_action(user["id"], "user_logout", target_type="user", target_id=user["id"])
    st.session_state["authenticated"] = False
    st.session_state["user"] = None
    st.session_state["auth_view"] = "login"

def get_current_user() -> Optional[Dict[str, Any]]:
    """Returns currently authenticated user profile."""
    return st.session_state.get("user") if st.session_state.get("authenticated") else None

def is_admin() -> bool:
    """Checks if currently logged in user is an administrator."""
    user = get_current_user()
    return user is not None and user.get("role") == "admin"
