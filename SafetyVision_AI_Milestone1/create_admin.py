#!/usr/bin/env python3
"""
Detecto AI - Administrator Account Provisioning Tool
Safe, CLI-based initial administrator creation without hardcoded credentials.
"""
import sys
import argparse
import getpass
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from database import initialize_database, get_user_by_email, create_user, update_user_role, update_user_status, log_audit_action
from auth import validate_email, validate_password_strength, hash_password

def setup_admin():
    parser = argparse.ArgumentParser(description="Create or promote an administrator account for Detecto AI.")
    parser.add_argument("--email", help="Administrator email address", default=None)
    parser.add_argument("--name", help="Full name of the administrator", default=None)
    parser.add_argument("--password", help="Administrator password (optional, prompted securely if omitted)", default=None)
    parser.add_argument("--workplace", help="Organization / Workplace name", default="Safety Operations HQ")
    args = parser.parse_args()

    initialize_database()

    print("\n" + "="*50)
    print(" Detecto AI - Secure Administrator Provisioning")
    print("="*50 + "\n")

    # 1. Email input
    email = args.email
    if not email:
        email = input("Enter Administrator Email: ").strip()

    is_valid_email, err_email = validate_email(email)
    if not is_valid_email:
        print(f"\n[ERROR] {err_email}")
        sys.exit(1)

    existing_user = get_user_by_email(email)
    if existing_user:
        print(f"\n[NOTICE] User '{email}' already exists.")
        print(f"Current role: {existing_user['role']} | Status: {existing_user['status']}")
        if existing_user['role'] == 'admin' and existing_user['status'] == 'active':
            print("This user is already an active administrator.")
            sys.exit(0)

        confirm = input("Would you like to promote this user to Administrator and activate the account? (y/N): ").strip().lower()
        if confirm == 'y':
            update_user_role(existing_user['id'], 'admin', admin_user_id=existing_user['id'])
            update_user_status(existing_user['id'], 'active', admin_user_id=existing_user['id'])
            print(f"\n[SUCCESS] User '{email}' has been promoted to Administrator.")
            sys.exit(0)
        else:
            print("Action cancelled.")
            sys.exit(0)

    # 2. Name input
    name = args.name
    if not name:
        name = input("Enter Full Name: ").strip()
    if not name:
        print("\n[ERROR] Full name is required.")
        sys.exit(1)

    # 3. Password input
    password = args.password
    if not password:
        password = getpass.getpass("Enter Administrator Password (min 8 chars, 1 letter, 1 number/symbol): ")
        confirm_pw = getpass.getpass("Confirm Administrator Password: ")
        if password != confirm_pw:
            print("\n[ERROR] Passwords do not match.")
            sys.exit(1)

    is_valid_pw, err_pw = validate_password_strength(password)
    if not is_valid_pw:
        print(f"\n[ERROR] {err_pw}")
        sys.exit(1)

    # 4. Create administrator
    hashed_pw = hash_password(password)
    user_id = create_user(
        email=email,
        password_hash=hashed_pw,
        full_name=name,
        workplace=args.workplace or "Safety Operations HQ",
        role="admin"
    )

    log_audit_action(user_id, "admin_provisioned_cli", target_type="user", target_id=user_id, details=f"Email: {email}")
    print("\n" + "-"*50)
    print(f"[SUCCESS] Administrator account successfully created!")
    print(f"  User ID:    {user_id}")
    print(f"  Email:      {email}")
    print(f"  Name:       {name}")
    print(f"  Role:       admin")
    print(f"  Workplace:  {args.workplace or 'Safety Operations HQ'}")
    print("-"*50)
    print("You can now log in to the Detecto AI platform.\n")

if __name__ == "__main__":
    setup_admin()
