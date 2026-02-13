import os
import sys
import time
import json
import pyotp
from dotenv import load_dotenv
from neo_api_client import NeoAPI

# Load environment variables
load_dotenv()

def print_step(message):
    print(f"\n[STEP] {message}")

def print_success(message):
    print(f"[SUCCESS] {message}")

def print_error(message, details=None):
    print(f"\n[ERROR] {message}")
    if details:
        print(f"[DETAILS] {details}")

def get_totp(secret):
    try:
        totp = pyotp.TOTP(secret)
        return totp.now()
    except Exception as e:
        print_error("Failed to generate TOTP", str(e))
        sys.exit(1)

def main():
    print("----------------------------------------------------------------")
    print("               Kotak Neo API Auto-Login Script                  ")
    print("----------------------------------------------------------------")

    # 1. Retrieve Credentials
    print_step("Retrieving credentials from environment variables...")

    # Try using CONSUMER_KEY first (standard), then fallback to NEO_APP_TOKEN if user configured it that way
    consumer_key = os.getenv("CONSUMER_KEY")
    if not consumer_key:
        consumer_key = os.getenv("NEO_APP_TOKEN")

    mobile_number = os.getenv("KOTAK_MOBILE_NUMBER")
    ucc = os.getenv("KOTAK_UCC")
    mpin = os.getenv("KOTAK_MPIN")
    totp_secret = os.getenv("TOTP_SECRET")

    # Validate credentials
    missing_vars = []
    if not consumer_key: missing_vars.append("CONSUMER_KEY (or NEO_APP_TOKEN)")
    if not mobile_number: missing_vars.append("KOTAK_MOBILE_NUMBER")
    if not ucc: missing_vars.append("KOTAK_UCC")
    if not mpin: missing_vars.append("KOTAK_MPIN")
    if not totp_secret: missing_vars.append("TOTP_SECRET")

    if missing_vars:
        print_error("Missing required environment variables:", ", ".join(missing_vars))
        sys.exit(1)

    print_success("Credentials loaded successfully.")

    # 2. Generate TOTP
    print_step("Generating TOTP...")
    current_totp = get_totp(totp_secret)
    print_success(f"TOTP generated: {current_totp}")

    # 3. Initialize NeoAPI Client
    print_step("Initializing NeoAPI Client...")
    try:
        # Initializing without consumer_secret as confirmed by user success
        client = NeoAPI(environment='prod', consumer_key=consumer_key)
        print_success("NeoAPI Client initialized.")
    except Exception as e:
        print_error("Failed to initialize NeoAPI Client", str(e))
        sys.exit(1)

    # 4. TOTP Login (Step 1 of 2FA)
    print_step("Performing TOTP Login (Step 1/2)...")
    try:
        # mobile_number must be with country code e.g., +91...
        login_response = client.totp_login(mobile_number=mobile_number, ucc=ucc, totp=current_totp)

        # Check if response implies success.
        if isinstance(login_response, dict) and 'error' in login_response:
             print_error("TOTP Login failed", json.dumps(login_response, indent=2))
             sys.exit(1)

        print_success("TOTP Login successful.")
        # print("Response:", json.dumps(login_response, indent=2))

    except Exception as e:
        print_error("Exception during TOTP Login", str(e))
        sys.exit(1)

    # 5. TOTP Validate (Step 2 of 2FA - MPIN)
    print_step("Validating MPIN (Step 2/2)...")
    try:
        validate_response = client.totp_validate(mpin=mpin)

        if isinstance(validate_response, dict) and 'error' in validate_response:
             print_error("MPIN Validation failed", json.dumps(validate_response, indent=2))
             sys.exit(1)

        print_success("MPIN Validation successful. Session established.")
        # print("Response:", json.dumps(validate_response, indent=2))

    except Exception as e:
        print_error("Exception during MPIN Validation", str(e))
        sys.exit(1)

    # 6. Verify Session
    print_step("Verifying session by fetching positions...")
    try:
        positions = client.positions()

        if isinstance(positions, dict) and 'error' in positions:
             print_error("Failed to fetch positions (Session might be invalid)", json.dumps(positions, indent=2))
        elif isinstance(positions, dict) and 'stCode' in positions and positions['stCode'] != 200:
             # Handle non-standard success codes (like 5203 No Data)
             print_success("Session verified! API responded.")
             print(json.dumps(positions, indent=2))
        else:
             print_success("Session verified! Positions fetched successfully.")
             print(json.dumps(positions, indent=2))

    except Exception as e:
        print_error("Exception during session verification", str(e))

    print("\n----------------------------------------------------------------")
    print("               Login Process Completed                          ")
    print("----------------------------------------------------------------")

if __name__ == "__main__":
    main()
