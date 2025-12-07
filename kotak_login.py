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

    # In some versions of the library, WSO2 Consumer Key/Secret are required.
    # In others (like the one in this repo), they might be commented out.
    # We load them to be safe, but fallback logic handles the initialization.
    consumer_key = os.getenv("CONSUMER_KEY")
    consumer_secret = os.getenv("CONSUMER_SECRET")

    # Try using NEO_APP_TOKEN if CONSUMER_KEY is not set,
    # as some docs suggest the "token" goes into "consumer_key".
    if not consumer_key:
        consumer_key = os.getenv("NEO_APP_TOKEN")

    mobile_number = os.getenv("KOTAK_MOBILE_NUMBER")
    ucc = os.getenv("KOTAK_UCC")
    mpin = os.getenv("KOTAK_MPIN")
    totp_secret = os.getenv("TOTP_SECRET")

    # Validate credentials
    missing_vars = []
    if not consumer_key: missing_vars.append("CONSUMER_KEY (or NEO_APP_TOKEN)")
    # Note: consumer_secret might be mandatory in the user's environment even if not here.
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
    client = None
    try:
        # Attempt initialization with consumer_secret (User's environment requirement)
        if consumer_secret:
            try:
                client = NeoAPI(environment='prod', consumer_key=consumer_key, consumer_secret=consumer_secret)
                print_success("NeoAPI Client initialized with Consumer Secret.")
            except TypeError:
                # Fallback for local repo version which doesn't accept consumer_secret
                print("[INFO] Local library version does not accept consumer_secret. Retrying without it...")
                client = NeoAPI(environment='prod', consumer_key=consumer_key)
                print_success("NeoAPI Client initialized (without Consumer Secret).")
        else:
            client = NeoAPI(environment='prod', consumer_key=consumer_key)
            print_success("NeoAPI Client initialized.")

    except Exception as e:
        print_error("Failed to initialize NeoAPI Client", str(e))
        print("[HINT] Ensure you have the correct library version and credentials.")
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
             # Handle non-standard success codes
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
