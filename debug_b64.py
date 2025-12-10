import base64
import binascii

def try_decode(s, name):
    print(f"--- Testing {name} ---")
    print(f"Original Length: {len(s)}")
    print(f"Length % 4: {len(s) % 4}")
    
    # Strategy 1: Simple decode
    try:
        base64.b64decode(s, validate=False)
        print("Strategy 1 (Simple): Success")
        return
    except Exception as e:
        print(f"Strategy 1 (Simple): Failed - {e}")

    # Strategy 2: Fix padding logic (add =)
    s2 = s
    missing_padding = len(s2) % 4
    if missing_padding:
        s2 += '=' * (4 - missing_padding)
    try:
        base64.b64decode(s2, validate=False)
        print(f"Strategy 2 (Padding fix -> {len(s2)}): Success")
        return
    except Exception as e:
        print(f"Strategy 2 (Padding fix): Failed - {e}")

    # Strategy 3: Truncate last char if % 4 == 1
    if len(s) % 4 == 1:
        s3 = s[:-1]
        missing_padding = len(s3) % 4
        if missing_padding:
            s3 += '=' * (4 - missing_padding)
        try:
            base64.b64decode(s3, validate=False)
            print(f"Strategy 3 (Truncate 1 char): Success")
            return
        except Exception as e:
            print(f"Strategy 3 (Truncate 1 char): Failed - {e}")

# Case 1: Length 25 (e.g. 24 chars + 1 garbage)
case1 = "A" * 25
try_decode(case1, "Length 25 (Valid chars)")

# Case 2: Length 25 with special chars
case2 = "A" * 24 + "\n"
try_decode(case2, "Length 25 (Trailing newline)")
