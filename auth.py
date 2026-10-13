"""Authentication module for Claude SST.

Extracts session cookies directly from Google Chrome (Linux / Gnome Keyring)
or environment variables.
"""

import os
import shutil
import sqlite3
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, Tuple, Optional
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

CHROME_COOKIE_PATH = Path.home() / ".config" / "google-chrome" / "Default" / "Cookies"


def get_chrome_key() -> Optional[bytes]:
    """Retrieve Chrome's encryption key from Gnome Keyring / secret-tool."""
    try:
        proc = subprocess.run(
            ["secret-tool", "search", "application", "chrome"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        password = None
        for line in proc.stdout.splitlines():
            if line.startswith("secret = "):
                password = line.split(" = ", 1)[1].strip()
                break
        if not password:
            return None
        salt = b"saltysalt"
        return hashlib.pbkdf2_hmac("sha1", password.encode("utf-8"), salt, 1, 16)
    except Exception as e:
        print(f"[auth] Failed to retrieve Chrome key via secret-tool: {e}")
        return None


def decrypt_v11(data: bytes, key: bytes) -> Optional[str]:
    """Decrypt Chrome v11 encrypted cookie value."""
    if not data or not data.startswith(b"v11"):
        return None
    try:
        payload = data[3:]
        iv = b" " * 16
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        dec = decryptor.update(payload) + decryptor.finalize()
        pad = dec[-1]
        raw = dec[:-pad]
        # First 32 bytes are SHA-256 HMAC header
        return raw[32:].decode("utf-8", errors="ignore")
    except Exception:
        return None


def extract_chrome_claude_cookies() -> Dict[str, str]:
    """Extract all claude.ai cookies from Chrome's SQLite database."""
    if not CHROME_COOKIE_PATH.exists():
        return {}

    key = get_chrome_key()
    if not key:
        return {}

    temp_db = Path("/tmp/claude_sst_chrome_cookies.sqlite")
    try:
        shutil.copy2(CHROME_COOKIE_PATH, temp_db)
        conn = sqlite3.connect(temp_db)
        cur = conn.cursor()
        cur.execute(
            'SELECT name, encrypted_value FROM cookies WHERE host_key LIKE "%claude.ai%"'
        )
        cookies = {}
        for name, enc_val in cur.fetchall():
            val = decrypt_v11(enc_val, key)
            if val:
                cookies[name] = val
        conn.close()
        return cookies
    except Exception as e:
        print(f"[auth] Error reading Chrome cookies: {e}")
        return {}
    finally:
        if temp_db.exists():
            temp_db.unlink(missing_ok=True)


class ClaudeAuth:
    """Manages Claude session credentials and headers."""

    def __init__(self):
        self._cached_cookies: Dict[str, str] = {}
        self._session_key: Optional[str] = os.getenv("CLAUDE_SESSION_KEY")
        self._org_uuid: Optional[str] = os.getenv("CLAUDE_ORG_UUID")

    def get_credentials(self) -> Tuple[Dict[str, str], str, str]:
        """Returns (cookie_dict, session_key, org_uuid).

        Refreshes cookies from Chrome if needed.
        """
        # 1. Environment variables if provided
        if self._session_key and self._org_uuid:
            cookie_dict = {
                "sessionKey": self._session_key,
                "lastActiveOrg": self._org_uuid,
            }
            return cookie_dict, self._session_key, self._org_uuid

        # 2. Extract from Chrome
        cookies = extract_chrome_claude_cookies()
        if cookies and "sessionKey" in cookies:
            self._cached_cookies = cookies
            session_key = cookies["sessionKey"]
            org_uuid = cookies.get("lastActiveOrg", "")
            return cookies, session_key, org_uuid

        # 3. Fallback to cached
        if self._cached_cookies and "sessionKey" in self._cached_cookies:
            return (
                self._cached_cookies,
                self._cached_cookies["sessionKey"],
                self._cached_cookies.get("lastActiveOrg", ""),
            )

        raise RuntimeError(
            "No valid Claude cookies found. Please log in to claude.ai in Chrome or set CLAUDE_SESSION_KEY and CLAUDE_ORG_UUID in the environment."
        )

    def get_headers(self) -> Dict[str, str]:
        cookies, session_key, org_uuid = self.get_credentials()
        cookie_header = "; ".join(f"{k}={v}" for k, v in cookies.items())
        return {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            "Cookie": cookie_header,
            "Origin": "https://claude.ai",
            "Referer": "https://claude.ai/new",
        }


auth_manager = ClaudeAuth()
