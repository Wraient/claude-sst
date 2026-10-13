"""Account Pool Manager for Claude SST & Conversational Audio.

Manages a pool of Claude accounts, rotates between them to balance load,
handles rate limits (with automatic failover to the next healthy account),
and tracks account health to avoid using dead/expired sessions.
"""

import os
import json
import time
import uuid
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import aiohttp

logger = logging.getLogger("account_pool")

ACCOUNTS_FILE = Path(__file__).parent / "accounts.json"


class AccountPool:
    def __init__(self, filepath: Path = ACCOUNTS_FILE):
        self.filepath = filepath
        self._accounts: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._load_or_bootstrap()

    def _load_or_bootstrap(self):
        """Load accounts from accounts.json or bootstrap from browser profiles."""
        if self.filepath.exists():
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._accounts = data.get("accounts", [])
                    logger.info(f"Loaded {len(self._accounts)} accounts from {self.filepath}")
                    return
            except Exception as e:
                logger.error(f"Failed to read {self.filepath}: {e}")

        # If empty or file didn't exist, bootstrap from discovered browser accounts
        self._bootstrap_browser_accounts()

    def _bootstrap_browser_accounts(self):
        """Discover existing Claude accounts from Chrome and Brave."""
        from auth import extract_chrome_claude_cookies, get_chrome_key, decrypt_v11
        import sqlite3
        import shutil
        import hashlib

        discovered = []

        # 1. Chrome
        try:
            cookies = extract_chrome_claude_cookies()
            sk = cookies.get("sessionKey")
            org = cookies.get("lastActiveOrg", "")
            if sk:
                discovered.append({
                    "id": "chrome_default",
                    "name": "Chrome Account",
                    "sessionKey": sk,
                    "orgUuid": org,
                    "status": "active",
                    "rate_limited_until": 0,
                    "last_used": 0,
                    "usage_count": 0,
                    "error_count": 0,
                    "last_error": None,
                })
        except Exception as e:
            logger.debug(f"Chrome discovery error: {e}")

        # 2. Brave
        try:
            salt = b"saltysalt"
            key = hashlib.pbkdf2_hmac("sha1", "QPTzDTCQmsqy4E1HNR/jxg==".encode("utf-8"), salt, 1, 16)
            brave_p = Path.home() / ".config" / "BraveSoftware" / "Brave-Origin" / "Default" / "Cookies"
            if brave_p.exists():
                tmp = Path("/tmp/brave_bootstrap.sqlite")
                shutil.copy2(brave_p, tmp)
                conn = sqlite3.connect(tmp)
                cur = conn.cursor()
                cur.execute('SELECT encrypted_value FROM cookies WHERE host_key LIKE "%claude.ai%" AND name = "sessionKey"')
                row = cur.fetchone()
                if row:
                    sk = decrypt_v11(row[0], key)
                    if sk and sk not in [d["sessionKey"] for d in discovered]:
                        discovered.append({
                            "id": "brave_origin",
                            "name": "Brave Account",
                            "sessionKey": sk,
                            "orgUuid": "",
                            "status": "active",
                            "rate_limited_until": 0,
                            "last_used": 0,
                            "usage_count": 0,
                            "error_count": 0,
                            "last_error": None,
                        })
                conn.close()
                tmp.unlink(missing_ok=True)
        except Exception as e:
            logger.debug(f"Brave discovery error: {e}")

        self._accounts = discovered
        self._save_sync()

    def _save_sync(self):
        """Save account state to disk."""
        try:
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump({"accounts": self._accounts}, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save accounts to {self.filepath}: {e}")

    @staticmethod
    async def validate_session_key(session_key: str) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
        """Validate a Claude sessionKey against the Claude API.

        Returns (is_valid, org_uuid, org_name, error_msg).
        """
        headers = {
            "Cookie": f"sessionKey={session_key}",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            "Origin": "https://claude.ai",
            "Referer": "https://claude.ai/new",
        }
        try:
            timeout = aiohttp.ClientTimeout(total=8)
            async with aiohttp.ClientSession(timeout=timeout) as s:
                async with s.get("https://claude.ai/api/organizations", headers=headers) as resp:
                    if resp.status == 200:
                        orgs = await resp.json()
                        if orgs and isinstance(orgs, list) and len(orgs) > 0:
                            first_org = orgs[0]
                            org_uuid = first_org.get("uuid")
                            org_name = first_org.get("name") or "Claude Organization"
                            return True, org_uuid, org_name, None
                        return True, None, "Valid Session (No Orgs)", None
                    elif resp.status == 429:
                        return False, None, None, "Rate limited (HTTP 429)"
                    elif resp.status in (401, 403):
                        return False, None, None, f"Unauthorized / Session Expired (HTTP {resp.status})"
                    else:
                        return False, None, None, f"HTTP Error {resp.status}"
        except Exception as e:
            return False, None, None, str(e)

    async def initialize(self):
        """Initial check and metadata population for accounts."""
        async with self._lock:
            modified = False
            for acc in self._accounts:
                if not acc.get("orgUuid") or acc.get("name") in ("Chrome Account", "Brave Account"):
                    valid, org_uuid, org_name, err = await self.validate_session_key(acc["sessionKey"])
                    if valid and org_uuid:
                        acc["orgUuid"] = org_uuid
                        if org_name:
                            acc["name"] = org_name
                        acc["status"] = "active"
                        acc["last_error"] = None
                        modified = True
                    elif not valid and ("401" in str(err) or "403" in str(err)):
                        acc["status"] = "dead"
                        acc["last_error"] = err
                        modified = True
            if modified:
                self._save_sync()

    def get_next_account(self, exclude_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Select the best healthy account from the pool.

        Uses least-recently-used rotation to distribute load evenly across all accounts.
        Automatically lifts rate limit status once cooldown expires.
        """
        exclude_ids = exclude_ids or []
        now = time.time()

        # 1. Update expired rate limits
        for acc in self._accounts:
            if acc["status"] == "rate_limited":
                if now >= acc.get("rate_limited_until", 0):
                    acc["status"] = "active"
                    acc["last_error"] = None
                    logger.info(f"[ACCOUNT_POOL] Rate limit cooldown expired for {acc['name']} ({acc['id']}). Status set to active.")

        # 2. Filter active candidates
        candidates = [
            acc for acc in self._accounts
            if acc["status"] == "active" and acc["id"] not in exclude_ids
        ]

        if not candidates:
            # Check if any non-excluded are rate-limited with soonest expiry
            rate_limited = [
                acc for acc in self._accounts
                if acc["status"] == "rate_limited" and acc["id"] not in exclude_ids
            ]
            if rate_limited:
                rate_limited.sort(key=lambda a: a.get("rate_limited_until", 0))
                best = rate_limited[0]
                wait_sec = max(0, int(best.get("rate_limited_until", 0) - now))
                raise RuntimeError(
                    f"All Claude accounts in pool are currently rate-limited. Next account ({best['name']}) resets in {wait_sec}s."
                )

            dead_count = sum(1 for a in self._accounts if a["status"] == "dead")
            if dead_count == len(self._accounts) and self._accounts:
                raise RuntimeError(
                    f"All {dead_count} Claude accounts in the pool are dead/expired. Please update session keys."
                )

            raise RuntimeError("No available Claude accounts in pool.")

        # 3. Sort by least-recently-used (last_used ascending)
        candidates.sort(key=lambda a: a.get("last_used", 0))
        selected = candidates[0]
        return selected

    def mark_success(self, account_id: str):
        """Mark an account's request as successful."""
        for acc in self._accounts:
            if acc["id"] == account_id:
                acc["last_used"] = time.time()
                acc["usage_count"] = acc.get("usage_count", 0) + 1
                acc["error_count"] = 0
                acc["status"] = "active"
                acc["last_error"] = None
                self._save_sync()
                break

    def mark_rate_limited(self, account_id: str, cooldown_seconds: int = 900, reason: str = "Rate limit reached"):
        """Mark an account as rate-limited with a backoff cooldown (default: 15 mins)."""
        now = time.time()
        for acc in self._accounts:
            if acc["id"] == account_id:
                acc["status"] = "rate_limited"
                acc["rate_limited_until"] = now + cooldown_seconds
                acc["last_error"] = reason
                acc["error_count"] = acc.get("error_count", 0) + 1
                logger.warning(
                    f"[ACCOUNT_POOL] Account '{acc['name']}' ({account_id}) rate-limited for {cooldown_seconds}s. Reason: {reason}"
                )
                self._save_sync()
                break

    def mark_dead(self, account_id: str, reason: str = "Authentication failed"):
        """Mark an account as dead (expired session key / 401 / 403)."""
        for acc in self._accounts:
            if acc["id"] == account_id:
                acc["status"] = "dead"
                acc["last_error"] = reason
                acc["error_count"] = acc.get("error_count", 0) + 1
                logger.error(
                    f"[ACCOUNT_POOL] Account '{acc['name']}' ({account_id}) marked DEAD. Reason: {reason}"
                )
                self._save_sync()
                break

    async def add_account(self, session_key: str, name: Optional[str] = None, org_uuid: Optional[str] = None) -> Dict[str, Any]:
        """Validate and add a new account to the pool."""
        session_key = session_key.strip()
        # Check for duplicates
        for acc in self._accounts:
            if acc["sessionKey"] == session_key:
                raise ValueError("An account with this session key already exists in the pool.")

        is_valid, discovered_org, discovered_name, err = await self.validate_session_key(session_key)
        if not is_valid:
            raise ValueError(f"Session key validation failed: {err}")

        acc_id = f"acc_{uuid.uuid4().hex[:8]}"
        account = {
            "id": acc_id,
            "name": name or discovered_name or f"Claude Account {len(self._accounts) + 1}",
            "sessionKey": session_key,
            "orgUuid": org_uuid or discovered_org or "",
            "status": "active",
            "rate_limited_until": 0,
            "last_used": 0,
            "usage_count": 0,
            "error_count": 0,
            "last_error": None,
        }
        self._accounts.append(account)
        self._save_sync()
        logger.info(f"[ACCOUNT_POOL] Added new account: {account['name']} ({acc_id})")
        return account

    def remove_account(self, account_id: str) -> bool:
        """Remove an account from the pool."""
        initial_len = len(self._accounts)
        self._accounts = [a for a in self._accounts if a["id"] != account_id]
        if len(self._accounts) != initial_len:
            self._save_sync()
            logger.info(f"[ACCOUNT_POOL] Removed account: {account_id}")
            return True
        return False

    async def refresh_account(self, account_id: str) -> Dict[str, Any]:
        """Test and update the health of a specific account."""
        target = None
        for acc in self._accounts:
            if acc["id"] == account_id:
                target = acc
                break
        if not target:
            raise ValueError(f"Account {account_id} not found")

        is_valid, org_uuid, org_name, err = await self.validate_session_key(target["sessionKey"])
        if is_valid:
            target["status"] = "active"
            target["last_error"] = None
            if org_uuid:
                target["orgUuid"] = org_uuid
            if org_name and target["name"].startswith("Claude Account"):
                target["name"] = org_name
        else:
            if "429" in str(err):
                target["status"] = "rate_limited"
                target["rate_limited_until"] = time.time() + 900
            elif "401" in str(err) or "403" in str(err):
                target["status"] = "dead"
            target["last_error"] = err

        self._save_sync()
        return target

    def list_accounts(self, mask_keys: bool = True) -> List[Dict[str, Any]]:
        """List all accounts in the pool with masked session keys and countdowns."""
        now = time.time()
        results = []
        for acc in self._accounts:
            item = dict(acc)
            if mask_keys and item.get("sessionKey"):
                sk = item["sessionKey"]
                item["sessionKey"] = sk[:15] + "..." + sk[-6:] if len(sk) > 25 else "..."
            
            # Add dynamic fields
            rl_until = item.get("rate_limited_until", 0)
            if item["status"] == "rate_limited" and rl_until > now:
                item["cooldown_seconds_remaining"] = int(rl_until - now)
            else:
                item["cooldown_seconds_remaining"] = 0

            results.append(item)
        return results


account_pool = AccountPool()
