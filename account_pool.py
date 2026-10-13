"""Account Pool Manager for Claude SST & Conversational Audio.

Manages a pool of Claude Farm accounts, rotates between them to balance load,
handles rate limits (with automatic failover to the next healthy account),
and tracks account health to avoid using dead/expired sessions.

Strictly excludes personal Gmail / Chrome / Brave accounts.
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
CLAUDE_FARM_ACCOUNTS_DIR = Path("/home/wraient/Projects/claude-farm/accounts")


def _is_personal_or_gmail(acc: Dict[str, Any]) -> bool:
    """Detect if an account is a personal browser/Gmail account that must be purged."""
    acc_id = acc.get("id", "").lower()
    name = acc.get("name", "").lower()
    email = acc.get("email", "").lower()

    if "gmail.com" in name or "gmail.com" in email:
        return True
    if acc_id in ("chrome_default", "brave_origin"):
        return True
    if "chrome" in name or "brave" in name:
        return True
    return False


class AccountPool:
    def __init__(self, filepath: Path = ACCOUNTS_FILE, farm_dir: Path = CLAUDE_FARM_ACCOUNTS_DIR):
        self.filepath = filepath
        self.farm_dir = farm_dir
        self._accounts: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._load_or_bootstrap()

    def _load_or_bootstrap(self):
        """Load accounts from accounts.json (purging Gmail) and sync from Claude Farm."""
        existing = []
        if self.filepath.exists():
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    raw_accounts = data.get("accounts", [])
                    # Strictly filter out personal / Gmail accounts
                    existing = [a for a in raw_accounts if not _is_personal_or_gmail(a)]
                    logger.info(f"Loaded {len(existing)} non-Gmail accounts from {self.filepath}")
            except Exception as e:
                logger.error(f"Failed to read {self.filepath}: {e}")

        self._accounts = existing
        # Synchronize with Claude Farm accounts directory
        self.sync_claude_farm_accounts(save=False)
        self._save_sync()

    def sync_claude_farm_accounts(self, save: bool = True) -> int:
        """Scan Claude Farm accounts directory for valid accounts and update pool."""
        if not self.farm_dir.exists():
            logger.warning(f"Claude Farm directory does not exist: {self.farm_dir}")
            return 0

        added_or_updated = 0
        known_by_email = {a.get("email", "").lower(): a for a in self._accounts if a.get("email")}
        known_by_key = {a.get("sessionKey", ""): a for a in self._accounts}

        for meta_file in self.farm_dir.glob("*/meta.json"):
            # Skip expired/archived subdirectories
            if "expired" in meta_file.parts:
                continue
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                email = (meta.get("email") or meta_file.parent.name).strip().lower()
                sk = (meta.get("session_key") or "").strip()
                org_id = (meta.get("org_id") or "").strip()

                if not sk or "gmail.com" in email:
                    continue

                if email in known_by_email:
                    acc = known_by_email[email]
                    if acc.get("sessionKey") != sk or acc.get("orgUuid") != org_id:
                        acc["sessionKey"] = sk
                        if org_id:
                            acc["orgUuid"] = org_id
                        acc["status"] = "active"
                        acc["last_error"] = None
                        added_or_updated += 1
                elif sk in known_by_key:
                    acc = known_by_key[sk]
                    acc["email"] = email
                    if org_id:
                        acc["orgUuid"] = org_id
                else:
                    safe_id = f"farm_{email.replace('@', '_').replace('.', '_')}"
                    new_acc = {
                        "id": safe_id,
                        "email": email,
                        "name": f"{email}'s Organization",
                        "sessionKey": sk,
                        "orgUuid": org_id,
                        "status": "active",
                        "rate_limited_until": 0,
                        "last_used": 0,
                        "usage_count": 0,
                        "error_count": 0,
                        "last_error": None,
                    }
                    self._accounts.append(new_acc)
                    known_by_email[email] = new_acc
                    known_by_key[sk] = new_acc
                    added_or_updated += 1
                    logger.info(f"[ACCOUNT_POOL] Discovered Claude Farm account: {email}")

            except Exception as e:
                logger.debug(f"Error reading {meta_file}: {e}")

        # Purge any remaining personal Gmail entries
        self._accounts = [a for a in self._accounts if not _is_personal_or_gmail(a)]

        if save:
            self._save_sync()

        return added_or_updated

    def _save_sync(self):
        """Save account state to disk."""
        try:
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            # Ensure no Gmail accounts ever persist
            clean_accounts = [a for a in self._accounts if not _is_personal_or_gmail(a)]
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump({"accounts": clean_accounts}, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save accounts to {self.filepath}: {e}")

    @staticmethod
    async def validate_session_key(session_key: str) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
        """Validate a Claude sessionKey against the Claude API.

        Returns (is_valid, org_uuid, org_name, error_msg).
        """
        headers = {
            "Cookie": f"sessionKey={session_key}",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
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
        """Initial sync from Claude Farm and metadata validation."""
        async with self._lock:
            # Sync fresh accounts from Claude Farm
            self.sync_claude_farm_accounts(save=False)
            modified = False
            for acc in self._accounts:
                if not acc.get("orgUuid") or not acc.get("name"):
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
                    f"All {dead_count} Claude accounts in the pool are dead/expired. Please generate fresh accounts via claude-farm."
                )

            raise RuntimeError("No available Claude accounts in pool. Please check Claude Farm accounts.")

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
            "name": name or discovered_name or f"Claude Farm Account {len(self._accounts) + 1}",
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
            if org_name:
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

            rl_until = item.get("rate_limited_until", 0)
            if item["status"] == "rate_limited" and rl_until > now:
                item["cooldown_seconds_remaining"] = int(rl_until - now)
            else:
                item["cooldown_seconds_remaining"] = 0

            results.append(item)
        return results


account_pool = AccountPool()
