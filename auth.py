"""Authentication module for Claude SST.

Provides session credentials and headers sourced from the Claude Farm account pool
or explicit environment variables. Personal browser accounts (Chrome/Brave) are strictly excluded.
"""

import os
import logging
from typing import Dict, Tuple, Optional, Any

logger = logging.getLogger("auth")


class ClaudeAuth:
    """Manages Claude session credentials and headers across account pool."""

    def __init__(self):
        self._session_key: Optional[str] = os.getenv("CLAUDE_SESSION_KEY")
        self._org_uuid: Optional[str] = os.getenv("CLAUDE_ORG_UUID")

    def get_credentials(self, account_id: Optional[str] = None) -> Tuple[Dict[str, str], str, str]:
        """Returns (cookie_dict, session_key, org_uuid).

        Selects an active Claude Farm account from the pool or environment.
        """
        # 1. Environment variables if explicitly provided
        if self._session_key and self._org_uuid:
            cookie_dict = {
                "sessionKey": self._session_key,
                "lastActiveOrg": self._org_uuid,
            }
            return cookie_dict, self._session_key, self._org_uuid

        # 2. Select from Account Pool (Claude Farm accounts)
        try:
            from account_pool import account_pool
            acc = account_pool.get_next_account() if not account_id else next((a for a in account_pool._accounts if a["id"] == account_id), None)
            if acc:
                cookie_dict = {
                    "sessionKey": acc["sessionKey"],
                    "lastActiveOrg": acc.get("orgUuid", ""),
                }
                return cookie_dict, acc["sessionKey"], acc.get("orgUuid", "")
        except Exception as e:
            logger.error(f"[auth] Failed to retrieve account from pool: {e}")
            raise

        raise RuntimeError(
            "No valid Claude Farm accounts found. Please ensure accounts exist in Claude Farm or check accounts.json."
        )

    def get_headers(self, account: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        if account:
            session_key = account.get("sessionKey", "")
            org_uuid = account.get("orgUuid", "")
            cookie_str = f"sessionKey={session_key}; lastActiveOrg={org_uuid}"
        else:
            cookies, session_key, org_uuid = self.get_credentials()
            cookie_str = "; ".join(f"{k}={v}" for k, v in cookies.items())

        return {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Cookie": cookie_str,
            "Origin": "https://claude.ai",
            "Referer": "https://claude.ai/new",
        }


auth_manager = ClaudeAuth()
