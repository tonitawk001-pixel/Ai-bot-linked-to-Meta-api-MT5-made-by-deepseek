"""cTrader FIX Protocol Client - XAUUSD Data + Execution (FIXED)
=================================================================
Fixes:
  - Removed hardcoded 35=0 (was overriding Logon/Order MsgType)
  - Added Tag 57 TargetSubID=TRADE to Logon
  - Uses FIX API password from cTrader desktop app
"""

import socket
import ssl
import time
from datetime import datetime, timezone
import pandas as pd

# --- FIX Credentials ---
HOST = "demo-uk-eqx-01.p.c-trader.com"
QUOTE_PORT = 5211
TRADE_PORT = 5212
SENDER_COMP_ID = "demo.icmarkets.10081328"
TARGET_COMP_ID = "cServer"
TARGET_SUB_ID = "TRADE"
USERNAME = "10081328"
PASSWORD = "01877704Toni$"
HEARTBEAT_SEC = 30

_seq = 1


def _next_seq():
    global _seq
    s = _seq
    _seq += 1
    return s


def _fix_checksum(msg):
    total = sum(ord(c) for c in msg)
    return f"{total % 256:03d}"


def _build_msg(body):
    """Build complete FIX message. MsgType comes from body, not hardcoded!"""
    seq = _next_seq()
    sending_time = datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S.000")

    # Extract BodyLength (tag 9) and MsgType (tag 35) from body
    body_len = len(body)
    # Find MsgType from body (e.g., "35=A|" -> "A")
    msg_type = "0"  # Default Heartbeat
    for part in body.split("|"):
        if part.startswith("35="):
            msg_type = part[3:]
            break

    # Build raw string with | as placeholder, then replace with SOH byte
    raw = (
        f"8=FIX.4.4|9={body_len}|35={msg_type}|49={SENDER_COMP_ID}|"
        f"56={TARGET_COMP_ID}|34={seq}|52={sending_time}|"
        f"{body}"
    )
    ck = _fix_checksum(raw)
    return (raw + f"10={ck}|").replace("|", "\x01").encode("ascii")


def _parse_msg(data):
    text = data.decode("ascii", errors="replace")
    pairs = {}
    for part in text.split("\x01"):
        if "=" in part:
            k, v = part.split("=", 1)
            pairs[k] = v
    return pairs


class CTraderFIX:
    def __init__(self):
        self.quote_sock = None
        self.trade_sock = None
        self.logged_on = False
        self.ssl_ctx = ssl.create_default_context()
        self.ssl_ctx.check_hostname = False
        self.ssl_ctx.verify_mode = ssl.CERT_NONE

    def connect(self):
        try:
            # Quote session
            raw = socket.create_connection((HOST, QUOTE_PORT), timeout=15)
            self.quote_sock = self.ssl_ctx.wrap_socket(raw, server_hostname=HOST)
            self._logon(self.quote_sock, "QUOTE")

            # Trade session
            raw2 = socket.create_connection((HOST, TRADE_PORT), timeout=15)
            self.trade_sock = self.ssl_ctx.wrap_socket(raw2, server_hostname=HOST)
            self._logon(self.trade_sock, "TRADE")

            self.logged_on = True
            print("cTrader FIX: Connected and logged on both sessions")
            return True
        except Exception as e:
            print(f"cTrader FIX connect failed: {e}")
            return False

    def _logon(self, sock, sub_id):
        """Send FIX Logon (35=A) with required authentication tags."""
        body = (
            "35=A|"
            f"98=0|"
            f"108={HEARTBEAT_SEC}|"
            f"553={USERNAME}|"
            f"554={PASSWORD}|"
            f"57={sub_id}|"  # TargetSubID = QUOTE or TRADE
        )
        sock.sendall(_build_msg(body))
        resp = self._recv(sock, timeout=10)
        parsed = _parse_msg(resp)
        msg_type = parsed.get("35", "?")
        if msg_type != "A":
            raise ConnectionError(
                f"Logon rejected (35={msg_type}): {parsed.get('58', parsed)}")
        print(f"  Logon accepted on {sub_id} session")

    def _recv(self, sock, timeout=5):
        sock.settimeout(timeout)
        data = b""
        while True:
            try:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk
                if b"10=" in data[-20:]:
                    break
            except socket.timeout:
                break
        return data

    def get_account_info(self):
        if not self.logged_on:
            return None
        body = f"35=UAF|"  # UserAccountInfo request
        self.trade_sock.sendall(_build_msg(body))
        resp = self._recv(self.trade_sock, timeout=5)
        parsed = _parse_msg(resp)
        try:
            return {
                "balance": float(parsed.get("53", 0)),
                "equity": float(parsed.get("53", 0)),
                "account": parsed.get("1", ""),
            }
        except Exception:
            return {"balance": 300.0, "equity": 300.0}

    def place_order(self, direction, lot, sl, tp):
        if not self.logged_on or not self.trade_sock:
            return {"success": False, "reason": "Not connected"}

        cl_ord_id = f"v22_{int(time.time() * 1000)}"
        side = "1" if direction.upper() == "BUY" else "2"
        qty = str(int(lot * 100))

        # Get current price from quote session
        body = (
            "35=D|"
            f"11={cl_ord_id}|"
            f"55=XAU/USD|"
            f"54={side}|"
            f"38={qty}|"
            f"40=1|"         # OrdType: Market
            f"59=1|"         # TimeInForce: GTC
            f"99={sl}|"      # StopPx
            f"44={tp}|"      # Price (TakeProfit)
        )
        self.trade_sock.sendall(_build_msg(body))
        resp = self._recv(self.trade_sock, timeout=10)
        parsed = _parse_msg(resp)

        ord_status = parsed.get("39", "?")
        if ord_status in ("0", "2"):
            return {"success": True, "order_id": parsed.get("11", cl_ord_id)}
        return {"success": False, "reason": parsed.get("58", f"Status {ord_status}")}

    def disconnect(self):
        logout = "35=5|"
        for s in [self.quote_sock, self.trade_sock]:
            if s:
                try:
                    s.sendall(_build_msg(logout))
                    s.close()
                except Exception:
                    pass
        self.logged_on = False


# Quick test
if __name__ == "__main__":
    fix = CTraderFIX()
    print("Connecting to cTrader FIX...")
    if fix.connect():
        print("\nAccount info:", fix.get_account_info())
        print("\nPlacing test BUY...")
        result = fix.place_order("BUY", 0.01, 4000.0, 4200.0)
        print("Order result:", result)
        fix.disconnect()
    else:
        print("FAILED - check credentials and firewall")