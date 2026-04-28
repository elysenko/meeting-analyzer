#!/usr/bin/env python3
# Automated smoke tests — run after every refactor phase.
# Tests the endpoints that don't require authentication.
# Usage: python3 test_smoke.sh [BASE_URL]
# Default BASE_URL: http://meeting-analyzer.colossus-apps.svc.cluster.local (internal cluster DNS)

import sys
import urllib.request
import urllib.error

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://meeting-analyzer.colossus-apps.svc.cluster.local"
PASS = 0
FAIL = 0


def check(id_, desc, url, expected_status, expected_body=None):
    global PASS, FAIL
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as r:
            status = r.status
            body = r.read().decode()
    except urllib.error.HTTPError as e:
        status = e.code
        body = e.read().decode()
    except Exception as e:
        print(f"FAIL [{id_}] {desc} — {e}")
        FAIL += 1
        return

    ok = (status == expected_status)
    if ok and expected_body and expected_body not in body:
        ok = False
    label = "PASS" if ok else "FAIL"
    detail = "" if ok else f" (expected {expected_status}, body={body[:120]})"
    print(f"{label} [{id_}] {desc} — HTTP {status}{detail}")
    if ok:
        PASS += 1
    else:
        FAIL += 1


def check_redirect(id_, desc, url):
    """Check that the URL issues a redirect (3xx) without following it."""
    global PASS, FAIL

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **kw):
            return None

    opener = urllib.request.build_opener(NoRedirect())
    try:
        with opener.open(url, timeout=10) as r:
            status = r.status
    except urllib.error.HTTPError as e:
        status = e.code
    except Exception as e:
        print(f"FAIL [{id_}] {desc} — {e}")
        FAIL += 1
        return

    ok = 300 <= status < 400
    label = "PASS" if ok else "FAIL"
    detail = "" if ok else f" (expected 3xx, got {status})"
    print(f"{label} [{id_}] {desc} — HTTP {status}{detail}")
    if ok:
        PASS += 1
    else:
        FAIL += 1


print(f"=== Smoke tests against {BASE} ===\n")

check("A1", "GET /health",       f"{BASE}/health",       200, "ok")
check("A2", "GET /health/live",  f"{BASE}/health/live",  200, "alive")
check("A3", "GET /health/ready", f"{BASE}/health/ready", 200, "ready")
check("A4", "GET /healthz",      f"{BASE}/healthz",      200)
check("A6", "GET /favicon.ico",  f"{BASE}/favicon.ico",  200)
check_redirect("A5", "GET / unauthenticated → redirect to login", f"{BASE}/")

print(f"\n=== Results: {PASS} passed, {FAIL} failed ===")
sys.exit(0 if FAIL == 0 else 1)
