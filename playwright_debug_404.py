"""
Playwright diagnostic: capture any 404 responses during document preview.

Usage (run from host machine):
  pip install playwright
  playwright install chromium
  python3 playwright_debug_404.py

The script opens the app, waits for you to navigate to a document preview
or trigger the document generation that causes the 404, then prints every
request that got a 4xx/5xx response including the full URL.

Set APP_URL to the base URL of your whisper-live instance.
"""

import sys
from playwright.sync_api import sync_playwright

APP_URL = "https://ubuntu.desmana-truck.ts.net"  # adjust if needed

failed_requests = []

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(ignore_https_errors=True)
        page = context.new_page()

        # Intercept all responses and record failures
        def on_response(response):
            if response.status >= 400:
                entry = {
                    "url": response.url,
                    "status": response.status,
                    "method": response.request.method,
                    "resource_type": response.request.resource_type,
                }
                failed_requests.append(entry)
                print(f"[{response.status}] {response.request.resource_type:10s}  {response.url}")

        page.on("response", on_response)

        print(f"Opening {APP_URL} ...")
        page.goto(APP_URL, wait_until="domcontentloaded")
        print("\nBrowser is open. Reproduce the 404 (open a document preview or click Download).")
        print("Press Enter here when done to print the summary and close.\n")
        input()

        browser.close()

    print("\n--- All failed requests ---")
    if not failed_requests:
        print("None captured.")
    for r in failed_requests:
        print(f"  [{r['status']}] {r['method']} {r['url']}")
        print(f"          resource_type={r['resource_type']}")

if __name__ == "__main__":
    run()
