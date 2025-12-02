import subprocess
import requests
import time
import hashlib

API_URL = "http://127.0.0.1:3000/docs/private/api.json"
CHECK_INTERVAL = 5  # seconds

def get_hash_of_url(url):
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        return hashlib.sha256(r.content).hexdigest()
    except Exception as e:
        print(f"[Watcher] Error fetching {url}: {e}")
        return None

def run_orval():
    print("[Watcher] Running yarn run orval...")
    result = subprocess.run(["yarn", "run", "orval"])
    if result.returncode != 0:
        print("[Watcher] orval exited with an error")

def main():
    last_hash = get_hash_of_url(API_URL)
    if last_hash is None:
        print("[Watcher] Initial fetch failed, continuing anyway...")

    # First orval run before starting dev
    run_orval()

    # Start yarn dev (long-running)
    dev_process = subprocess.Popen(["yarn", "run", "dev"])

    try:
        while True:
            time.sleep(CHECK_INTERVAL)
            new_hash = get_hash_of_url(API_URL)
            if new_hash and new_hash != last_hash:
                print("[Watcher] Change detected in api.json")
                last_hash = new_hash
                run_orval()
    except KeyboardInterrupt:
        print("[Watcher] Stopping...")
        dev_process.terminate()

if __name__ == "__main__":
    main()
