import requests
import sys
from datetime import datetime

def keep_alive():
    url = "https://montecarlo-trading-tool.streamlit.app/"
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Pinging {url}...")
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            print(f"Success! Status Code: {response.status_code}")
        else:
            print(f"Warning! Status Code: {response.status_code}")
    except Exception as e:
        print(f"Error pinging application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    keep_alive()
