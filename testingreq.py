import requests

API_KEY = "rc-e2a8efe2866caa4abb0fc073d72b3b51952e675990ed8137"
URL = "https://developer.amd.com.cn/radeon/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "model": "DeepSeek-V4-Flash",
    "messages": [
        {"role": "user", "content": "Respond with: AMD Token Factory connection verified."}
    ],
    "max_tokens": 50,
    "temperature": 0.2
}

print(f"Connecting to {URL}...")
response = requests.post(URL, headers=headers, json=payload, timeout=20)
print(f"Status Code: {response.status_code}")
if response.status_code == 200:
    print("\nResult:")
    print(response.json()["choices"][0]["message"]["content"])
else:
    print(f"Error: {response.text}")