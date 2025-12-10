import requests

try:
    r = requests.get('http://localhost:8000/download/gds')
    print(f"Status: {r.status_code}")
    print(f"Headers: {r.headers}")
    if 'Content-Disposition' in r.headers:
        print(f"Content-Disposition: {r.headers['Content-Disposition']}")
    else:
        print("Content-Disposition header MISSING!")
except Exception as e:
    print(e)
