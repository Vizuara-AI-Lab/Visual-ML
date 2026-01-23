"""
Quick test script to verify cookie-based authentication setup.
Run this to check if cookies are being set correctly.
"""

import requests

# Test login
login_url = "http://localhost:8000/api/v1/auth/student/login"
login_data = {
    "emailId": "saam.official.use@gmail.com",
    "password": "stringS2st"  # Replace with actual password
}

print("🔐 Testing login...")
session = requests.Session()  # Use session to automatically handle cookies

try:
    # Login
    response = session.post(login_url, json=login_data)
    print(f"✅ Login Status: {response.status_code}")
    print(f"📦 Response: {response.json()}")
    
    # Check cookies
    print(f"\n🍪 Cookies set:")
    for cookie in session.cookies:
        print(f"  - {cookie.name}: {cookie.value[:20]}... (HttpOnly: {cookie.has_nonstandard_attr('HttpOnly')})")
    
    # Test protected endpoint
    print("\n🔒 Testing protected endpoint (/projects)...")
    projects_response = session.get("http://localhost:8000/api/v1/projects")
    print(f"✅ Projects Status: {projects_response.status_code}")
    
    if projects_response.status_code == 200:
        print("🎉 SUCCESS! Cookies are working correctly!")
    else:
        print(f"❌ FAILED: {projects_response.text}")
        
except Exception as e:
    print(f"❌ Error: {e}")
