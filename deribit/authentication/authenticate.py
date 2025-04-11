# /mnt/p/perpetual/deribit/authentication/authenticate.py
import json
import time
import logging
import os
import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load API credentials from environment
def load_credentials():
    """Load Deribit API credentials from environment file"""
    load_dotenv('/mnt/p/perpetual/config/credentials.env')
    api_key = os.getenv("DERIBIT_API_KEY")
    api_secret = os.getenv("DERIBIT_API_SECRET")
    
    if not api_key or not api_secret:
        logger.error("❌ Deribit API credentials missing. Please set DERIBIT_API_KEY and DERIBIT_API_SECRET in credentials.env")
        return None, None
        
    return api_key, api_secret

def authenticate_http(scope="session:trade:read"):
    """
    Authenticate with Deribit API using API key and secret via HTTP
    
    Args:
        scope: API access scope (default: session:trade:read)
        
    Returns:
        dict: Authentication result containing access_token, or None on failure
    """
    api_key, api_secret = load_credentials()
    if not api_key or not api_secret:
        logger.warning("📋 Authentication credentials missing, continuing with public access only")
        return None
    
    try:
        logger.info("🔑 Authenticating with Deribit API...")
        
        auth_params = {
            "grant_type": "client_credentials",
            "client_id": api_key.strip(),  # Added strip() to remove any whitespace
            "client_secret": api_secret.strip(),  # Added strip() to remove any whitespace
            "scope": scope
        }
        
        response = requests.post(
            "https://www.deribit.com/api/v2/public/auth",
            json=auth_params
        )
        
        if response.status_code != 200:
            logger.error(f"❌ Authentication failed: {response.text}")
            return None
            
        result = response.json().get("result")
        
        if not result:
            logger.error(f"❌ Authentication failed: {response.json().get('error')}")
            return None
        
        logger.info("✅ Authentication successful!")
        return result
    except Exception as e:
        logger.error(f"❌ Authentication error: {e}")
        return None

def get_auth_headers(access_token):
    """
    Get HTTP headers for authenticated requests
    
    Args:
        access_token: The access token from authentication
        
    Returns:
        dict: Headers dictionary with Authorization
    """
    return {
        "Authorization": f"Bearer {access_token}"
    }

def add_auth_to_params(params, access_token):
    """
    Add access token to a request params
    
    Args:
        params: The request params dictionary
        access_token: The access token from authentication
        
    Returns:
        dict: Updated params with access token
    """
    if params is None:
        params = {}
        
    params["access_token"] = access_token
    return params