"""JWT authentication handler for the customer support platform."""

import jwt
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional
from jose import JWTError


class JWTHandler:
    """JWT token handler for authentication."""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        """Initialize JWT handler."""
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def create_token(
        self,
        user_id: str,
        email: str,
        roles: list = None,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a JWT token."""
        if expires_delta is None:
            expires_delta = timedelta(hours=24)
        
        expire = datetime.now(timezone.utc) + expires_delta
        
        payload = {
            "sub": user_id,
            "email": email,
            "roles": roles or [],
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "access"
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def decode_token(self, token: str) -> Dict[str, Any]:
        """Decode and validate JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError as e:
            raise ValueError(f"Invalid token: {str(e)}")
    
    def refresh_token(self, token: str) -> str:
        """Refresh an existing token."""
        try:
            payload = self.decode_token(token)
            # Remove exp and iat to create new ones
            payload.pop("exp", None)
            payload.pop("iat", None)
            
            # Create new token
            return self.create_token(
                user_id=payload["sub"],
                email=payload["email"],
                roles=payload.get("roles", [])
            )
        except Exception as e:
            raise ValueError(f"Cannot refresh token: {str(e)}")
    
    def verify_token(self, token: str) -> bool:
        """Verify if token is valid."""
        try:
            self.decode_token(token)
            return True
        except ValueError:
            return False