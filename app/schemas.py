from pydantic import BaseModel

class ImageResponse(BaseModel):
    status: str
    image: bytes

class TextResponse(BaseModel):
    status: str
    text: str
