from pydantic import BaseModel

class valide_request(BaseModel):
    model_name: str
    mode: str