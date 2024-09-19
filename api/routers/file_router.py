# this router will serve the files to the user

from fastapi import APIRouter
from fastapi.responses import FileResponse
# from core.security import verify_token
import os

router = APIRouter(tags=["file server"])

R_CODE_DIRECTORY = os.path.join(os.path.dirname(__file__), '../code')


@router.get('/figures/{user_id}/{file_path}')
async def get_file(user_id: str , file_path: str):
    try:     
        return FileResponse(f"{R_CODE_DIRECTORY}/{user_id}/figures/{file_path}")
    except Exception as e:
        return {"message": "Error in uploading file", "error": str(e)}

    # return {"test"}
    
@router.get('/files/{user_id}/{file_path}')
async def get_file(user_id: str ,file_path: str):
    try:     
        return FileResponse(f"{R_CODE_DIRECTORY}/{user_id}/files/{file_path}")
    except Exception as e:
        return {"message": "Error in uploading file", "error": str(e)}
    



@router.get('/figures/micro/{user_id}/{file_path}')
async def get_file(user_id: str , file_path: str):
    try:     
        return FileResponse(f"{R_CODE_DIRECTORY}/{user_id}/micro/figures/{file_path}")
    except Exception as e:
        return {"message": "Error in uploading file", "error": str(e)}
    


@router.get('/files/micro/{user_id}/{file_path}')
async def get_file(user_id: str ,file_path: str):
    try:     
        return FileResponse(f"{R_CODE_DIRECTORY}/{user_id}/micro/files/{file_path}")
    except Exception as e:
        return {"message": "Error in uploading file", "error": str(e)}