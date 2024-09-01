# this router will serve the files to the user

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from core.security import verify_token

router = APIRouter(tags=["file server"])


@router.get('/figures/{user_id}/{file_path}')
async def get_file(user_id: str , file_path: str):
    try:     
        return FileResponse(f"code/{user_id}/figures/{file_path}")
    except Exception as e:
        return {"message": "Error in uploading file", "error": str(e)}

    # return {"test"}
    
@router.get('/files/{user_id}/{file_path}')
async def get_file(user_id: str ,file_path: str):
    try:     
        return FileResponse(f"code/{user_id}/files/{file_path}")
    except Exception as e:
        return {"message": "Error in uploading file", "error": str(e)}
    


    
