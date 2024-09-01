# this router will serve the files to the user

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from core.security import verify_token

router = APIRouter(tags=["file server"])


@router.get('/figures/{file_path}')
async def get_file(file_path: str,  user_info: dict = Depends(verify_token)):
    print(f"{file_path}")
    try:     
        return FileResponse(f"code/{str(user_info['user_id'])}/figures/{file_path}")
    except Exception as e:
        return {"message": "Error in uploading file", "error": str(e)}
    
@router.get('/files/{file_path}')
async def get_file(file_path: str,  user_info: dict = Depends(verify_token)):
    print(f"{file_path}")
    try:     
        return FileResponse(f"code/{str(user_info['user_id'])}/files/{file_path}")
    except Exception as e:
        return {"message": "Error in uploading file", "error": str(e)}
    


    
