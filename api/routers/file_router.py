# this router will serve the files to the user

from fastapi import APIRouter
from fastapi.responses import FileResponse
from fastapi import Query

import shutil
# from core.security import verify_token
import os

router = APIRouter(tags=["file server"])

R_CODE_DIRECTORY = os.path.join(os.path.dirname(__file__), '../code')


def delete_file(filepath: str):
    if os.path.exists(filepath):
        os.remove(filepath)
        return {"message": "File deleted successfully"}
    else:
        return {"message": "File not found"}


@router.get('/figures/{user_id}/{file_path}')
async def get_file(user_id: str , file_path: str):
    try:     
        return FileResponse(f"{R_CODE_DIRECTORY}/{user_id}/figures/{file_path}")
    except Exception as e:
        return {"message": "Error in uploading file", "error": str(e)}

    # return {"test"}
    
    
@router.delete('/figures/{user_id}/{file_path}')
async def delete_figure(user_id: str, file_path: str):
    return delete_file(f"{R_CODE_DIRECTORY}/{user_id}/figures/{file_path}")

    
@router.get('/files/{user_id}/{file_path}')
async def get_file(user_id: str ,file_path: str):
    try:     
        print(f"{R_CODE_DIRECTORY}/{user_id}/files/{file_path}")
        return FileResponse(f"{R_CODE_DIRECTORY}/{user_id}/files/{file_path}")
    except Exception as e:
        return {"message": "Error in uploading file", "error": str(e)}
    
    
@router.delete('/files/{user_id}/{file_path}')
async def delete_file_route(user_id: str, file_path: str):
    return delete_file(f"{R_CODE_DIRECTORY}/{user_id}/files/{file_path}")
    



@router.get('/figures/micro/{user_id}/{file_path}')
async def get_file(user_id: str , file_path: str):
    try:     
        return FileResponse(f"{R_CODE_DIRECTORY}/{user_id}/micro/figures/{file_path}")
    except Exception as e:
        return {"message": "Error in uploading file", "error": str(e)}
    
    
@router.delete('/figures/micro/{user_id}/{file_path}')
async def delete_micro_figure(user_id: str, file_path: str):
    return delete_file(f"{R_CODE_DIRECTORY}/{user_id}/micro/figures/{file_path}")


@router.get('/files/micro/{user_id}/{file_path}')
async def get_file(user_id: str ,file_path: str):
    try:     
        return FileResponse(f"{R_CODE_DIRECTORY}/{user_id}/micro/files/{file_path}")
    except Exception as e:
        return {"message": "Error in uploading file", "error": str(e)}
    
    
@router.delete('/files/micro/{user_id}/{file_path}')
async def delete_micro_file(user_id: str, file_path: str):
    return delete_file(f"{R_CODE_DIRECTORY}/{user_id}/micro/files/{file_path}")




@router.get('/figures/annotation/{user_id}/{file_path}')
async def get_file(user_id: str , file_path: str):
    try:     
        return FileResponse(f"{R_CODE_DIRECTORY}/{user_id}/annotation/figures/{file_path}")
    except Exception as e:
        return {"message": "Error in uploading file", "error": str(e)}
    
    
@router.delete('/figures/annotation/{user_id}/{file_path}')
async def delete_annotation_figure(user_id: str, file_path: str):
    return delete_file(f"{R_CODE_DIRECTORY}/{user_id}/annotation/figures/{file_path}")

@router.get('/files/annotation/{user_id}/{file_path}')
async def get_file(user_id: str ,file_path: str):
    try:     
        return FileResponse(f"{R_CODE_DIRECTORY}/{user_id}/annotation/files/{file_path}")
    except Exception as e:
        return {"message": "Error in uploading file", "error": str(e)}
    
@router.delete('/files/annotation/{user_id}/{file_path}')
async def delete_annotation_file(user_id: str, file_path: str):
    return delete_file(f"{R_CODE_DIRECTORY}/{user_id}/annotation/files/{file_path}")



@router.get('/figures/venn/{user_id}/{file_path}')
async def get_file(user_id: str , file_path: str):
    try:     
        return FileResponse(f"{R_CODE_DIRECTORY}/{user_id}/venn/figures/{file_path}")
    except Exception as e:
        return {"message": "Error in uploading file", "error": str(e)}
    
@router.delete('/figures/venn/{user_id}/{file_path}')
async def delete_venn_figure(user_id: str, file_path: str):
    return delete_file(f"{R_CODE_DIRECTORY}/{user_id}/venn/figures/{file_path}")


@router.get('/files/venn/{user_id}/{file_path}')
async def get_file(user_id: str ,file_path: str):
    try:     
        return FileResponse(f"{R_CODE_DIRECTORY}/{user_id}/venn/files/{file_path}")
    except Exception as e:
        return {"message": "Error in uploading file", "error": str(e)}
    
@router.delete('/files/venn/{user_id}/{file_path}')
async def delete_venn_file(user_id: str, file_path: str):
    return delete_file(f"{R_CODE_DIRECTORY}/{user_id}/venn/files/{file_path}")

    

@router.get('/figures/heatmap/{user_id}/{file_path}')
async def get_file(user_id: str , file_path: str):
    try:     
        return FileResponse(f"{R_CODE_DIRECTORY}/{user_id}/heatmap/figures/{file_path}")
    except Exception as e:
        return {"message": "Error in uploading file", "error": str(e)}


@router.delete('/figures/heatmap/{user_id}/{file_path}')
async def delete_heatmap_figure(user_id: str, file_path: str):
    return delete_file(f"{R_CODE_DIRECTORY}/{user_id}/heatmap/figures/{file_path}")


@router.get('/files/heatmap/{user_id}/{file_path}')
async def get_file(user_id: str ,file_path: str):
    try:     
        return FileResponse(f"{R_CODE_DIRECTORY}/{user_id}/heatmap/files/{file_path}")
    except Exception as e:
        return {"message": "Error in uploading file", "error": str(e)}
    
@router.delete('/figures/heatmap/{user_id}/{file_path}')
async def delete_heatmap_figure(user_id: str, file_path: str):
    return delete_file(f"{R_CODE_DIRECTORY}/{user_id}/heatmap/figures/{file_path}")



@router.delete("/delete-all/{user_id}")
async def delete_all_files(
    user_id: str,
    folder: str = Query(default="", description="Relative subfolder path like 'files', 'micro/files', 'venn/figures'")
):
    target_dir = os.path.join(R_CODE_DIRECTORY, user_id, folder)

    if not os.path.exists(target_dir):
        return {"message": "Directory does not exist", "path": target_dir}
    
    if not os.path.isdir(target_dir):
        return {"message": "Path is not a directory", "path": target_dir}

    deleted_files = []
    failed_files = []

    for root, _, files in os.walk(target_dir):
        for f in files:
            file_path = os.path.join(root, f)
            try:
                os.remove(file_path)
                deleted_files.append(file_path)
            except Exception as e:
                failed_files.append({"file": file_path, "error": str(e)})

    return {
        "message": "Deletion completed",
        "deleted_file_count": len(deleted_files),
        "deleted_files": deleted_files,
        "errors": failed_files
    }