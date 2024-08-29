

from fastapi import APIRouter, File, UploadFile
import rpy2.robjects as robjects
import os

router = APIRouter(prefix='/operations', tags=['operation'])

R_CODE_DIRECTORY = os.path.join(os.path.dirname(__file__), '../code')

FILE_DIR = os.path.join(R_CODE_DIRECTORY, 'files')


@router.post('/init')
async def init(count_data: UploadFile = File(...), meta_data: UploadFile = File(...)):

    try:
        os.makedirs(FILE_DIR, exist_ok=True)
        robjects.r['setwd'](R_CODE_DIRECTORY)
        file1_path = os.path.join( FILE_DIR, "count_data.csv")
        with open(file1_path, "wb") as f:
            f.write(await count_data.read())

        file2_path = os.path.join(FILE_DIR, "meta_data.csv")
        with open(file2_path, "wb") as f:
            f.write(await meta_data.read())
        
        robjects.r(
        """
            source("req_packages.R")
            count_data <- read.csv("files/count_data.csv", header = TRUE, row.names = 1)
            sample_info <- read.csv("files/meta_data.csv", header = TRUE, row.names = 1)
            saveRDS(count_data, "rds/count_data.rds")
            saveRDS(sample_info, "rds/sample_info.rds")
        """)



    except Exception as e:
        return {"message": "Error in uploading file", "error": str(e)}
    
    

    return {"message": "file uploadeded & Processed successfully!" ,"count_data": file1_path, "meta_data": file2_path}   




@router.get('/analyze')
async def analyze():
    try:
        robjects.r("source('analyze.R')")
    except Exception as e:
        return {"message": "Error in analyzing file", "error": str(e)}
    
    return {"message": "Analysis completed successfully!"}
       
    
    









# print(r_script_dir)

# robjects.r['setwd'](r_script_dir)

# robjects.r(""" 
# l <- c(1,2,3,4,5)
# saveRDS(l, "l.rds")
# """
# )


# def fun():

#     print(os.getcwd())

#     try:


#         if os.path.exists('test.R'):
#             robjects.r('source("test.R")')
#         else:
#             print("File test.R not found.")

#     except Exception as e:
#         print(e)


# def test():
#     fun()

# test()

# def best():
#     robjects.r("source('test2.R')")


# best()