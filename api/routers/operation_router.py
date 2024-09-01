

from fastapi import APIRouter, File, UploadFile, Depends
from core.security import verify_token
import rpy2.robjects as robjects
import os
import subprocess
from models.schema import OutlierSchema
from core.consts import BASE_URL


router = APIRouter(prefix='/operations', tags=['operation'])

R_CODE_DIRECTORY = os.path.join(os.path.dirname(__file__), '../code')




@router.post('/init')
async def init(count_data: UploadFile = File(...), meta_data: UploadFile = File(...), user_info: dict = Depends(verify_token)):

    try:

        os.makedirs(os.path.join(R_CODE_DIRECTORY, str(user_info['user_id'])), exist_ok=True)
        os.makedirs(os.path.join(R_CODE_DIRECTORY, str(user_info['user_id']), "rds"), exist_ok=True)
        os.makedirs(os.path.join(R_CODE_DIRECTORY, str(user_info['user_id']), "files"), exist_ok=True)
        os.makedirs(os.path.join(R_CODE_DIRECTORY, str(user_info['user_id']), "figures"), exist_ok=True)

        FILE_DIR = os.path.join(R_CODE_DIRECTORY, f"{user_info['user_id']}", "files")
        robjects.r['setwd'](R_CODE_DIRECTORY + f"/{user_info['user_id']}")
        file1_path = os.path.join( FILE_DIR, f"count_data.csv")
        with open(file1_path, "wb") as f:
            f.write(await count_data.read())

        file2_path = os.path.join(FILE_DIR, f"meta_data.csv")
        with open(file2_path, "wb") as f:
            f.write(await meta_data.read())
        
        robjects.r(
        f"""
            source("../req_packages.R")
            count_data <- read.csv("files/count_data.csv", header = TRUE, row.names = 1)
            sample_info <- read.csv("files/meta_data.csv", header = TRUE, row.names = 1)
            saveRDS(count_data, "rds/count_data.rds")
            saveRDS(sample_info, "rds/sample_info.rds")
        """)



    except Exception as e:
        return {"message": "Error in uploading file", "error": str(e)}
    
    

    return {"message": "file uploadeded & Processed successfully!" ,"count_data": file1_path, "meta_data": file2_path}   




@router.get('/analyze')
async def analyze(user_info: dict = Depends(verify_token)):
    try:

        robjects.r['setwd'](R_CODE_DIRECTORY)

        run_r_script("analyze.R", [str(user_info['user_id'])])
    except Exception as e:
        return {"message": "Error in analyzing file", "error": str(e)}
    
    return {"message": "Analysis completed successfully!", "results" : [
        {"file": f"{BASE_URL}/figures/{user_info['user_id']}/Boxplot_denorm.png", "title": "Denormalized Boxplot"},
        {"file": f"{BASE_URL}/figures/{user_info['user_id']}/Boxplot_norm.png", "title": "Normalized Boxplot"},
        {"file": f"{BASE_URL}/figures/{user_info['user_id']}/htree_denorm.png", "title": "Denormalized Hierarchical Tree"},
        {"file": f"{BASE_URL}/figures/{user_info['user_id']}/htree_norm.png", "title": "Normalized Hierarchical Tree"},
        {"file": f"{BASE_URL}/figures/{user_info['user_id']}/PCA_denorm.png", "title": "Denormalized PCA"},
        {"file": f"{BASE_URL}/figures/{user_info['user_id']}/PCA_norm.png", "title": "Normalized PCA"},
        {"file": f"{BASE_URL}/figures/{user_info['user_id']}/tSNE_denorm.png", "title": "Denormalized tSNE"},
        {"file": f"{BASE_URL}/figures/{user_info['user_id']}/tSNE_norm.png", "title": "Normalized tSNE"},
        {"file": f"{BASE_URL}/figures/{user_info['user_id']}/UMAP_denorm.png", "title": "Denormalized UMAP"},
        {"file": f"{BASE_URL}/figures/{user_info['user_id']}/UMAP_norm.png", "title": "Normalized UMAP"},

        {"file": f"{BASE_URL}/figures/{user_info['user_id']}/Boxplot_denorm.pdf", "title": "Denormalized Boxplot"},
        {"file": f"{BASE_URL}/figures/{user_info['user_id']}/Boxplot_norm.pdf", "title": "Normalized Boxplot"},
        {"file": f"{BASE_URL}/figures/{user_info['user_id']}/htree_denorm.pdf", "title": "Denormalized Hierarchical Tree"},
        {"file": f"{BASE_URL}/figures/{user_info['user_id']}/htree_norm.pdf", "title": "Normalized Hierarchical Tree"},
        {"file": f"{BASE_URL}/figures/{user_info['user_id']}/PCA_denorm.pdf", "title": "Denormalized PCA"},
        {"file": f"{BASE_URL}/figures/{user_info['user_id']}/PCA_norm.pdf", "title": "Normalized PCA"},
        {"file": f"{BASE_URL}/figures/{user_info['user_id']}/tSNE_denorm.pdf", "title": "Denormalized tSNE"},
        {"file": f"{BASE_URL}/figures/{user_info['user_id']}/tSNE_norm.pdf", "title": "Normalized tSNE"},
        {"file": f"{BASE_URL}/figures/{user_info['user_id']}/UMAP_denorm.pdf", "title": "Denormalized UMAP"},
        {"file": f"{BASE_URL}/figures/{user_info['user_id']}/UMAP_norm.pdf", "title": "Normalized UMAP"},

        {
            "file": f"{BASE_URL}/files/{user_info['user_id']}/Normalized_Count_Data.csv",
            "title": "Normalized Count Data"
        },
        {
            "file": f"{BASE_URL}/files/{user_info['user_id']}/count_data.csv",
            "title": "Count Data"
        },
        {
            "file": f"{BASE_URL}/files/{user_info['user_id']}/meta_data.csv",
            "title": "Meta Data"
        }


    
    ]}
       
    

@router.post('/remove_outliers')
async def remove_outliers(data: OutlierSchema, user_info: dict = Depends(verify_token)):

    try:

        genes = data.genes
        genes_str = 'c(' + ', '.join([f'"{gene}"' for gene in genes]) + ')'

        print(genes_str)

        # change the data in the RDS and 

        ot = robjects.r(
        f"""
            user_data <- readRDS("code/{user_info['user_id']}/rds/count_data.rds")
            user_sample_info <- readRDS("code/{user_info['user_id']}/rds/sample_info.rds")

            user_data <- user_data[, !colnames(user_data) %in% {genes_str}]
            user_sample_info <- user_sample_info[!rownames(user_sample_info) %in% {genes_str}, , drop = FALSE]

            print(head(user_data))

            saveRDS(user_data, "code/{str(user_info['user_id'])}/rds/count_data.rds")
            saveRDS(user_sample_info, "code/{str(user_info['user_id'])}/rds/sample_info.rds")
        """)


        print(ot)

        return {"message": "Outliers removed successfully!"}
    
    except Exception as e:
        return {"message": "Error in removing outliers", "error": str(e)}
    



    
    



@router.get("/test")
async def test():
    # os.makedirs(FILE_DIR, exist_ok=True)
    cmd = ["Rscript", os.path.join(R_CODE_DIRECTORY, "test.R")]
    robjects.r['setwd'](R_CODE_DIRECTORY + "/13")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print(stdout.decode())
    # run_r_script("test.R")



def run_r_script(script_name, args=None):
    cmd = ["Rscript", os.path.join(R_CODE_DIRECTORY, script_name)]
    # print(...args)
    if args:
        cmd.extend(args)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print(stdout.decode())
    if process.returncode != 0:
        raise Exception(f"R script failed: {stderr.decode()}")

    return stdout.decode()




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