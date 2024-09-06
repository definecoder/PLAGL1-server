

from fastapi import APIRouter, File, UploadFile, Depends
from core.security import verify_token
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import os
import subprocess
from models.schema import OutlierSchema
from core.consts import BASE_URL


pandas2ri.activate()

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

        return {"message": "file uploadeded & Processed successfully!" ,"count_data": file1_path, "meta_data": file2_path}   

    except Exception as e:
        return {"message": "Error in uploading file", "error": str(e)}
    
    

@router.get('/analyze')
async def analyze(user_info: dict = Depends(verify_token)):
    try:

        # robjects.r['setwd'](R_CODE_DIRECTORY)

        run_r_script("analyze.R", [str(user_info['user_id'])])
    except Exception as e:
        return {"message": "Error in analyzing file", "error": str(e)}
    
    return {
        "message": "Analysis completed successfully!",
        "results": {
            "boxplot_denorm_img": f"{BASE_URL}/figures/{user_info['user_id']}/Boxplot_denorm.png",
            "boxplot_norm_img": f"{BASE_URL}/figures/{user_info['user_id']}/Boxplot_norm.png",
            "htree_denorm_img": f"{BASE_URL}/figures/{user_info['user_id']}/htree_denorm.png",
            "htree_norm_img": f"{BASE_URL}/figures/{user_info['user_id']}/htree_norm.png",
            "pca_denorm_img": f"{BASE_URL}/figures/{user_info['user_id']}/PCA_denorm.png",
            "pca_norm_img": f"{BASE_URL}/figures/{user_info['user_id']}/PCA_norm.png",
            "tsne_denorm_img": f"{BASE_URL}/figures/{user_info['user_id']}/tSNE_denorm.png",
            "tsne_norm_img": f"{BASE_URL}/figures/{user_info['user_id']}/tSNE_norm.png",
            "umap_denorm_img": f"{BASE_URL}/figures/{user_info['user_id']}/UMAP_denorm.png",
            "umap_norm_img": f"{BASE_URL}/figures/{user_info['user_id']}/UMAP_norm.png",

            "boxplot_denorm_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/Boxplot_denorm.pdf",
            "boxplot_norm_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/Boxplot_norm.pdf",
            "htree_denorm_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/htree_denorm.pdf",
            "htree_norm_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/htree_norm.pdf",
            "pca_denorm_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/PCA_denorm.pdf",
            "pca_norm_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/PCA_norm.pdf",
            "tsne_denorm_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/tSNE_denorm.pdf",
            "tsne_norm_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/tSNE_norm.pdf",
            "umap_denorm_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/UMAP_denorm.pdf",
            "umap_norm_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/UMAP_norm.pdf",

            "normalized_data_csv": f"{BASE_URL}/files/{user_info['user_id']}/Normalized_Count_Data.csv",
            "count_data_csv": f"{BASE_URL}/files/{user_info['user_id']}/count_data.csv",
            "meta_data_csv": f"{BASE_URL}/files/{user_info['user_id']}/meta_data.csv"
        }
    }




@router.post('/remove_outliers')
async def remove_outliers(data: OutlierSchema, user_info: dict = Depends(verify_token)):

    try:

        robjects.r['setwd'](R_CODE_DIRECTORY)

        print(f"current python wd: {os.getcwd()}")

        file_path = f"{user_info['user_id']}/rds/genes.rds"
        

        print(robjects.r('getwd()'))

        genes = data.genes
        r_genes_list = robjects.StrVector(genes)
        r_saveRDS = robjects.r['saveRDS']
        r_saveRDS(r_genes_list, file=file_path)

        # robjects.r['setwd'](R_CODE_DIRECTORY)
        # robjects.r['setwd'](R_CODE_DIRECTORY)
        print("ekhane ashenai")
        run_r_script("remove_outlier.R", [str(user_info['user_id'])])
        run_r_script("analyze.R", [str(user_info['user_id'])]) 


        return {"message": "Outliers removed successfully!", 
                "results": {
                "boxplot_denorm_img": f"{BASE_URL}/figures/{user_info['user_id']}/Boxplot_denorm.png",
                "boxplot_norm_img": f"{BASE_URL}/figures/{user_info['user_id']}/Boxplot_norm.png",
                "htree_denorm_img": f"{BASE_URL}/figures/{user_info['user_id']}/htree_denorm.png",
                "htree_norm_img": f"{BASE_URL}/figures/{user_info['user_id']}/htree_norm.png",
                "pca_denorm_img": f"{BASE_URL}/figures/{user_info['user_id']}/PCA_denorm.png",
                "pca_norm_img": f"{BASE_URL}/figures/{user_info['user_id']}/PCA_norm.png",
                "tsne_denorm_img": f"{BASE_URL}/figures/{user_info['user_id']}/tSNE_denorm.png",
                "tsne_norm_img": f"{BASE_URL}/figures/{user_info['user_id']}/tSNE_norm.png",
                "umap_denorm_img": f"{BASE_URL}/figures/{user_info['user_id']}/UMAP_denorm.png",
                "umap_norm_img": f"{BASE_URL}/figures/{user_info['user_id']}/UMAP_norm.png",

                "boxplot_denorm_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/Boxplot_denorm.pdf",
                "boxplot_norm_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/Boxplot_norm.pdf",
                "htree_denorm_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/htree_denorm.pdf",
                "htree_norm_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/htree_norm.pdf",
                "pca_denorm_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/PCA_denorm.pdf",
                "pca_norm_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/PCA_norm.pdf",
                "tsne_denorm_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/tSNE_denorm.pdf",
                "tsne_norm_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/tSNE_norm.pdf",
                "umap_denorm_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/UMAP_denorm.pdf",
                "umap_norm_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/UMAP_norm.pdf",

                "normalized_data_csv": f"{BASE_URL}/files/{user_info['user_id']}/Normalized_Count_Data.csv",
                "count_data_csv": f"{BASE_URL}/files/{user_info['user_id']}/count_data.csv",
                "meta_data_csv": f"{BASE_URL}/files/{user_info['user_id']}/meta_data.csv"
            }
        }
    
    except Exception as e:
        return {"message": "Error in removing outliers", "error": str(e)}
    



    
@router.get('/conditions')
async def show_conditions(user_info: dict = Depends(verify_token)):
    try:

        pandas2ri.activate()
        robjects.r['setwd'](R_CODE_DIRECTORY + f"/{user_info['user_id']}")
        readRDS = robjects.r['readRDS']
        condition_vector = readRDS("rds/condition.rds")
        # Extract factor levels
        factor_levels = robjects.r['levels'](condition_vector)        
        # Convert factor levels to a Python list
        factor_levels_list = list(factor_levels)

        # robjects.r['setwd'](R_CODE_DIRECTORY)
        return {"options": factor_levels_list}

        # run_r_script("conditions.R", [str(user_info['user_id'])])
    except Exception as e:
        return {"message": "Error in showing conditions", "error": str(e)}



@router.get('/select_condition/{option}')
async def select_condition(option: int, user_info: dict = Depends(verify_token)):
    try:

        robjects.r['setwd'](R_CODE_DIRECTORY + f"/{user_info['user_id']}")

        robjects.r(
        f"""
            library(DESeq2)
            condition <- readRDS("rds/condition.rds")
            dds <- readRDS("rds/dds.rds")
            condition <- as.data.frame(condition)
            ref_level <- as.character(condition[{option},]) 
            dds$Treatment <- relevel(dds$Treatment, ref = ref_level)
            dds <- DESeq(dds)
            coeff_names <- as.data.frame(resultsNames(dds))
            saveRDS(coeff_names, "rds/coeff_names.rds")
            saveRDS(ref_level, "rds/ref_level.rds")
            saveRDS(dds, "rds/dds.rds")
        """)

        pandas2ri.activate()

        readRDS = robjects.r['readRDS']
        coeff_vector = readRDS("rds/coeff_names.rds")

        coeff_vector = list(coeff_vector[0])

        # print(coeff_vector[0])

        coeffs = []

        for coeff in coeff_vector:
            coeffs.append(str(coeff))
    
        return {"options": coeffs}
    
    except Exception as e:
        return {"message": "Error in selecting condition", "error": str(e)}
    


    


# this will let the user to select the condition and it will show the coeffs 

@router.get('/volcano/{coeff}')
async def volcano_plot(coeff: str, user_info: dict = Depends(verify_token)):

    try:

        robjects.r['setwd'](R_CODE_DIRECTORY + f"/{user_info['user_id']}")

        robjects.r(
        f"""
            coeff_names <- readRDS("rds/coeff_names.rds")
            X <- coeff_names[{coeff},]
            saveRDS(X, "rds/X.rds")
        """)

        print("etotuku no problem")

        run_r_script("volcano.R", [str(user_info['user_id'])])
        # f"{BASE_URL}/figures/{user_info['user_id']}/htree_norm.pdf",

        return {"message": "Volcano plot generated successfully!",
                "results": {
                    "histogram_img": f"{BASE_URL}/figures/{user_info['user_id']}/histogram_pvalues.png",
                    "histogram_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/histogram_pvalues.pdf",
                    "volcano_img": f"{BASE_URL}/figures/{user_info['user_id']}/volcano_plot.png",
                    "volcano_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/volcano_plot.pdf",
                    "upregulated_genes" : f"{BASE_URL}/files/{user_info['user_id']}/Upregulated_padj.csv",
                    "downregulated_genes" : f"{BASE_URL}/files/{user_info['user_id']}/Downregulated_padj.csv",
                    "resLFC" : f"{BASE_URL}/files/{user_info['user_id']}/resLFC.csv"
                }
               }
    
    except Exception as e:
        return {"message": "Error in generating volcano plot", "error": str(e)}




# user will pass the coeff and it will show the volcano plot

@router.post('/highlighted_volcano')
async def highlighted_volcano(data: OutlierSchema, user_info: dict = Depends(verify_token)):

    try:
        robjects.r['setwd'](R_CODE_DIRECTORY + f"/{user_info['user_id']}")
        r_genes_to_highlight = robjects.StrVector(data.genes)
        r_saveRDS = robjects.r['saveRDS']
        r_saveRDS(r_genes_to_highlight, file="rds/genes_to_highlight.rds")
        run_r_script("highlighted_volcano.R", [str(user_info['user_id'])])

        return {"message": "Highlighted Volcano plot generated successfully!",
                "results":{
                    "highlighted_volcano_img": f"{BASE_URL}/figures/{user_info['user_id']}/highlighted_volcano.png",
                    "highlighted_volcano_pdf": f"{BASE_URL}/figures/{user_info['user_id']}/highlighted_volcano.pdf"
                }
               }

        
    except Exception as e:
        return {"message": "Error in generating highlighted volcano plot", "error": str(e)}
    






    






@router.get("/test")
async def test():
    # os.makedirs(FILE_DIR, exist_ok=True)
    robjects.r['setwd'](R_CODE_DIRECTORY + "/1")
    ot = robjects.r(
        f"""
            md <- readRDS("rds/count_data.rds")
            head(md)
        """
    )

    robjects.r['setwd'](R_CODE_DIRECTORY)

    print(ot)

    return {"message": "Tested successfully!"}
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