# To run this code, first download example files and plink

#Install and load SKAT packages
if (!require("SKAT", quietly = TRUE))
  install.packages("SKAT")
library(SKAT)

#Input parameter
file_name="Example1"
only_significant = TRUE
threshold = 0.2
remove_temporary = TRUE
setwd(".") #set working directory
dir.create("tmp", showWarnings = F)


# To run this code, first download example files and plink


############################## Quality Control

system(paste0("plink --bfile ",file_name," --geno 0.2 --mind 0.1 --maf 0.05 --hwe 1e-06 --make-bed --out tmp\\",file_name,"_QC"))



############################# Compute SKAT
# Generate SSD file
File.Bed = paste0("./tmp/",file_name,"_QC.bed")
File.Bim = paste0("./tmp/",file_name,"_QC.bim")
File.Fam = paste0("./tmp/",file_name,"_QC.fam")
File.SetID = paste0("./",file_name,".SetID")
File.SSD = paste0("./tmp/",file_name,".SSD")      #output
File.Info = paste0("./tmp/",file_name,".SSD.info")  #output
#read FAM with covariates
File.Cov = paste0("./",file_name,".Cov")
FAM_Cov = Read_Plink_FAM_Cov(File.Fam, File.Cov, Is.binary=FALSE)
# To use binary files, you have to generate SSD file first.
Generate_SSD_SetID(File.Bed, File.Bim, File.Fam, File.SetID, File.SSD, File.Info)

# Open the SSD file. After finishing using it, close it.
SSD.INFO = Open_SSD(File.SSD, File.Info) 

X1 = FAM_Cov$Age
X2 = FAM_Cov$Gender
y = FAM_Cov$Phenotype

obj = SKAT_Null_Model(y ~ X1 + X2, out_type="C")
out = SKAT.SSD.All(SSD.INFO, obj)

output.df = out$results
write.table(out$results, file="./SKAT_results.txt", col.names=TRUE, row.names=FALSE, quote=FALSE)

Close_SSD()


############################## Compute gene variant score

snps = read.table(File.SetID)
genes=out$OUT.snp.mac

# Extract SNPs from experiment
if(only_significant){ #take SNPs from significant genes (based on threshold)
  sign_genes = output.df[output.df$P.value <= threshold,]$SetID
  snps = unique(snps[snps$V1 %in% sign_genes,]$V2)
  genes = genes[sign_genes]
} else { #take SNPs from all genes
  snps = snps$V2  
  genes = out$OUT.snp.mac
}

write.table(snps, "tmp/snps.txt", quote=F, row.names = F, col.names = F)
system(paste0("plink --bfile tmp\\",file_name, "_QC --extract tmp\\snps.txt --make-bed --out tmp\\Core"))
system("plink --bfile tmp\\Core --recodeA --out tmp\\Core_recoded")


data= read.table("tmp\\Core_recoded.raw", header=T, sep=" ")
data = data[,-c(1,3,4,5,6)]
data[data == 2]=1
data[is.na(data)]=0

#remove _ in SNPs colnames
newcol = colnames(data)[-1]
newcol=sapply(newcol,function(x) substr(x,0,nchar(x)-2))  
colnames(data)[-1]=newcol

output=setNames(data.frame(matrix(ncol = length(genes)+1, nrow = nrow(data))), c("IID", names(genes)))
output$IID=data$IID  


for (g in 1:length(genes)){
  set=names(genes[[g]])
  
  if (is.vector(data[,set])) { res= data[,set] }
  else {res=apply(data[,set],1,sum)}
  
  output[,g+1]=1.0*res/length(genes[[g]])
}

write.table(output,"genetics_output.raw", quote=F, sep = " ", row.names = F, col.names = T)

if(remove_temporary)
  if (file.exists("tmp")) {
    unlink("tmp",recursive = TRUE)
  }
