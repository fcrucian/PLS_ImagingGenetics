import os
import pandas as pd
import numpy as np

def drop_suffix(self, suffix):
	# Auxiliary function to drop columns suffix
    self.columns = self.columns.str.replace(suffix,'')
    return self

def write_xlsx_freesurfer(path, measure):
	# Function to extract a group excel file from the differenttt Freesurfer .txt files
	# Input: 
	# path -> path to the freesurfer extracted .txt files
	# measure -> 'volume' or 'thickness'. Measure to be extracted from the stats files
	# Output: grouped excel file includingggggg the selected measures for all the subject cohort

	pd.core.frame.DataFrame.drop_suffix = drop_suffix

	lab = pd.read_csv(path+'/fs_default_mod.csv', sep=",")
	lab['ROI'] = lab['ROI'].str.replace(r'ctx-','')

	aseg = pd.read_csv(path+'/aseg_stats.txt',sep="\t").sort_values(by='Measure:volume')
	aparc_r = pd.read_csv(path+'/aparc_stats_'+measure+'_rh.txt',sep="\t").sort_values(by='rh.aparc.'+measure)
	aparc_l = pd.read_csv(path+'/aparc_stats_'+measure+'_lh.txt',sep="\t").sort_values(by='lh.aparc.'+measure)
	aparc_l.drop_suffix('_'+measure)
	aparc_r.drop_suffix('_'+measure)
	aparc_l.columns = aparc_l.columns.str.replace(r'lh_','lh-')
	aparc_r.columns = aparc_r.columns.str.replace(r'rh_','rh-')

	tmp = pd.merge(aseg,aparc_r,left_on='Measure:volume',right_on='rh.aparc.'+measure)
	tmp.drop('rh.aparc.'+measure,axis=1,inplace=True)

	df = pd.merge(tmp, aparc_l,left_on='Measure:volume',right_on='lh.aparc.'+measure)
	df.drop('lh.aparc.'+measure,axis=1,inplace=True)

	ROI = np.array(lab['ROI'])
	#ROI_up = np.zeros([ROI.shape[0]+1])
	ROI_up = []
	ROI_up.append("Measure:volume")
	ROI_up[1::] = ROI

	new = df[ROI_up].copy()
	new = new.rename(columns={'Measure:volume':'PTID'})
	new['PTID'] = new['PTID'].map(lambda x: x.rstrip('.nii'))

	new.to_excel(path+'/freesurfer_features_aseg+'+measure+'.xlsx', index=False)

#### MAIN ####

#run fsl_anat for T1 preprocessing (N.B. FSL required to be installed in the system)
if not os.path.exists("./data/Sample_subj/fsl_anat.anat"):
	os.system('fsl_anat -i ./data/Sample_subj/T1.nii.gz -o ./data/Sample_subj/fsl_anat')

# run Freesurfer for brain segmentation (N.B. Freesurfer required to be installed in the system)
if not os.path.exists("./T1"):
	os.mkdir('./T1')
os.system('cp ./data/Sample_subj/fsl_anat.anat/T1_biascorr.nii.gz ./T1/Sample_subj.nii.gz')
os.chdir('./T1')
os.system('ls *.nii.gz | parallel --jobs 12 recon-all -s {.} -i {} -all -qcache')

# extract freesurfer stats for all the subjects
os.system("asegstats2table --subjects $SUBJECTS_DIR/*.nii --skip --meas volume --tablefile ./data/aseg_stats.txt")
os.system("aparcstats2table --subjects $SUBJECTS_DIR/*.nii --skip --hemi rh --meas thickness --tablefile ./data/aparc_stats_thickness_rh.txt")
os.system("aparcstats2table --subjects $SUBJECTS_DIR/*.nii --skip --hemi lh --meas thickness --tablefile ./data/aparc_stats_thickness_lh.txt")
os.system("aparcstats2table --subjects $SUBJECTS_DIR/*.nii --skip --hemi rh --meas volume --tablefile ./data/aparc_stats_volume_rh.txt")
os.system("aparcstats2table --subjects $SUBJECTS_DIR/*.nii --skip --hemi lh --meas volume --tablefile ./data/aparc_stats_volume_lh.txt")

# organize the freesurfer stats in excel files
path='./data/'
write_xlsx_freesurfer(path, 'thickness')
write_xlsx_freesurfer(path, 'volume')
