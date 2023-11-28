import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# set plotting options
plt.rc('xtick', labelsize=18)  # fontsize of the tick labels
plt.rc('ytick', labelsize=18)  # fontsize of the tick labels
fontsize_pt = plt.rcParams['ytick.labelsize'] # get the tick label font size
dpi = 72.2


def average_regions(df):
    '''
    Function to average the HPA expression values over the Desikan killany regions
    :param df: dataframe containing the HPA expression values after pivoting on the brain tissues (size: NxM, N = number of genes, M = number of brain tissues)
    :return: df: updated dataframe where the expression values are averaged over brain regions in order to mimic Desikan-Killany brain parcellation
    ylab: long brain regions name
    '''
    anteriorcingulate = ['anterior cingulate cortex, supragenual-dorsal',
                         'anterior cingulate cortex, supragenual-ventral', 'middle cingulate cortex']
    rostralanteriorcingulate = ['anterior cingulate gyrus, pregenual-dorsal',
                                'anterior cingulate gyrus, pregenual-ventral']
    isthmucingulate = ['anterior cingulate gyrus, subgenual']
    insula = ['anterior insular cortex, dorsal', 'anterior insular cortex, ventral', 'posterior insular cortex']
    paracentral = ['paracentral lobule, anterior', 'paracentral lobule, posterior']
    postcentral = ['postcentral gyrus', 'postcentral gyrus, dorsal', 'postcentral gyrus, middle',
                   'postcentral gyrus, ventral']
    posteriorcingulate = ['posterior cingulate cortex','posterior cingulate cortex, dorsal','posterior cingulate cortex, ventral']

    precentral = ['precentral gyrus', 'precentral gyrus, dorsal', 'precentral gyrus, middle',
                  'precentral gyrus, ventral']
    transversaltemporal = ['transversal temporal gyrus, anterior', 'transversal temporal gyrus, posterior']
    hippocampus = ['hippocampus, CA1', 'hippocampus, CA2', 'hippocampus, CA3', 'hippocampus']
    pallidum = ['globus pallidus, externus', 'globus pallidus, internus']
    thalamus = ['anterior thalamic nucleus, ventral', 'centromedial thalamic nucleus', 'lateral thalamic nuclei',
                'medial dorsal thalamic nucleus', 'posterior thalamic nucleus',
                'ventral posterolateral thalamic nucleus', 'ventral posteromedial thalamic nucleus',
                'ventral thalamic nuclei', 'ventromedial nucleus']

    df['anteriorcingulate'] = df[anteriorcingulate].mean(axis=1)
    df.drop(anteriorcingulate, axis=1, inplace=True)
    df['posteriorcingulate'] = df[posteriorcingulate].mean(axis=1)
    df.drop(posteriorcingulate, axis=1, inplace=True)
    df['rostralanteriorcingulate'] = df[rostralanteriorcingulate].mean(axis=1)
    df.drop(rostralanteriorcingulate, axis=1, inplace=True)
    df['isthmucingulate'] = df[isthmucingulate].mean(axis=1)
    df.drop(isthmucingulate, axis=1, inplace=True)
    df['insula'] = df[insula].mean(axis=1)
    df.drop(insula, axis=1, inplace=True)
    df['paracentral'] = df[paracentral].mean(axis=1)
    df.drop(paracentral, axis=1, inplace=True)
    df['postcentral'] = df[postcentral].mean(axis=1)
    df.drop(postcentral, axis=1, inplace=True)
    df['precentral'] = df[precentral].mean(axis=1)
    df.drop(precentral, axis=1, inplace=True)
    df['transversaltemporal'] = df[transversaltemporal].mean(axis=1)
    df.drop(transversaltemporal, axis=1, inplace=True)
    df['hipp'] = df[hippocampus].mean(axis=1)
    df.drop(hippocampus, axis=1, inplace=True)
    df['pallidum'] = df[pallidum].mean(axis=1)
    df.drop(pallidum, axis=1, inplace=True)
    df['thalamus'] = df[thalamus].mean(axis=1)
    df.drop(thalamus, axis=1, inplace=True)

    ylab = ['amygdala', 'caudate', 'cerebellar ctx', 'cuneus', 'entorhinal ctx', 'frontal pole', \
            'fusiform gyrus', 'pars opercularis', 'pars orbitalis', 'pars triangularis', \
            'inferior parietal gyrus', 'inferior temporal gyrus', 'lingual gyrus', 'middle frontal gyrus', \
            'middle temporal gyrus', 'accumbens', 'occipital ctx', 'lateral orbitofrontal gyrus',
            'medial orbitofrontal gyrus', \
            'parahippocampal gyrus', 'precuneus', 'putamen', 'superior frontal gyrus', 'superior parietal gyrus', \
            'superior temporal gyrus', 'supramarginal gyrus', 'temporal pole', 'anterior cingulate gyrus', \
            'posterior cingulate gyrus', 'rostral anterior cingulate gyrus', 'isthmucingulate gyrus', \
            'insular ctx', 'paracentral gyrus', 'postcentral gyrus', 'precentral gyrus', 'transverse temporal gyrus', \
            'hippocampus', 'pallidum', 'thalamus']

    return df, ylab

save_path = './'
hpa_path = './'

df = pd.read_csv(hpa_path+"/rna_tissue_hpa.tsv", sep='\t')
df.to_csv(hpa_path+"/rna_tissue_hpa.csv")

# normalization
df[df[['nTPM']] < 1] = np.nan
df['nTPM'] = np.log2(df['nTPM']).replace(np.nan,0)

# Select the genes of interest
Hetionet_genes = ['VEGFA', 'TF', 'PPARGC1A', 'CDH12', 'CHAT','PTGS2','LPL','CYP2D6','ABCC2','AKAP13','BDNF','DPYD']
df_Hetio = df.loc[df['Gene name'].isin(Hetionet_genes)]

# select brain tissues from HPA csv file
ctx_brain_tissues = ['anterior cingulate cortex, supragenual-dorsal','anterior cingulate cortex, supragenual-ventral','anterior cingulate gyrus, pregenual-dorsal',
                     'anterior cingulate gyrus, pregenual-ventral','anterior cingulate gyrus, subgenual','anterior insular cortex, dorsal',
                     'anterior insular cortex, ventral','frontopolar cortex','fusiform gyrus',
                     'inferior frontal gyrus, opercular','inferior frontal gyrus, orbital','inferior frontal gyrus, triangular','inferior parietal lobule',
                     'inferior temporal gyrus','lingual gyrus','middle cingulate cortex','middle frontal gyrus','middle temporal gyrus',
                     'occipital cortex','orbitofrontal gyrus, lateral','orbitofrontal gyrus, medial',
                     'paracentral lobule, anterior','paracentral lobule, posterior',
                     'postcentral gyrus','postcentral gyrus, dorsal','postcentral gyrus, middle',
                     'postcentral gyrus, ventral','posterior cingulate cortex','posterior cingulate cortex, dorsal','posterior cingulate cortex, ventral',
                     'posterior insular cortex','precentral gyrus','precentral gyrus, dorsal','precentral gyrus, middle','precentral gyrus, ventral',
                     'precuneus','superior frontal gyrus','superior parietal lobule','superior temporal gyrus',
                     'supramarginal gyrus','temporal pole','transversal temporal gyrus, anterior','transversal temporal gyrus, posterior',]
HPF_brain_tissues = ['hippocampus','entorhinal gyrus','parahippocampal cortex','hippocampus, CA1','hippocampus, CA2','hippocampus, CA3']
AMY_brain_tissues = ['amygdala']
BG_brain_tissues = ['caudate nucleus','globus pallidus, externus','globus pallidus, internus','nucleus accumbens','putamen']
TH_brain_tissues = ['anterior thalamic nucleus, ventral','centromedial thalamic nucleus','lateral thalamic nuclei','medial dorsal thalamic nucleus','posterior thalamic nucleus','ventral posterolateral thalamic nucleus','ventral posteromedial thalamic nucleus','ventral thalamic nuclei','ventromedial nucleus']
MB_brain_tissues = ['cuneiform nucleus']
Cer_brain_tissues = ['cerebellar cortex']

brain_tissues = ctx_brain_tissues + HPF_brain_tissues + AMY_brain_tissues + BG_brain_tissues + TH_brain_tissues + Cer_brain_tissues + MB_brain_tissues

# Select only the interesting brain tissues from the full dataframe
df_Hetio_brain = df_Hetio.loc[df_Hetio['Tissue'].isin(brain_tissues)]

# Pivot the HPA dataframe on brain tissues and average expression over different tissue to mimic the Desikan Killany parcellation
df_Hetio_brain_pivot_nTPM = pd.pivot_table(df_Hetio_brain,values='nTPM',index='Gene name',columns=['Tissue'])
df_Hetio_brain_pivot_nTPM, ylab = average_regions(df_Hetio_brain_pivot_nTPM)

# Generate and save the heatmap figure
vmin = 0
vmax = 10
fig, ax = plt.subplots(figsize=(28,9))
g = sns.heatmap(df_Hetio_brain_pivot_nTPM, linewidths=.5, ax=ax, cmap='plasma', xticklabels=ylab, vmin=vmin, vmax=vmax)
g.set_yticklabels(g.get_yticklabels(), rotation =0)
g.set_xticklabels(g.get_xticklabels(), rotation =90)
g.set_title('Hetionet genes')
fig.tight_layout()
fig.savefig(save_path+'brain_expression_Hetionet_genes_norm.png')
