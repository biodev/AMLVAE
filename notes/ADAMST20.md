# Mutation

In the MDS dataset, there are 12 total mutations in the ADAMST20 gene. Of these 8 are in cluster 1 (128 latent dim vae version) and 4 are not, which receives a q-value of 0.017. 


| array_id  | symbol   | vaf    | consequence        |
| --------- | -------- | ------ | ------------------ |
| MLL_10702 | ADAMTS20 | 0.5248 | missense_variant   |
| MLL_10786 | ADAMTS20 | 0.5823 | missense_variant   |
| MLL_10953 | ADAMTS20 | 0.4314 | missense_variant   |
| MLL_11044 | ADAMTS20 | 0.4154 | missense_variant   |
| MLL_11087 | ADAMTS20 | 0.5146 | missense_variant   |
| MLL_11232 | ADAMTS20 | 0.4375 | missense_variant   |
| MLL_11289 | ADAMTS20 | 0.1494 | missense_variant   |
| MLL_11319 | ADAMTS20 | 0.1848 | stop_gained        |
| MLL_11360 | ADAMTS20 | 0.5254 | missense_variant   |
| MLL_11363 | ADAMTS20 | 0.4545 | missense_variant   |
| MLL_11414 | ADAMTS20 | 0.4783 | missense_variant   |
| MLL_11054 | ADAMTS20 | 0.2456 | frameshift_variant |


They are located throughout the gene, but exons only (may be the assay used). 

There are a number of other mutations that correlate with ADAMST20 mutation. 


| symbol | spearman |
| ------ | -------- |
| DST    | 0.175    |
| MYBPC3 | 0.160    |
| ABCA7  | 0.160    |
| PHF6   | 0.160    |
| NBAS   | 0.160    |
| FLNB   | 0.152    |
| SCN10A | 0.152    |
| LAMA3  | 0.132    |
| DNAH7  | 0.132    |


In cluster 1, are there any other mutations that cause similar expression patterns - OR logic that may describe ADAMST20 role? 

What exppression patterns are caused/associated by ADAMST20

The ADAMST20 mutations are significantly associated with MDS RAEB if we group types 1 and 2. 

google: Refractory Anemia with Excess Blasts (RAEB) is a severe subtype of Myelodysplastic Syndrome (MDS) characterized by cytopenias, multilineage dysplasia, and 5% to 19% blasts in the bone marrow or blood.  It accounts for approximately 30–40% of MDS cases and primarily affects adults over 50, carrying a significant risk of transformation into Acute Myeloid Leukemia (AML).

cytopenias: reduction in the number of mature red blood cells in circulation 

multilineage dysplasia: affects all 3 myeloid lineages 



# CNV

There are 5 ADAMST20 copy number variations, with 4 of them being amplifications and 1 being loss. Not significantly correlated with cluster 1. 


| array_id  | cluster | log2_cr  | call |
| --------- | ------- | -------- | ---- |
| MLL_10840 | 1       | 0.147264 | +    |
| MLL_10833 | 3       | 0.315232 | +    |
| MLL_10868 | 4       | 0.224147 | +    |
| MLL_11111 | 2       | 0.148993 | +    |
| MLL_10861 | 2       | -0.67601 | -    |


# SV

There are 4 patients with structural variants that are either from or to ADAMST20, with one potential fusion with KIF1B. 


|     | array_id  | from_gene   | to_gene     | cluster | potential_fusions |
| --- | --------- | ----------- | ----------- | ------- | ----------------- |
| 0   | MLL_11089 | nan         | ADAMTS20(-) | 1       | nan               |
| 1   | MLL_11224 | ADAMTS20(-) | nan         | 7       | nan               |
| 2   | MLL_11578 | KIF1B(+)    | ADAMTS20(-) | 7       | ADAMTS20::KIF1B   |
| 3   | MLL_10981 | ADAMTS20(-) | nan         | 9       | nan               |


# Expression

Cluster 1 has slightly higher ADAMST20 expression than the other clusters (mannwhitney p-value: 0.01). Of the genes that STRINGDB reports have a functional interactions with ADAMST20, ADAMST20 expression correlates with 3 of them (ACAN, SCARB1, SLC45A2): 


| gene    | spearman_r |
| ------- | ---------- |
| ACAN    | 0.240715   |
| B3GLCT  | 0.063549   |
| CD36    | 0.013958   |
| FURIN   | -0.068544  |
| KITLG   | 0.067118   |
| POFUT2  | 0.047384   |
| SCARB1  | 0.187888   |
| SCARB2  | 0.000633   |
| SLC45A2 | 0.237661   |
| SSBP2   | 0.086360   |


Comparing ADAMST20 expression to all other genes in the expression dataset, the top 20 most significant correlations are shown below: 


| gene         | spearman_r | q_value  |
| ------------ | ---------- | -------- |
| PNMAL2       | 0.331      | 7.68e-15 |
| TRIM46       | 0.334      | 7.68e-15 |
| ZFHX2        | 0.325      | 1.62e-14 |
| KNDC1        | 0.326      | 1.62e-14 |
| SLC52A3      | 0.316      | 9.65e-14 |
| ALDH3B2      | 0.317      | 9.65e-14 |
| AFAP1-AS1    | 0.316      | 9.65e-14 |
| PP14571      | 0.314      | 1.03e-13 |
| ESPNL        | 0.314      | 1.03e-13 |
| KRT4         | 0.315      | 1.03e-13 |
| TDRD6        | 0.313      | 1.14e-13 |
| ADCY8        | 0.313      | 1.19e-13 |
| SLC22A11     | 0.310      | 1.91e-13 |
| LINC00323    | 0.309      | 1.96e-13 |
| CCDC85C      | 0.310      | 1.96e-13 |
| SIX5         | 0.310      | 1.96e-13 |
| SLC8A2       | 0.306      | 4.21e-13 |
| CASC10       | 0.305      | 4.92e-13 |
| LOC100289580 | 0.300      | 1.33e-12 |
| DLGAP3       | 0.299      | 1.63e-12 |


# ADAMST20 mutations correlate with gene expression

Several genes are significantly different in patients that have ADAMST20 mutations. 


| gene         | mean_log2_FPKM_mut | mean_log2_FPKM_wt | q_value  |
| ------------ | ------------------ | ----------------- | -------- |
| MIR1302-8    | 0.213049           | 0                 | 1.07e-09 |
| MIR19B1      | 1.58e-05           | 0                 | 1.07e-09 |
| LOC101929148 | 0.002188           | 0                 | 1.07e-09 |
| MIR3140      | 0.520214           | 0                 | 1.07e-09 |
| HSFY2        | 0.000452           | 1.27e-05          | 0.000354 |
| HSFY1        | 0.015694           | 0.000489          | 0.000354 |
| MIR3180-5    | 0.552962           | 0.035286          | 0.000590 |
| LINC01339    | 0.026599           | 0.000211          | 0.000943 |
| SPANXN4      | 0.035493           | 0.002735          | 0.0113   |
| TBL1Y        | 0.046727           | 0.008117          | 0.0141   |
| ARHGAP26-AS1 | 0.032361           | 0.004665          | 0.0237   |
| LINC00965    | 0.015481           | 0.002264          | 0.0322   |


---

Is cluster 1 stable across multiple VAE parameters? 
WGCNA - ADAMST20 ? 
ADAMST20, is it targetable? 
GSEA on cluster1 ADAMST20  
Network analysis 



can we predict ADAMST20/cluster 1 from clinical mutations? 