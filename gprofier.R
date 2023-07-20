library(gprofiler2)
#
gene_list_zebrafish <- read.csv("/local/gaurav/Paper/GO_term/manuscript/zebrafish_genes.csv")
#gene_list <- levels(droplevels(gene_list_zebrafish$Gene.names))

gostres <- gost(query = gene_list_zebrafish$Gene.names, 
                organism = "drerio", ordered_query = TRUE, 
                multi_query = FALSE, significant = TRUE, exclude_iea = FALSE, 
                measure_underrepresentation = FALSE, evcodes = FALSE, 
                user_threshold = 0.05, correction_method = "g_SCS", 
                domain_scope = "annotated", custom_bg = NULL, 
                numeric_ns = "", sources = NULL, as_short_link = FALSE)



df <- apply(gostres$result,2,as.character)

write.csv(df,"/local/gaurav/Paper/GO_term/manuscript/zebrafish_GO_terms_gproflier.csv")

gene_list_hema <- read.csv("/local/gaurav/Paper/GO_term/manuscript/zebrafish_genes.csv")
#gene_list_hema <- levels(droplevels(gene_list_hema$Gene.names))

gostres_hema <- gost(query = gene_list_hema$Gene.names, 
                organism = "mmusculus", ordered_query = TRUE, 
                multi_query = FALSE, significant = TRUE, exclude_iea = FALSE, 
                measure_underrepresentation = FALSE, evcodes = FALSE, 
                user_threshold = 0.05, correction_method = "g_SCS", 
                domain_scope = "annotated", custom_bg = NULL, 
                numeric_ns = "", sources = NULL, as_short_link = FALSE)


df <- apply(gostres_hema$result,2,as.character)

write.csv(df,"/local/gaurav/Paper/GO_term/manuscript/hema_GO_terms_gproflier.csv")

