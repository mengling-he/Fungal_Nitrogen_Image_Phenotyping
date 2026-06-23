library(ggplot2)
library(tidyr)
library(dplyr)
library(emmeans)
library(rstudioapi)
setwd(dirname(getActiveDocumentContext()$path))
library(readxl)

#----read data----
df_resp_0 <- read_excel('../../../Data/MV_final_cleaned_combined.xlsx', sheet = 'wide')

df_resp <- df_resp_0 %>%
  dplyr::select(label, respiration) %>%
  rename(plate = label) %>%
  separate(plate, into = c("num", "Fungal_Strain", "Nitrogen_Level", "rep", "extra"), 
           sep = "_", remove = FALSE, extra = "drop")


colnames(df_resp)
font_size =16
p = ggplot(df_resp, aes(x = Nitrogen_Level, y = respiration, color = Fungal_Strain)) +
  geom_boxplot(
    position = position_dodge(width = 0.7),
    width = 0.6,
    fill = NA,            
    outlier.shape = NA
  ) +
  geom_jitter(
    position = position_jitterdodge(
      jitter.width = 0.15,
      dodge.width = 0.7
    ),
    alpha = 0.7,
    size = 2
  ) +
  theme_bw()+
  theme(
    axis.title = element_text(size = font_size+2),      # x/y label
    axis.text = element_text(size = font_size),       # tick labels
    legend.title = element_text(size = font_size),    # legend title
    legend.text = element_text(size = font_size-2)      # legend items
  )
outdir = "../results/plot/"
ggsave(paste0(outdir, "resp_box_dist.png"), plot = p, width = 10, height = 8)

#----LMM 1----
# each plate is independent from each other, so use lm is ok here. 
lmm_model_resp <- lm(
  respiration ~ Fungal_Strain * Nitrogen_Level,
  data = df_resp
)


summary(lmm_model_resp)
print(anova(lmm_model_resp))

emm_fung <- emmeans(lmm_model_resp,
               ~ Fungal_Strain | Nitrogen_Level)
pairs(emm_fung)

emm_nitro <- emmeans(lmm_model_resp,
               ~ Nitrogen_Level | Fungal_Strain)
pairs(emm_nitro)




par(mfrow = c(2,2))
plot(lmm_model_resp)
aov1 <- aov(
  respiration ~ Fungal_Strain * Nitrogen_Level,
  data = df_resp
)
aov1




#----LMM 2 : delete the outliers----
df_resp_reduced <- df_resp[df_resp$respiration<5,]
p2 = ggplot(df_resp_reduced, aes(x = Nitrogen_Level, y = respiration, color = Fungal_Strain)) +
  geom_boxplot(
    position = position_dodge(width = 0.7),
    width = 0.6,
    fill = NA,
    outlier.shape = NA
  ) +
  geom_jitter(
    position = position_jitterdodge(
      jitter.width = 0.15,
      dodge.width = 0.7
    ),
    alpha = 0.7,
    size = 2
  ) +
  theme_bw() +
  theme(
    axis.title = element_text(size = font_size+2),      # x/y label
    axis.text = element_text(size = font_size),       # tick labels
    legend.title = element_text(size = font_size),    # legend title
    legend.text = element_text(size = font_size-2)      # legend items
  )
outdir = "../results/plot/"
ggsave(paste0(outdir, "resp_box_dist_no_outlier.png"), plot = p2, width = 10, height = 8)
lmm_model_resp2 <- lm(
  respiration ~ Fungal_Strain * Nitrogen_Level,
  data = df_resp_reduced
)

summary(lmm_model_resp2)
print(anova(lmm_model_resp2))

emm_fung_2 <- emmeans(lmm_model_resp2,
                    ~ Fungal_Strain | Nitrogen_Level)
pairs(emm_fung_2)

emm_nitro_2 <- emmeans(lmm_model_resp2,
                     ~ Nitrogen_Level | Fungal_Strain)
pairs(emm_nitro_2)



aov(
  respiration ~ Fungal_Strain,
  data = df_resp
)
aov(
  respiration ~ Nitrogen_Level,
  data = df_resp
)

aov(
  respiration ~ Fungal_Strain,
  data = df_resp_reduced
)
aov(
  respiration ~ Nitrogen_Level,
  data = df_resp_reduced
)
