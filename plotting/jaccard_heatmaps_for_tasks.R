library(tidyverse)
library(ggnuplot)
library(cowplot)
library(ggpubr)
library(extrafont)
library(RColorBrewer)
library("gridExtra")
library(ggtext)
library(ggthemes)

selectedmethods <- c('Maxu.', 'fair-bpa2', 'fair-exposure', 'fair-epsilon-greedy', 'divtopk')

names(selectedmethods) <- selectedmethods


proportional_task <- read_csv("proportional_task.csv") %>%
  select(subset, method, data_name) %>%
  filter(method != "ta" & 
           method != "bpa" & 
           method != "divtopk" &
           method != "fair-ta" &
           method != "fair-fagins" &
           method != "fair-bpa" &
           method != "fagins") %>%
  mutate(method=recode(method,
                       `bpa2` = "Maxu.")) %>%
  mutate(method = recode(method, !!!selectedmethods, .default = "Fa*ir")) %>%
  mutate(method=recode(method,
                       `fair-bpa2` = "FMCS",
                       `fair-exposure` = "Rexp.",
                       `fair-epsilon-greedy` = "Eps.",
                       `divtopk` = "Dtk")) %>%
  mutate(data_name=recode(data_name,
                       `bank` = "Bank",
                       `credit` = "Credit",
                       `gauss` = "Gauss",
                       `lc` = "Low Corr",
                       `hc` = "High Corr"))



equal_task <- read_csv("equal_task.csv") %>%
  select(subset, method, data_name) %>%
  filter(method != "ta" &
           method != "bpa" &
           method != "divtopk" &
           method != "fair-ta" &
           method != "fair-fagins" &
           method != "fair-bpa" &
           method != "fagins" &
           method != "divtopk" ) %>%
  mutate(method=recode(method,
                       `bpa2` = "Maxu.")) %>%
  mutate(method = recode(method, !!!selectedmethods, .default = "Fa*ir")) %>%
  mutate(method=recode(method,
                       `fair-bpa2` = "FMCS",
                       `fair-exposure` = "Rexp.",
                       `fair-epsilon-greedy` = "Eps.")) %>%
  mutate(data_name=recode(data_name,
                          `bean` = "Bean",
                          `iit` = "IIT",
                          `gauss` = "Gauss",
                          `lc` = "Low Corr",
                          `hc` = "High Corr"))



rooney_task <- read_csv("rooney_taskadult.csv") %>%
  select(subset, method, r_value) %>%
  filter(method != "ta" &
           method != "bpa" &
           method != "divtopk" &
           method != "fair-ta" &
           method != "fair-fagins" &
           method != "fair-bpa" &
           method != "fagins" &
           method != "divtopk" ) %>%
  mutate(method=recode(method,
                       `bpa2` = "Maxu.")) %>%
  mutate(method = recode(method, !!!selectedmethods, .default = "Fa*ir")) %>%
  mutate(method=recode(method,
                       `fair-bpa2` = "FMCS",
                       `fair-exposure` = "Rexp.",
                       `fair-epsilon-greedy` = "Eps.")) 



high_color <- "#31a354"
j_lims <- c(.2,1)
legend_str <- 'Jaccard'
textsize <- 4
pdfwidth <- 15
pdfheight <- 3
#colors <- "RdYlBu"
colors <- "PiYG"

#define clean function
clean <- function(x){
  dropleft <- gsub("[[]", "", x)
  dropright <- gsub("[]]", "", dropleft)
  dropn <- gsub("[\r\n]", "", dropright)
  
  vect <- as.integer(unlist(strsplit(dropn, split = " ")))
  vect <- vect[!is.na(vect)]
  return(vect)
} 


jaccard <- function(a, b) {
  intersection = length(intersect(a, b))
  union = length(a) + length(b) - intersection
  return (intersection/union)
}

make_maps <- function(dataframe, dataset) {
  

  
  data <-  dataframe %>%
      filter(.data$data_name == .env$dataset) 
  
  
  cross_join <- merge(x = data, y = data, by = NULL)
  
  cross_join$subset.y <- lapply(cross_join$subset.y,clean)
  cross_join$subset.x <- lapply(cross_join$subset.x,clean)
  cross_join$jaccard <- mapply(jaccard,cross_join$subset.x, cross_join$subset.y)
  cross_join$jaccard <- signif(cross_join$jaccard, digits = 2)
  
  
  
    
  plot_p <- ggplot(cross_join, aes(x = method.x, y = method.y)) +
  geom_tile(aes(fill = jaccard), colour = "white") +
  geom_text(aes(label = jaccard), size = textsize) +
  scale_fill_distiller(palette = colors, direction = 1, name = "Jaccard \n Index") +
  theme_minimal_grid()+
  theme(panel.grid = element_blank(),
        plot.margin=unit(c(0.2,0,0,0), "cm"),
        axis.text.x = element_text(angle = 45,hjust=.7))+
  labs(title = glue::glue({dataset}), x = element_blank(), y = element_blank())
  
  
  return(plot_p)}


make_rooney_maps <- function(dataframe, r_val) {
  
  
  
  data <-  dataframe %>%
    filter(.data$r_value == .env$r_val) 
  
  
  cross_join <- merge(x = data, y = data, by = NULL)
  
  cross_join$subset.y <- lapply(cross_join$subset.y,clean)
  cross_join$subset.x <- lapply(cross_join$subset.x,clean)
  cross_join$jaccard <- mapply(jaccard,cross_join$subset.x, cross_join$subset.y)
  cross_join$jaccard <- signif(cross_join$jaccard, digits = 2)
  
  
  
  
  plot_p <- ggplot(cross_join, aes(x = method.x, y = method.y)) +
    geom_tile(aes(fill = jaccard), colour = "white") +
    geom_text(aes(label = jaccard), size = textsize) +
    scale_fill_distiller(palette = colors, direction = 1, name = "Jaccard \n Index") +
    theme_minimal_grid()+
    theme(panel.grid = element_blank(),
          plot.margin=unit(c(0.2,0,0,0), "cm"),
          axis.text.x = element_text(angle = 45,hjust=.7))+
    labs(title = "r = 10", x = element_blank(), y = element_blank())
  
  
  return(plot_p)}



bp <- make_maps(proportional_task, 'Bank')
cp <- make_maps(proportional_task, 'Credit')
gp <- make_maps(proportional_task, 'Gauss')
hp <- make_maps(proportional_task, 'High Corr')
lp <- make_maps(proportional_task, 'Low Corr')
prop_maps <- ggarrange(bp, cp,gp, hp,lp,
          ncol = 5, nrow = 1, common.legend = TRUE, legend = "right")

# ggsave(prop_maps, filename = 'proportional_maps.pdf', device = cairo_pdf,
#        width = pdfwidth, height = pdfheight, units = "in")


be <- make_maps(equal_task, 'Bean')
ie <- make_maps(equal_task, 'IIT')
ge <- make_maps(equal_task, 'Gauss')
he <- make_maps(equal_task, 'High Corr')
le <- make_maps(equal_task, 'Low Corr')
equal_maps <- ggarrange(be, ie,ge, he,le,
                       ncol = 5, nrow = 1, common.legend = TRUE, legend = "right")

# ggsave(equal_maps, filename = 'equal_maps.pdf', device = cairo_pdf,
#        width = pdfwidth, height = pdfheight, units = "in")


r10 <- make_rooney_maps(rooney_task, 10)

