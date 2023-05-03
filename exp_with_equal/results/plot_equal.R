library(tidyverse)
library(ggnuplot)
library(cowplot)
library(ggpubr)
library(extrafont)
library(RColorBrewer)
library("gridExtra")
library(ggtext)
library(scales)

squish_trans <- function(from, to, factor) {
  
  trans <- function(x) {
    
    if (any(is.na(x))) return(x)
    
    # get indices for the relevant regions
    isq <- x > from & x < to
    ito <- x >= to
    
    # apply transformation
    x[isq] <- from + (x[isq] - from)/factor
    x[ito] <- from + (to - from)/factor + (x[ito] - to)
    
    return(x)
  }
  
  inv <- function(x) {
    
    if (any(is.na(x))) return(x)
    
    # get indices for the relevant regions
    isq <- x > from & x < from + (to - from)/factor
    ito <- x >= from + (to - from)/factor
    
    # apply transformation
    x[isq] <- from + (x[isq] - from) * factor
    x[ito] <- to + (x[ito] - (from + (to - from)/factor))
    
    return(x)
  }
  
  # return the transformation
  return(trans_new("squished", trans, inv))
}


data <- read_csv("equal_study.csv") %>%
  select(delta_val, utility_ratio, fairness_ratio, wall_time, method, data_name) %>%
  filter(!grepl('protect', method)) %>%
  filter(delta_val < 1) %>%
  filter(data_name == 'hc') %>%
  filter(!grepl('ta', method)) %>%
  filter(method != 'fair-bpa') %>%
    filter(method != 'fair-GBG_bpa')
  


#try to filter to divtopk, fair$share bpa2, fair GBG bpa2, fair Greedy, 




pt_size <- 4 #3
title_size <- 10
linesize <- 1
axistext <- 14


multi_colors <- c('darkviolet', '#009e73', '#56b4e9','#e69f00', '#f0e442', 'red')
multi_shapes <- c(15, 16, 18, 8, 5, 1)



#fairness 
ggplot(data, aes(fill = method, x  = as.numeric(as.character(delta_val)),
                 y = utility_ratio)) +
  geom_bar(stat = 'identity')
  
x_string <- "Delta Value"

ggplot(data, aes(color = method, x  = as.factor(delta_val),
                 y = utility_ratio, shape = method)) +
  geom_point(size = pt_size, position = 'jitter')+
  geom_line(size = linesize) +
  theme_gnuplot()+
  xlab(x_string)+
  ylab("Utility Ratio")+
  theme_gnuplot()+
  theme(legend.position = "top",
        legend.direction = "horizontal",
        axis.title.y = element_text(size = axistext),
        axis.title.x = element_text(size = axistext))+
  ggtitle("Utility")+
  scale_shape_manual(values=multi_shapes)+
  scale_color_manual(values=multi_colors)+
  guides(color=guide_legend(nrow=1))+
  guides(shape = guide_legend(nrow = 1))