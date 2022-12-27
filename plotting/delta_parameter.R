library(tidyverse)
library(ggnuplot)
library(cowplot)
library(ggpubr)
library(extrafont)
library(RColorBrewer)
library("gridExtra")
library(ggtext)


proportional_task <- read_csv("proportional_study_delta.csv") %>%
  select(method, delta_val, data_name, sa_count, ra_count, wall_time,
         fairness_ratio, utility_ratio, position_seen_prop)  %>%
  mutate(method=recode(method,
                             `fair-fagins` = "fagins",
                             `fair-ta` = "ta",
                             `fair-bpa` = "bpa",
                            `fair-bpa2` = "bpa2"))%>%
  filter(method != "fagins")


equal_task <- read_csv("equal_study_delta.csv") %>%
  select(method, delta_val, data_name, sa_count, ra_count, wall_time,
         fairness_ratio, utility_ratio, position_seen_prop)  %>%
  mutate(method=recode(method,
                       `fair-fagins` = "fagins",
                       `fair-ta` = "ta",
                       `fair-bpa` = "bpa",
                       `fair-bpa2` = "bpa2"))%>%
  filter(method != "fagins")

pdfwidth = 8
pdfheight = 5
pt_size <- 2.5 #3
title_size <- 10
linesize <- 1
axistext <- 14
multi_colors <- c('darkviolet', '#009e73', '#56b4e9','#e69f00', '#f0e442', 'red')


multi_shapes <- c(15, 16, 18, 8, 5, 1)
x_string <- '\U03B4'

legend_str <- paste('Threshold Parameter', '\U03C4')

make_panel <- function(dataframe, dataset) {
  
  
  
    
  data <-  dataframe %>%
      filter(.data$data_name == .env$dataset) 
  
  u <- ggplot(data, aes(color = method, x  = as.factor(delta_val), y = utility_ratio, shape = method)) +
  geom_point(size = pt_size)+
  geom_line(size = linesize)+
  theme_gnuplot()+
  xlab(x_string)+
  ylab("Utility Ratio")+
  theme_gnuplot()+
  theme(legend.position = "top",
  legend.direction = "horizontal",
  axis.title.y = element_text(size = axistext),
  axis.title.x = element_text(size = axistext))+
  ggtitle("Utility Ratio")+
  scale_shape_manual(values=multi_shapes)+
  scale_color_manual(values=multi_colors)
  
  f <- ggplot(data, aes(color = method, x  = as.factor(delta_val), y = fairness_ratio, shape = method)) +
    geom_point(size = pt_size)+
    geom_line(size = linesize)+
    theme_gnuplot()+
    xlab(x_string)+
    ylab("Fairness Ratio")+
    theme_gnuplot()+
    theme(legend.position = "top",
          legend.direction = "horizontal",
          axis.title.y = element_text(size = axistext),
          axis.title.x = element_text(size = axistext))+
    ggtitle("Fairness Ratio")+
    scale_shape_manual(values=multi_shapes)+
    scale_color_manual(values=multi_colors)
  
  
  se <- ggplot(data) +
    geom_col(aes(x = as.factor(delta_val),y = position_seen_prop, fill = method), position = "dodge")+
    theme_gnuplot()+
    scale_fill_gnuplot()+
    labs(x = x_string, y = 'Seen Ratio')+
    theme(legend.position = "top",
          legend.direction = "horizontal",
          axis.title.y = element_text(size = axistext),
          axis.title.x = element_text(size = axistext))+
    ggtitle("Seen Ratio")+
    labs(fill = legend_str)
  
  r <- ggplot(data) +
    geom_col(aes(x = as.factor(delta_val),y = ra_count, fill = method), position = "dodge")+
    theme_gnuplot()+
    scale_fill_gnuplot()+
    labs(x = x_string, y = 'Random Access')+
    theme(legend.position = "top",
          legend.direction = "horizontal",
          axis.title.y = element_text(size = axistext),
          axis.title.x = element_text(size = axistext))+
    ggtitle("Random Access")+
    labs(fill = legend_str)
  
  s <- ggplot(data) +
    geom_col(aes(x = as.factor(delta_val),y = sa_count, fill = method), position = "dodge")+
    theme_gnuplot()+
    scale_fill_gnuplot()+
    labs(x = x_string, y = 'Sorted Access')+
    theme(legend.position = "top",
          legend.direction = "horizontal",
          axis.title.y = element_text(size = axistext),
          axis.title.x = element_text(size = axistext))+
    ggtitle("Sorted Access")+
    labs(fill = legend_str)
  
  t <- ggplot(data) +
    geom_col(aes(x = as.factor(delta_val),y = wall_time, fill = method), position = "dodge")+
    theme_gnuplot()+
    scale_fill_gnuplot()+
    labs(x = x_string, y = 'Time (s)')+
    theme(legend.position = "top",
          legend.direction = "horizontal",
          axis.title.y = element_text(size = axistext),
          axis.title.x = element_text(size = axistext))+
    ggtitle("Time (s)")+
    labs(fill = legend_str)
  
  panel <- ggarrange(se,u,f,s, r, t,
                           ncol = 3, nrow = 2, common.legend = TRUE, legend = "top")
  dataframe_str <- substitute(dataframe)
  dataset_str <-glue::glue({dataset})
  
  
  ggsave(panel, filename = paste0('delta/',dataframe_str, '_', dataset_str, '.pdf'), device = cairo_pdf,
         width = pdfwidth, height = pdfheight, units = "in")
  
}

make_panel(proportional_task, 'bank')
make_panel(proportional_task, 'credit')
make_panel(proportional_task, 'gauss')
make_panel(proportional_task, 'lc')
make_panel(proportional_task, 'hc')

make_panel(equal_task, 'bean')
make_panel(equal_task, 'iit')
make_panel(equal_task, 'gauss')
make_panel(equal_task, 'lc')
make_panel(equal_task, 'hc')
