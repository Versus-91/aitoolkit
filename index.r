
install.packages("ggplot2",repos = "http://cran.us.r-project.org")
library(ggplot2) 
library(GGally) 
data(mtcars) 
Scatter_Matrix <- ggpairs(mtcars,columns = c(1, 3:6), 
                          title = "Scatter Plot Matrix for mtcars Dataset", 
                          axisLabels = "show") 
ggsave("Scatter plot matrix.png", Scatter_Matrix, width = 7, 
       height = 7, units = "in") 
Scatter_Matrix