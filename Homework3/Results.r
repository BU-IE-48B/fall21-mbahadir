acc_res_ordered

acc_res_ordered[1]

acc_res_ordered[1][,c("approach", "k", "avg_acc")]

final_res







acc_res_ordered

acc_res_ordered[1]

acc_res_ordered[1][,c("approach", "k", "avg_acc")]

final_res







acc_res_ordered

acc_res_ordered[1]

acc_res_ordered[1][,c("approach", "k", "avg_acc")]

final_res







acc_res_ordered

acc_res_ordered[1]

acc_res_ordered[1][,c("approach", "k", "avg_acc")]

final_res







acc_res_ordered

acc_res_ordered[1]

acc_res_ordered[1][,c("approach", "k", "avg_acc")]

final_res







library(data.table)

all_res_list=vector("list", 5)

res1=data.table(Dataset="Beef", Representation="1 Difference", Distance="Euclidean", K="1", "Cross Validation Mean Accuracy"= 0.56, "Test Accuracy"=0.7)
all_res_list[[1]]=res1

res2=data.table(Dataset="Bird Chicken", Representation="1 Difference", Distance="Dynamic Time Warping with epsilon 0.05", K="1", "Cross Validation Mean Accuracy"= 0.91, "Test Accuracy"=0.75)
all_res_list[[2]]=res2

res3=data.table(Dataset="BME", Representation="2 Difference", Distance="Dynamic Time Warping with epsilon 0.05", K="1", "Cross Validation Mean Accuracy"= 1.00, "Test Accuracy"=0.98)
all_res_list[[3]]=res3

res4=data.table(Dataset="Coffee", Representation="Raw Dataset", Distance="Dynamic Time Warping with epsilon 0.05", K="3", "Cross Validation Mean Accuracy"= 1.00, "Test Accuracy"=0.9285714)
all_res_list[[4]]=res4

res5=data.table(Dataset="Wine", Representation="Piecewise Aggregate Approximation with segment lenght 9", Distance="Euclidean", K="1", "Cross Validation Mean Accuracy"= 1.00, "Test Accuracy"=0.6111111)
all_res_list[[5]]=res5

all_result=rbindlist(all_res_list)
all_result


