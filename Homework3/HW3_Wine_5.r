library(data.table)
library(TSrepr)
library(TSdist)
library(dtw)
library(TunePareto)
library(dplyr)

dataset_path="D:/Datasets/Univariate2018_arff/Univariate_arff/"

distance_path="C:/Users/bahad/GitHub/IE48B/Homework3/Distances/"

first_dataset="Beef"
second_dataset="BirdChicken"
third_dataset="BMW"
fourth_dataset="Coffee"
fifth_dataset="Wine"

traindata=as.matrix(fread(sprintf('%s%s/%s_TRAIN.txt',dataset_path, fifth_dataset,fifth_dataset)))

head(traindata)

str(traindata)

trainclass=traindata[,1] 

traindata=traindata[,2:ncol(traindata)]

tlength=ncol(traindata)
n_series_train=nrow(traindata)

set.seed(35)
nof_rep=5
n_fold=10

cv_indices=generateCVRuns(trainclass, ntimes =nof_rep, nfold = n_fold, 
                          leaveOneOut = FALSE, stratified = TRUE)

str(cv_indices)

nn_classify_cv=function(dist_matrix,train_class,test_indices,k){
    
    test_distances_to_train=dist_matrix[test_indices,]
    test_distances_to_train=test_distances_to_train[,-test_indices]
    train_class=train_class[-test_indices]

    ordered_indices=apply(test_distances_to_train,1,order)
    if(k==1){
        nearest_class=as.numeric(train_class[as.numeric(ordered_indices[1,])])
        nearest_class=data.table(id=test_indices,nearest_class)
    } else {
        nearest_class=apply(ordered_indices[1:k,],2,function(x) {train_class[x]})
        nearest_class=data.table(id=test_indices,t(nearest_class))
    }
    
    long_nn_class=melt(nearest_class,'id')

    class_counts=long_nn_class[,.N,list(id,value)]
    class_counts[,predicted_prob:=N/k]
    wide_class_prob_predictions=dcast(class_counts,id~value,value.var='predicted_prob')
    wide_class_prob_predictions[is.na(wide_class_prob_predictions)]=0
    class_predictions=class_counts[,list(predicted=value[which.max(N)]),by=list(id)]
    
    
    return(list(prediction=class_predictions,prob_estimates=wide_class_prob_predictions))
    
}



dt_ts_train=data.table(traindata)
dt_ts_train[,id:=1:.N]
long_train=melt(dt_ts_train,id.vars=c('id'))
long_train[,time:=as.numeric(gsub("\\D", "", variable))-1]
long_train=long_train[order(id,time)]
diff_long=copy(long_train)
diff_long[,diff_series:=value-shift(value,1),by=list(id)]
head(diff_long)

diff_train=dcast(diff_long[!is.na(diff_series)],id~time,value.var='diff_series')
diff_train=diff_train[,-c("id")]
head(diff_train)
diff_train=as.matrix(diff_train)

difference_obtainer=function(traindata, diff_value){
    dt_ts_train=data.table(traindata)
    dt_ts_train[,id:=1:.N]
    long_train=melt(dt_ts_train,id.vars=c('id'))
    long_train[,time:=as.numeric(gsub("\\D", "", variable))-1]
    long_train=long_train[order(id,time)]
    diff_long=copy(long_train)
    diff_long[,diff_series:=value-shift(value,diff_value),by=list(id)]#Lag value is assigned by diff_value
    head(diff_long)
    
    diff_train=dcast(diff_long[!is.na(diff_series)],id~time,value.var='diff_series')
    diff_train=diff_train[,-c("id")]
    head(diff_train)
    diff_train=as.matrix(diff_train)
    
    return(diff_train)
}

diff_train=difference_obtainer(traindata,1)

diff_train_2=difference_obtainer(traindata,2)





segment_length=5

paa_results=vector("list", max(long_train$id))

for(i in 1:max(long_train$id)){
    current_ts=long_train[id==i,]$value
    
    paa_rep=repr_paa(current_ts, segment_length, meanC)
    current_dt=data.table(time=1:length(long_train[id==i,]$value))
    result_dt=data.table(time=c(1:(length(paa_rep)))*segment_length, values=paa_rep)
    all_dt=merge(current_dt, result_dt, by='time',all.x=T)
    all_dt[,values:=nafill(values,'nocb')]
    paa_results[[i]]=transpose(data.table(values=all_dt$values))
    
}

paa_train=rbindlist(paa_results)

paa_train

paa_obtainer=function(traindata,segment_length){
    dt_ts_train=data.table(traindata)
    dt_ts_train[,id:=1:.N]
    long_train=melt(dt_ts_train,id.vars=c('id'))
    long_train[,time:=as.numeric(gsub("\\D", "", variable))-1]
    long_train=long_train[order(id,time)]
    
    paa_results=vector("list", max(long_train$id))
    for(i in 1:max(long_train$id)){
        current_ts=long_train[id==i,]$value

        paa_rep=repr_paa(current_ts, segment_length, meanC)
        current_dt=data.table(time=1:length(long_train[id==i,]$value))
        result_dt=data.table(time=c(1:(length(paa_rep)))*segment_length, values=paa_rep)
        all_dt=merge(current_dt, result_dt, by='time',all.x=T)
        all_dt[,values:=nafill(values,'nocb')]
        paa_results[[i]]=transpose(data.table(values=all_dt$values))

    }
    return(rbindlist(paa_results))
}

paa_train=paa_obtainer(traindata,9)

paa_train_2=paa_obtainer(traindata,18)





large_number=10000

dist_euc=as.matrix(dist(traindata))
diag(dist_euc)=large_number
fwrite(dist_euc,sprintf('%s%s/%s_euc_raw_dist.csv', distance_path, fifth_dataset, fifth_dataset),col.names=F)

dist_dtw=as.matrix(dtwDist(traindata))
diag(dist_dtw)=large_number
fwrite(dist_dtw,sprintf('%s%s/%s_dtw_raw_dist.csv', distance_path, fifth_dataset, fifth_dataset),col.names=F)

dist_lcss=TSDatabaseDistances(traindata,distance='lcss',epsilon=0.05)
dist_lcss=as.matrix(dist_lcss)
diag(dist_lcss)=large_number
fwrite(dist_lcss,sprintf('%s%s/%s_lcss_raw_epsilon_005.csv', distance_path, fifth_dataset, fifth_dataset),col.names=F)  

dist_erp=TSDatabaseDistances(traindata,distance='erp',g=0.5)
dist_erp=as.matrix(dist_erp)
diag(dist_erp)=large_number
fwrite(dist_erp,sprintf('%s%s/%s_erp_raw_gap_005.csv', distance_path, fifth_dataset, fifth_dataset),col.names=F)  

dist_euc_diff=as.matrix(dist(diff_train))
diag(dist_euc_diff)=large_number
fwrite(dist_euc_diff,sprintf('%s%s/%s_euc_diff_dist.csv', distance_path, fifth_dataset, fifth_dataset),col.names=F)

dist_dtw_diff=as.matrix(dtwDist(diff_train))
diag(dist_dtw_diff)=large_number
fwrite(dist_dtw_diff,sprintf('%s%s/%s_dtw_diff_dist.csv', distance_path, fifth_dataset, fifth_dataset),col.names=F)

dist_lcss_diff=TSDatabaseDistances(diff_train,distance='lcss',epsilon=0.05)
dist_lcss_diff=as.matrix(dist_lcss_diff)
diag(dist_lcss_diff)=large_number
fwrite(dist_lcss_diff,sprintf('%s%s/%s_lcss_diff_epsilon_005.csv', distance_path, fifth_dataset, fifth_dataset),col.names=F)  

dist_erp_diff=TSDatabaseDistances(diff_train,distance='erp',g=0.5)
dist_erp_diff=as.matrix(dist_erp_diff)
diag(dist_erp_diff)=large_number
fwrite(dist_erp_diff,sprintf('%s%s/%s_erp_diff_gap_005.csv', distance_path, fifth_dataset, fifth_dataset),col.names=F)  

dist_euc_diff_2=as.matrix(dist(diff_train_2))
diag(dist_euc_diff_2)=large_number
fwrite(dist_euc_diff_2,sprintf('%s%s/%s_euc_diff2_dist.csv', distance_path, fifth_dataset, fifth_dataset),col.names=F)

dist_dtw_diff_2=as.matrix(dtwDist(diff_train_2))
diag(dist_dtw_diff_2)=large_number
fwrite(dist_dtw_diff_2,sprintf('%s%s/%s_dtw_diff2_dist.csv', distance_path, fifth_dataset, fifth_dataset),col.names=F)

dist_lcss_diff_2=TSDatabaseDistances(diff_train_2,distance='lcss',epsilon=0.05)
dist_lcss_diff_2=as.matrix(dist_lcss_diff_2)
diag(dist_lcss_diff_2)=large_number
fwrite(dist_lcss_diff_2,sprintf('%s%s/%s_lcss_diff2_epsilon_005.csv', distance_path, fifth_dataset, fifth_dataset),col.names=F)  

dist_erp_diff_2=TSDatabaseDistances(diff_train_2,distance='erp',g=0.5)
dist_erp_diff_2=as.matrix(dist_erp_diff_2)
diag(dist_erp_diff_2)=large_number
fwrite(dist_erp_diff_2,sprintf('%s%s/%s_erp_diff2_gap_005.csv', distance_path, fifth_dataset, fifth_dataset),col.names=F)  

dist_euc_paa=as.matrix(dist(paa_train))
diag(dist_euc_paa)=large_number
fwrite(dist_euc_paa,sprintf('%s%s/%s_euc_paa_dist.csv', distance_path, fifth_dataset, fifth_dataset),col.names=F)

dist_dtw_paa=as.matrix(dtwDist(paa_train))
diag(dist_dtw_paa)=large_number
fwrite(dist_dtw_paa,sprintf('%s%s/%s_dtw_paa_dist.csv', distance_path, fifth_dataset, fifth_dataset),col.names=F)

dist_lcss_paa=TSDatabaseDistances(paa_train,distance='lcss',epsilon=0.05)
dist_lcss_paa=as.matrix(dist_lcss_paa)
diag(dist_lcss_paa)=large_number
fwrite(dist_lcss_paa,sprintf('%s%s/%s_lcss_paa_epsilon_005.csv', distance_path, fifth_dataset, fifth_dataset),col.names=F)  

dist_erp_paa=TSDatabaseDistances(paa_train,distance='erp',g=0.5)
dist_erp_paa=as.matrix(dist_erp_paa)
diag(dist_erp_paa)=large_number
fwrite(dist_erp_paa,sprintf('%s%s/%s_erp_paa_gap_005.csv', distance_path, fifth_dataset, fifth_dataset),col.names=F)  

dist_euc_paa_2=as.matrix(dist(paa_train_2))
diag(dist_euc_paa_2)=large_number
fwrite(dist_euc_paa_2,sprintf('%s%s/%s_euc_paa2_dist.csv', distance_path, fifth_dataset, fifth_dataset),col.names=F)

dist_dtw_paa_2=as.matrix(dtwDist(paa_train_2))
diag(dist_dtw_paa_2)=large_number
fwrite(dist_dtw_paa_2,sprintf('%s%s/%s_dtw_paa2_dist.csv', distance_path, fifth_dataset, fifth_dataset),col.names=F)

dist_lcss_paa_2=TSDatabaseDistances(paa_train_2,distance='lcss',epsilon=0.05)
dist_lcss_paa_2=as.matrix(dist_lcss_paa_2)
diag(dist_lcss_paa_2)=large_number
fwrite(dist_lcss_paa_2,sprintf('%s%s/%s_lcss_paa2_epsilon_005.csv', distance_path, fifth_dataset, fifth_dataset),col.names=F)  

dist_erp_paa_2=TSDatabaseDistances(paa_train_2,distance='erp',g=0.5)
dist_erp_paa_2=as.matrix(dist_erp_paa_2)
diag(dist_erp_paa_2)=large_number
fwrite(dist_erp_paa_2,sprintf('%s%s/%s_erp_paa2_gap_005.csv', distance_path, fifth_dataset, fifth_dataset),col.names=F)  

dist_folder=sprintf('%s%s', distance_path, fifth_dataset)
dist_files=list.files(dist_folder, full.names=T)

dist_files





k_levels=c(1,3,5)
approach_file=list.files(dist_folder)

result=vector('list',length(dist_files)*nof_rep*n_fold*length(k_levels))

iter=1
for(m in 1:length(dist_files)){ #
    print(dist_files[m])
    dist_mat=as.matrix(fread(dist_files[m],header=FALSE))
    for(i in 1:nof_rep){
        this_fold=cv_indices[[i]]
        for(j in 1:n_fold){
            test_indices=this_fold[[j]]
            for(k in 1:length(k_levels)){
                current_k=k_levels[k]
                current_fold=nn_classify_cv(dist_mat,trainclass,test_indices,k=current_k)
                accuracy=sum(trainclass[test_indices]==current_fold$prediction$predicted)/length(test_indices)
                tmp=data.table(approach=approach_file[m],repid=i,foldid=j,
                               k=current_k,acc=accuracy)
                result[[iter]]=tmp
                iter=iter+1
            }
            
        }
    
    }   
    
}

dataframe_result=rbindlist(result)
head(dataframe_result)

acc_res=dataframe_result[,list(avg_acc=mean(acc),sdev_acc=sd(acc), repid=max(repid), foldid=max(foldid), 
                                   result_count=.N),by=list(approach,k)]
acc_res_ordered=acc_res[order(avg_acc,decreasing = TRUE)]

acc_res_ordered

# require(ggplot2)
# ggplot(dataframe_result,aes(x=paste0(approach,'with K=',k), y=acc)) +
#         geom_boxplot()+
#         labs(title="Boxplot of Models")+
#         xlab("Model Types")+
#         coord_flip()

acc_res_ordered[1]

traindata=as.matrix(fread(sprintf('%s%s/%s_TRAIN.txt',dataset_path, fifth_dataset,fifth_dataset)))
testdata=as.matrix(fread(sprintf('%s%s/%s_TEST.txt',dataset_path, fifth_dataset,fifth_dataset)))

all_dt=rbind(traindata, testdata)

allclass=all_dt[,1] 
all_dt=all_dt[,2:ncol(all_dt)]

test_indices_last=(nrow(all_dt)+1-nrow(testdata)):nrow(all_dt)

test_indices_last

last_k=1

paa_test=paa_obtainer(all_dt,9)

dist_euc_paa_test=as.matrix(dist(paa_test))
diag(dist_euc_paa_test)=large_number

last_result=nn_classify_cv(dist_euc_paa_test,allclass,test_indices_last,k=last_k)
accuracy=sum(allclass[test_indices_last]==last_result$prediction$predicted)/length(test_indices_last)
final_res=data.table(approach="Wine_euc_paa_dist_Test.csv", k=last_k, acc=accuracy)

final_res

acc_res_ordered[1][,c("approach", "k", "avg_acc")]
