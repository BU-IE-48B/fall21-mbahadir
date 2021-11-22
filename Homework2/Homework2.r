library(data.table)

set.seed(35)

train=fread("CBF/CBF_TRAIN.txt")
test=fread("CBF/CBF_TEST.txt")

str(train)



library("genlasso", quietly = TRUE)

value_obtainer=function(dataset, index){
    return (as.numeric(dataset[index,2:129]))
}

fused_results=vector("list", 30)

fused_lasso=function(dataset, index, control=0){
    result = trendfilter(value_obtainer(train, index), ord=0) 
    cv = cv.trendfilter(result)   
    plot(cv)#Performance of the lambda values
    print(paste0(index,". Timeseries") )
    print(paste0("Lambda Value of ",index,". dataset for min value: ", cv$lambda.min))
    print(paste0("Lambda Value of ",index,". dataset for 1se value: ", cv$lambda.1se))
    sqrt(seq(1:500000)**2)# In order to get aligned plots with results.(Time consumption)
    
    plot(result, lambda=cv$lambda.min, main=paste0("Plot of ", index,". Timeseries with lambda min parameter"))  

    if(control==1){
        sqrt(seq(1:500000)**2)
        return(result)
    } 
    return(list(result, cv))
}

fused_results[[1]]=fused_lasso(train, 1, 0)

for(i in seq(1:30)){
    fused_results[[i]]=fused_lasso(train, i)
}

library(caret)
library(rpart)

train_perf=train[,c(-1)]

value_obtainer_2=function(dataset, index, time_index, control){# 0 for train data, 1 for teset data
    if(control==0){
        all_time=seq(1, 128)
        all_time_final = all_time[!all_time %in% time_index]
        a=data.table(time=as.numeric(all_time_final), values=as.numeric(dataset[index,.SD,.SDcols=all_time_final]))
        return(a)#Train Dataset
    }
    else if(control==1){
        a=data.table(time=as.numeric(time_index), values=as.numeric(dataset[index,.SD,.SDcols=time_index]))
        return(a)#Test Dataset    
    }
    
    return (a)
}

perf_function=function(forecast, actual){
    n=length(actual)
    error=actual-forecast
    mean=mean(actual)
    sd=sd(actual)
    FBias=sum(error)/sum(actual)
    MPE=sum(error/actual)/n
    MAPE=sum(abs(error/actual))/n
    RMSE=sqrt(sum(error^2))/n
    MAD=sum(abs(error))/n
    WMAPE=MAD/mean
    l=data.frame(n,mean,sd,FBias,MAPE,RMSE,MAD,WMAPE)
    return(l)
}

regression_results=vector("list", 5)

for(i in seq(1, 5)){
    regression_fold_result=vector("list", 10)
    for(j in seq(1, 10)){
        current_index=seq(0, 110, 10)+j#Test indexes by considering k information. it include k'th indexes
        if(j<=8){# In order to control above 120, a condition was created
            last_index=120+j
            current_index[13]=last_index
        }
        
        train_loop=value_obtainer_2(train_perf, 1, current_index, 0)# Train dataset by k'th index
        test_loop=value_obtainer_2(train_perf, 1, current_index, 1 )# Test dataset by k'th index
        
        model = train(values ~., data = train_loop, 
              method = "rpart",
              tuneGrid = data.frame(cp = 0),         
              control = rpart.control(minsplit = 20, minbucket = 10, maxdepth=i))# Max_depth parameter tuning   
        
        prediction_folds=predict(model,newdata=test_loop)
        regression_fold_result[[j]]=perf_function(prediction_folds, test_loop$values)#Performance Result for k'th performance.
        
}

    fold_results=t(colSums(rbindlist(regression_fold_result)))
    regression_results[[i]]=as.data.frame(fold_results)
}

reg_res=rbindlist(regression_results)
reg_res[,max_depth:=seq(1,5)]

reg_res

which.min(reg_res$RMSE)

current_dt=value_obtainer_2(train, 1, seq(1, 128), 1)

model_final = train(values ~., data = current_dt, 
          method = "rpart",
          tuneGrid = data.frame(cp = 0),
          control = rpart.control(minsplit = 20, minbucket = 10, maxdepth=which.min(reg_res$RMSE)))
    regression_results[[i]]=model$results 

predictions=predict(model_final,newdata=current_dt)

plot(predictions,type="l", xlim=c(0,130), ylim=c(-1.7,2), main="Plot of 1. Timeseries with Regressor and maxdepth=2")
points(current_dt[,.(values)])

regression_results=vector("list", 30)

regression_function=function(dataset, index){
    regression_results=vector("list", 5)

    current_dt=value_obtainer_2(dataset, index, seq(1, 128), 1)
    for(i in seq(1, 5)){
    regression_fold_result=vector("list", 10)
        for(j in seq(1, 10)){
            current_index=seq(0, 110, 10)+j#Test indexes by considering k information. it include k'th indexes
            if(j<=8){# In order to control above 120, a condition was created
                last_index=120+j
                current_index[13]=last_index
            }

            train_loop=value_obtainer_2(train_perf, index, current_index, 0)# Train dataset by k'th index
            test_loop=value_obtainer_2(train_perf, index, current_index, 1 )# Test dataset by k'th index

            model = train(values ~., data = train_loop, 
                  method = "rpart",
                  tuneGrid = data.frame(cp = 0),         
                  control = rpart.control(minsplit = 20, minbucket = 10, maxdepth=i))# Max_depth parameter tuning   
            
            prediction_folds=predict(model,newdata=test_loop)
            regression_fold_result[[j]]=perf_function(prediction_folds, test_loop$values)#Performance Result for k'th performance.

    }

    fold_results=t(colSums(rbindlist(regression_fold_result)))
    regression_results[[i]]=as.data.frame(fold_results)
}
    

    reg_res=rbindlist(regression_results)
    reg_res[,max_depth:=seq(1,5)]

    
    best_parameter=which.min(reg_res$RMSE)

    model_final = train(values ~., data = current_dt, 
              method = "rpart",
              tuneGrid = data.frame(cp = 0),
              control = rpart.control(minsplit = 20, minbucket = 10, maxdepth=best_parameter))

    predictions=predict(model_final,newdata=current_dt)
    
    plot(predictions,type="l", xlim=c(0,130), ylim=c(-1.7,2), main=paste0("Plot of ", index,". Timeseries with Regressor and maxdepth=",best_parameter))
    points(current_dt[,.(values)])
    
    return(list(reg_res,predictions))
      
}

for(i in seq(1:30)){  
    regression_results[[i]]=regression_function(train, i)
}

for(i in seq(1:30)){
    print(paste0(i,". Timeseries"))
    print(regression_results[[i]][[1]])
    cat(paste0("By considering RMSE Value, the best Max Depth Parameter: ", which.min(regression_results[[i]]$RMSE),"for ",
          i, ".Timeseries \n\n"))
}

head(train)

fused_list=vector("list", 30)

for(i in seq(1:30)){
    fused_list[[i]]=as.data.table(t(predict(fused_results[[i]][[1]], fused_results[[i]][[2]]$lambda.min)$fit))
}

fused_dt=rbindlist(fused_list)
fused_dt

regressor_list=vector("list", 30)

for(i in seq(1:30)){
    regressor_list[[i]]=as.data.table(t(regression_results[[i]][[2]]))
}

regressor_dt=rbindlist(regressor_list)
regressor_dt

library(hydroGOF)

performance_list=vector("list", 30)

mse_calculator=function(index, train, fused, regressor){
    fused_lasso_performance=sum(mse(train[1], fused[1]))/128
    regressor_performance=sum(mse(train[1], regressor[1]))/128
    perf_dt=data.frame(index, fused_lasso_performance, regressor_performance)
    return(perf_dt)
}

for(i in seq(1:30)){
    performance_list[[i]]=mse_calculator(i, train_perf[i], fused_dt[i], regressor_dt[i])   
}

performance_dt=rbindlist(performance_list)
performance_dt

box_plot_dt <- melt(performance_dt[,c(2,3)], id.vars=NULL)

box_plot_dt %>%
  ggplot( aes(x=variable, y=value, fill=variable)) +
    geom_boxplot() +
    ggtitle("Comparision Plot of 2 Methods with Box Plots") +
    xlab("Methods")+ylab("MSE Values")+
    scale_fill_discrete(labels = c("Fused Lasso", "Regression Tree"))+
    scale_x_discrete(labels= c("fused_lasso_performance"="Fused Lasso","regressor_performance"="Regression Tree"))


timeseries_class=as.matrix(train[,1])

large_number=10000

euc_dist_raw=as.matrix(dist(train_perf))

diag(euc_dist_raw)=large_number

neighborhood_raw=apply(euc_dist_raw,1,order)

predicted_raw=timeseries_class[neighborhood_raw[1,]]

table(timeseries_class,predicted_raw)

acc_raw=sum(timeseries_class==predicted_raw)/length(predicted_raw)
print(paste0("The accuracy of Raw Dataset: ", acc_raw))

euc_dist_fused=as.matrix(dist(fused_dt))

diag(euc_dist_fused)=large_number

neighborhood_fused=apply(euc_dist_fused,1,order)

predicted_fused=timeseries_class[neighborhood_fused[1,]]

table(timeseries_class,predicted_fused)

acc_lasso=sum(timeseries_class==predicted_fused)/length(predicted_fused)
print(paste0("The accuracy of Fused Lasso Model: ", acc_lasso))

euc_dist_regressor=as.matrix(dist(regressor_dt))

diag(euc_dist_regressor)=large_number

neighborhood_regressor=apply(euc_dist_regressor,1,order)

predicted_regressor=timeseries_class[neighborhood_regressor[1,]]

table(timeseries_class,predicted_regressor)

acc_fused=sum(timeseries_class==predicted_regressor)/length(predicted_regressor)
print(paste0("The accuracy of Regressor Tree Model: ", acc_fused))


