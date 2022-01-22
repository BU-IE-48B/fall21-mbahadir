library(data.table)
library(lubridate)
library(forecast)
library(ggplot2)
library(rpart)
library(rattle)

data=fread("production_with_weather_data/production_with_weather_data.csv")

str(data)

data[date=="2021-12-18",]

summary(data)

perf_dt=function(type,actual,forecast){
    name=type
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
    l=data.frame(name,n,mean,sd,FBias,MAPE,RMSE,MAD,WMAPE)
    return(l)
}

data[is.na(DSWRF_surface_38_35),]

data[is.na(RH_2.m.above.ground_38_35),]

data[is.na(TCDC_entire.atmosphere_38_35),]

data[is.na(TCDC_high.cloud.layer_38_35),]

data[is.na(TCDC_low.cloud.layer_38_35),]

data[is.na(TCDC_middle.cloud.layer_38_35),]

data[is.na(TMP_2.m.above.ground_38_35),]

data=as.data.frame(data)

data[is.na(data)] = 0

data=as.data.table(data)

corr_info=cor(data[,-c("date", "hour","production")])

corr_info[c(seq(1:25)),
          c(seq(1:25))]

pca_DSWRF=princomp(data[,c(seq(1:25)+3),with=F])
summary(pca_DSWRF,loadings=T)

data[, DSWRF:=pca_DSWRF$scores[,1]]

corr_info[c(seq(1:25)+25),
          c(seq(1:25)+25)]

pca_RH=princomp(data[,c(seq(1:25)+28),with=F])
summary(pca_RH,loadings=T)

data[, RH:=pca_RH$scores[,1]]

corr_info[c(seq(1:100)+50),
          c(seq(1:100)+50)]

pca_TCDC=princomp(data[,seq(1:100)+53,with=F])
summary(pca_TCDC,loadings=T)

data[, TCDC1:=pca_TCDC$scores[,1]]
data[, TCDC2:=pca_TCDC$scores[,2]]

corr_info[c(seq(1:25)+150),
          c(seq(1:25)+150)]

pca_Temp=princomp(data[,c(seq(1:25)+153),with=F])
summary(pca_Temp,loadings=T)

data[, Temp:=pca_Temp$scores[,1]]











data[,datetime:=ymd(date)+dhours(hour)]
data=data[order(datetime)] 

ggplot(data ,aes(x=datetime,y=production)) + geom_line()+
        labs(title="Production Data in betweem between 2019-09-01 and 2021-12-25")+ylab("Production")+xlab("Dates")

data[,trend:=1:.N]

pacf(data$production, 48)

data[,mon:=as.character(month(date,label=T))]
data[,w_day:=as.character(wday(date,label=T))]
data[,lag48:=shift(production,48)]

train=data[date<"2021-11-01"]
test=data[date>="2021-11-01"]

first_model=lm(production~trend+w_day+mon+hour,train)
summary(first_model)

second_model=lm(production~trend+mon+Temp+DSWRF+RH+TCDC1+TCDC2+hour,train)
summary(second_model)

lm2_model_dt=data.table(date=train$datetime, actual=train$production, predicted=predict(second_model,train))

ggplot(lm2_model_dt ,aes(x=date)) +
        geom_line(aes(y=actual,color='real')) + 
        geom_line(aes(y=predicted,color='trend'))+
        labs(title="Production Prediction with second Model")+ylab("Production")+xlab("Dates")

checkresiduals(second_model$residuals)

third_model=lm(production~trend+mon+Temp+DSWRF+RH+TCDC1+TCDC2+hour+lag48,train)
summary(third_model)

lm3_model_dt=data.table(date=train$datetime, actual=train$production, predicted=predict(third_model,train))

ggplot(lm3_model_dt ,aes(x=date)) +
        geom_line(aes(y=actual,color='real')) + 
        geom_line(aes(y=predicted,color='trend'))+
        labs(title="Production Prediction with second Model")+ylab("Production")+xlab("Dates")

checkresiduals(third_model$residuals)

train[, predicted:=predict(third_model,train)]
train[, residual:=production-predicted]

fit_res_tree1=rpart(residual~trend+mon+Temp+DSWRF+RH+TCDC1+TCDC2+hour,train,
                   control=rpart.control(cp=0,maxdepth=4))

fancyRpartPlot(fit_res_tree1)

train[,hour15_less:=as.numeric(hour<=15)]
train[,DSWRF_n1041_above:=as.numeric(DSWRF>-1041)]

third_model_v2=lm(production~trend+mon+Temp+DSWRF+RH+TCDC1+TCDC2+hour+lag48+hour15_less:DSWRF_n1041_above,train)
summary(third_model_v2)

lm3_model_dt_v2=data.table(date=train$datetime, actual=train$production, predicted=predict(third_model_v2,train))

ggplot(lm3_model_dt_v2 ,aes(x=date)) +
        geom_line(aes(y=actual,color='real')) + 
        geom_line(aes(y=predicted,color='trend'))+
        labs(title="Production Prediction with second Model")+ylab("Production")+xlab("Dates")

checkresiduals(third_model_v2$residuals)

train[, predicted2:=predict(third_model_v2,train)]
train[, residual:=production-predicted2]

fit_res_tree1=rpart(residual~trend+mon+Temp+DSWRF+RH+TCDC1+TCDC2+hour,train,
                   control=rpart.control(cp=0,maxdepth=4))

fancyRpartPlot(fit_res_tree1)

train[,hour22_above:=as.numeric(hour>22)]
train[,TCDC1_108_above:=as.numeric(TCDC1>108)]

third_model_v3=lm(production~trend+mon+Temp+DSWRF+RH+TCDC1+TCDC2+hour+lag48+hour15_less:DSWRF_n1041_above+ hour22_above:TCDC1_108_above,train)
summary(third_model_v3)

lm3_model_dt_v3=data.table(date=train$datetime, actual=train$production, predicted=predict(third_model_v3,train))

ggplot(lm3_model_dt_v3 ,aes(x=date)) +
        geom_line(aes(y=actual,color='real')) + 
        geom_line(aes(y=predicted,color='trend'))+
        labs(title="Production Prediction with second Model")+ylab("Production")+xlab("Dates")

checkresiduals(third_model_v3$residuals)

data[,hour15_less:=as.numeric(hour<=15)]
data[,DSWRF_n1041_above:=as.numeric(DSWRF>-1041)]

data[,hour22_above:=as.numeric(hour>22)]
data[,TCDC1_108_above:=as.numeric(TCDC1>108)]

train=data[date<"2021-11-01"]
test=data[date>="2021-11-01"]

test_start=as.Date("2021-11-01")

results_feature_reg=vector("list",nrow(test)-48)
count=1
reg_time_feat=Sys.time()
for(i in 1:(nrow(test[,.N,by=date])+1)){
    current_date=test_start+i-2
    train_data=data[date<current_date]
    test_data=data[date==current_date+1]
    control=0   
    # Predictions
    lm_model=lm(production~trend+mon+Temp+DSWRF+RH+TCDC1+TCDC2+hour+lag48+
                      hour15_less:DSWRF_n1041_above+ hour22_above:TCDC1_108_above, data=train_data)
    current_res=predict(lm_model,newdata=test_data)
    if(is.na(current_res[1])){
        next
        
    }
    for(j in 1:24){
        current_res=pmax(current_res,0) # Remove negative values        
        results_feature_reg[count]=current_res[j]
        count=count+1     
        
    }
}
reg_time_feat=Sys.time()-reg_time_feat

res_dt_regression_feature=t(rbind(results_feature_reg))
res_dt_regression_feature=as.data.table(res_dt_regression_feature)

setnames(res_dt_regression_feature,"results_feature_reg","Forescasted")

final_res_feat_regression=test[,c(1,3)]

final_res_feat_regression$forecast=as.numeric(res_dt_regression_feature$Forescasted)

final_res_feat_regression

perf_dt("Performance of Final Linear Regression with Slicing Window", final_res_feat_regression$production, as.numeric(final_res_feat_regression$forecast))



data[, lag48:=shift(production,48)]

perf_dt("Performance of Base Model(Lag 48)", final_res_feat_regression$production, data[date>="2021-11-01", lag48])


