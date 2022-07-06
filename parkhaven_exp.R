library(xts)
library(zoo)
library(plotly)
library(readxl)
library(plyr)
library(lubridate)

parkhaven_opp = EGV_parser('Data_EC/OW000-008 oppervlak Parkhaven export-trend-20150101000000-20201210171559.csv')
parkhaven_bodem = EGV_parser('Data_EC/OW000-008 bodem Parkhaven export-trend-20150101000000-20201210171559.csv')

parkhaven_opp_TS = maak_TS(parkhaven_opp) 
parkhaven_bodem_TS = maak_TS(parkhaven_bodem)

#maak_grafiek(parkhaven_opp)
#maak_grafiek(parkhaven_bodem)

debiet_parksluis_gemaal = as.data.frame(read_xlsx('Data_Schutten en debieten/waterstanden_debieten_HHD.xlsx',
                                    sheet = 'Debiet_parksluizen',skip = 4,
                                    col_types = c('date','numeric'),
                                    col_names = c('datetime','debiet')))
waterstanden_parksluizen = as.data.frame(read_xlsx('Data_Schutten en debieten/waterstanden_debieten_HHD.xlsx',
                                     sheet = 'Waterstand_parksluizen',skip = 4,
                                     col_names = c('datetime','HWZ','LWZ'),
                                     col_types = c('date','numeric','numeric')))

schuttingsdata_parksluizen = get_schutting('Data_Schutten en debieten/Passages parksluis2018 geanonimiseerd.xlsx')



kaders = data.frame('aanvang' = schuttingsdata_parksluizen$datetime)
stap = 360 #MIN
kaders$lb = kaders$aanvang-60*stap
kaders$ub = kaders$aanvang+60*stap
kader = kaders[1,]
ncols = dim(parkhaven_opp_TS[paste(kader[[2]],kader[[3]],sep = '/')] )[1]
buffer=20
empty_db = data.frame(matrix(NA,nrow = dim(kaders)[1],ncol = ncols+buffer))

test_set = parkhaven_opp_TS

for (row in 1:nrow(kaders)) {
  if (row %% 200 == 0) {
    print(row)
  }
  kader = kaders[row,]
  new_interval = interval(kader[[2]],kader[[3]])
  #kader_waardes = parkhaven_opp[parkhaven_opp$datetime %within% new_interval,5]
  kader_waardes = test_set[paste(kader[[2]],kader[[3]],sep = '/')]
  if (length(kader_waardes) < (ncols+buffer)) {
    kader_waardes = c(as.vector(kader_waardes),rep(NA,ncols+buffer-length(kader_waardes)))
  }
  #empty_db = rbind.fill(empty_db,as.data.frame(t(kader_waardes)))
  empty_db[row,] = as.vector(kader_waardes)
}



test = as.data.frame(t(empty_db))
test = test[1:72,]

#test = test[,which(apply(test,2,min,na.rm=T)>5)]


#parkhaven_opp[which((parkhaven_opp$datetime > kader[[2]] & parkhaven_opp$datetime < kader[[3]])),5]

myfunc = function(x) {
  return(x/mean(x[35],x[36]))
}

my_plotr = function(perc,a) {
  for (i in 1:a) {
    lines(seq(-36,35)*10,perc[,sample(x = ncol(perc),size = 1)])
  }
}
perc = apply(test,2,myfunc)
#plot(seq(-35,36)*10,perc[,sample(x = ncol(perc),size = 1)],ylim = c(0.5,2),type = 'lines')
#abline(v=0,col = 'blue')
#my_plotr(perc,10)


mn = apply(perc,2,mean,na.rm=T)[1:72]
sd0 = apply(perc,2,sd,na.rm=T)[1:72]
sd1 = mn+sd0
sd2 = mn-sd0
plot(seq(-35,36)*10,mn,type = 'lines',col = 'red',ylab = 'EGV % t.o.v. aanvang schutten',xlab = 'tijd voor/na schutten (min)',
     main = 'EGV in parkhaven voor en na schutten t.o.v. EGV op schutten',sub = 'Gemiddeld in rood, in zwart een standaard deviatie band') +
abline(v=0,col = 'blue') +
lines(seq(-35,36)*10,sd1) +
lines(seq(-35,36)*10,sd2)
