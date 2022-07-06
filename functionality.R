EGV_parser = function(file) {
  EGV_bestand = read.csv(file,row.names = NULL,sep="\t", fileEncoding="UTF-16LE")
  head(EGV_bestand)
  EGV_bestand$Waarde[which(is.na(EGV_bestand$Waarde))] = 0
  EGV_gelijmd_txt = paste(EGV_bestand$Tijd..Europe.Amsterdam.,EGV_bestand$Waarde,sep = ',')
  EGV_gelijmd = as.numeric(sub(",", ".", EGV_gelijmd_txt))
  EGV_bestand$Tijd..Europe.Amsterdam. = EGV_gelijmd
  colnames(EGV_bestand) = colnames(EGV_bestand)[2:dim(EGV_bestand)[2]]
  EGV_bestand = data.frame(EGV_bestand)[1:(dim(EGV_bestand)[2]-2)]
  head(EGV_bestand)
  EGV_bestand$datetime = as.POSIXlt(paste0(EGV_bestand$Datum,' ',EGV_bestand$Tijd..Europe.Amsterdam.))
  EGV_data = EGV_bestand
  return(EGV_data)
}

maak_TS = function(EGV_data) {
  require(xts)
  return(xts(EGV_data$Waarde,order.by =EGV_data$datetime))
}

maak_grafiek = function(EGV_data) {
  bron = as.character(substitute(EGV_data))
  print(bron)
  p = ggplot(data = EGV_data, aes(x = as.POSIXct(datetime), y = Waarde)) +
    geom_line() + scale_x_datetime(date_labels = "%Y-%M") + labs(x = 'Datum/tijd', y='EGV (mS/cm)',title = bron)
  plt = ggplotly(p,dynamicTicks = T)
  htmlwidgets::saveWidget(as_widget(plt), paste0(bron,'.html'))
}


get_schutting = function(file) {
  require(readxl)
  schuttingsdata_parksluizen = as.data.frame(read_xlsx(file),make.names = T)
  schuttingsdata_parksluizen$Tijd = strftime(schuttingsdata_parksluizen$Tijd,format = '%H:%M',tz = 'UTC')
  jaar = substr(schuttingsdata_parksluizen$Datum,0,4)
  maand = substr(schuttingsdata_parksluizen$Datum,5,6)
  dag = substr(schuttingsdata_parksluizen$Datum,7,8)
  datum = paste(jaar,maand,dag,sep = '-')
  datetime = as.POSIXct(paste(datum, schuttingsdata_parksluizen$Tijd))
  schuttingsdata_parksluizen$datetime = datetime
  
  schuttingsdata_parksluizen$`Tijd Aanvang` = datetime
  schuttingsdata_parksluizen$`Tijd Einde` = datetime + schuttingsdata_parksluizen$Duur*60
  return(schuttingsdata_parksluizen)
}

