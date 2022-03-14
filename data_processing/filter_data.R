# Identify chronic user and throw away the first three months for burn-in
library(lubridate)
setwd("~/Desktop/Research/Interpretable_Opioid/Code")

SAMPLE <- read.csv("../Data/SAMPLE.csv")
length(unique(SAMPLE$patient_id)) # 31091
length(unique(SAMPLE$prescriber_id)) # 29606
length(unique(SAMPLE$pharmacy_id)) # 5479
length(unique(SAMPLE$PROPRIETARYNAME)) # 136
length(unique(SAMPLE$LABELERNAME)) # 111
# Four unique class: Benzodiazepine, Opioid, Stimulant, Testosterone
unique(SAMPLE$class)

####################################
#### Focus on opioids first
####################################

SAMPLE_OPIOID <- SAMPLE[SAMPLE$class == 'Opioid',]
# Sort the rows by patient id and date
SAMPLE_OPIOID <- SAMPLE_OPIOID[order(SAMPLE_OPIOID$patient_id, SAMPLE_OPIOID$date_filled),]
SAMPLE_OPIOID$prescription_month <- month(as.POSIXlt(SAMPLE_OPIOID$date_filled, format="%m/%d/%Y"))
SAMPLE_OPIOID$prescription_year <- year(as.POSIXlt(SAMPLE_OPIOID$date_filled, format="%m/%d/%Y"))

# Compute the prescription quarter
compute_quarter <- function(prescription_month){
  if(prescription_month < 4){
    return (1)
  }else if(prescription_month < 7){
    return (2)
  }else if(prescription_month < 10){
    return (3)
  }else {
    return (4)
  }
}
SAMPLE_OPIOID$prescription_quarter <- unlist(lapply(SAMPLE_OPIOID$prescription_month, compute_quarter))

# New user if patient starting from April does not have prescription before
chronic_user <- function(id){
  PERSON <- SAMPLE_OPIOID[SAMPLE_OPIOID$patient_id == id,]
  first_month <- PERSON[1, c('prescription_month')]
  if(first_month < 4){
    return (1)
  }else{
    return (0)
  }
}
SAMPLE_OPIOID$chronic <- unlist(lapply(as.vector(SAMPLE_OPIOID$patient_id), chronic_user))

# Throw away the chronic user and the first three months
SAMPLE_OPIOID <- SAMPLE_OPIOID[SAMPLE_OPIOID$chronic == 0 & SAMPLE_OPIOID$prescription_month > 3,]
write.csv(SAMPLE_OPIOID, "../Data/SAMPLE_OPIOID.csv", row.names = FALSE)

####################################
##### Store the benzo as well
####################################

SAMPLE_BENZO <- SAMPLE[SAMPLE$class == 'Benzodiazepine',]
SAMPLE_BENZO <- SAMPLE_BENZO[order(SAMPLE_BENZO$patient_id, SAMPLE_BENZO$date_filled),]
SAMPLE_BENZO$prescription_month <- month(as.POSIXlt(SAMPLE_BENZO$date_filled, format="%m/%d/%Y"))
SAMPLE_BENZO$prescription_year <- year(as.POSIXlt(SAMPLE_BENZO$date_filled, format="%m/%d/%Y"))
SAMPLE_BENZO$prescription_quarter <- unlist(lapply(SAMPLE_BENZO$prescription_month, compute_quarter))
write.csv(SAMPLE_BENZO, "../Data/SAMPLE_BENZO.csv", row.names = FALSE)


