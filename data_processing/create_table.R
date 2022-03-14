# Identify long-term user and initial prescription
library(dplyr)
setwd("~/Desktop/Research/Interpretable_Opioid/Code")
SAMPLE <- read.csv("../Data/SAMPLE_OPIOID.csv")
SAMPLE$prescription_id = seq.int(nrow(SAMPLE))

########################################################################
##### Determine if a patient is ever a long term user
########################################################################

# A patient is consider a long-term user if he/she is on prescription 
# at least 90 days of supply in the past 180 days (2 quarters)
# 1 if patient has became long-term user at current prescription
long_term_yet <- function(pat_id, presc_id){
  
  # All prescription of a person in time order
  PATIENT <- SAMPLE[SAMPLE$patient_id == pat_id,]
  
  # The index/quarter of current prescription within the patient
  presc_index <- which(PATIENT$prescription_id == presc_id)
  presc_quarter <- PATIENT[presc_index, c('prescription_quarter')]

  # Focus on the prescriptions after and within 2 quarter only
  PATIENT_BEFORE <- PATIENT[1:presc_index, ]
  PATIENT_BEFORE <- PATIENT_BEFORE[PATIENT_BEFORE$prescription_quarter == presc_quarter | 
                                     PATIENT_BEFORE$prescription_quarter == (presc_quarter - 1), ]
  
  if(sum(PATIENT_BEFORE$days_supply) >= 90){
    return (1)
  } else{
    return (0)
  }
}

# PATIENT <- SAMPLE[SAMPLE$patient_id == '228',]
# PATIENT$long_term <- mapply(long_term_yet, PATIENT$patient_id, PATIENT$prescription_id)
SAMPLE$long_term_yet <- mapply(long_term_yet, SAMPLE$patient_id, SAMPLE$prescription_id)

########################################################################
##### Determine the date, create patient table
########################################################################

PATIENT_TABLE <- data.frame(Patient = unique(SAMPLE$patient_id))
# Return the date of the patient becomes long-term user, NA if short term
long_term_date <- function(pat_id){
  PATIENT <- SAMPLE[SAMPLE$patient_id == pat_id,]
  # at least one of the prescription is long term
  if(sum(PATIENT$long_term) >= 1){
    PATIENT_TEMP <- PATIENT[PATIENT$long_term == 1, ]
    date <- PATIENT_TEMP$date_filled[1]
    return (date)
  } else{
    return (NA)
  }
}
PATIENT_TABLE$Long_term_date <- mapply(long_term_date, PATIENT_TABLE$Patient)
PATIENT_TABLE$Gender <- mapply(function(pat_id){return(SAMPLE[SAMPLE$patient_id == pat_id,]$patient_gender[1])}, PATIENT_TABLE$Patient)
PATIENT_TABLE$Birth_year <- mapply(function(pat_id){return(SAMPLE[SAMPLE$patient_id == pat_id,]$patient_birth_year[1])}, PATIENT_TABLE$Patient)
PATIENT_TABLE$Zip <- mapply(function(pat_id){return(SAMPLE[SAMPLE$patient_id == pat_id,]$patient_zip[1])}, PATIENT_TABLE$Patient)
# length(unique(PATIENT_TABLE$Long_term_date)) # 167 out of 13987 patient
write.csv(PATIENT_TABLE, "../Data/PATIENT_TABLE.csv", row.names = FALSE)

########################################################################
##### Update long term label in prescription table
########################################################################

long_term_presc <- function(pat_id, presc_id){
  # Date become long term user, could be NA
  PATIENT <- PATIENT_TABLE[PATIENT_TABLE$Patient == pat_id,]
  longterm_date <- PATIENT$Long_term_date
  
  if (is.na(longterm_date)){
    return (0)
  } else{
    # Date prescription filled
    PATIENT_PRESC <- SAMPLE[SAMPLE$patient_id == pat_id,]
    presc_index <- which(PATIENT_PRESC$prescription_id == presc_id)
    presc_date <- PATIENT_PRESC[presc_index, c('date_filled')]
    if (as.numeric(difftime(as.Date(longterm_date, format = "%m/%d/%Y"), 
                            as.Date(presc_date, format = "%m/%d/%Y"), units = "days")) 
        <= 90){
      return (1)
    } else{
      return (0)
    }
  }
}
# PATIENT <- SAMPLE[SAMPLE$patient_id == '228',]
# PATIENT$long_term <- mapply(long_term_presc, PATIENT$patient_id, PATIENT$prescription_id)
SAMPLE$long_term_presc <- mapply(long_term_presc, SAMPLE$patient_id, SAMPLE$prescription_id)
write.csv(SAMPLE, "../Data/SAMPLE_LABELED.csv", row.names = FALSE)

########################################################################
##### Create prescriber table
########################################################################

# Note: better merge all into one function?
# PRESCRIBER_TABLE <- data.frame(Prescriber = unique(SAMPLE$prescriber_id))
# PRESCRIBER_TABLE$Zip <- mapply(function(id){return(SAMPLE[SAMPLE$prescriber_id == id,]$prescriber_zip[1])}, PRESCRIBER_TABLE$Prescriber)
# PRESCRIBER_TABLE$num_prescription <- mapply(function(id){return(nrow(SAMPLE[SAMPLE$prescriber_id == id,]))}, PRESCRIBER_TABLE$Prescriber)
# PRESCRIBER_TABLE$num_patient <- mapply(function(id){return(length(unique(SAMPLE[SAMPLE$prescriber_id == id,]$patient_id)))}, PRESCRIBER_TABLE$Prescriber)

# Count the number of long term users a prescriber has
# num_long_term <- function(id){
#   PRESCRIBER <- SAMPLE[SAMPLE$prescriber_id == id,]
#   PATIENT_ID_LIST <- unique(PRESCRIBER$patient_id)
#   
#   PATIENT_LIST <- PATIENT_TABLE[PATIENT_TABLE$Patient %in% PATIENT_ID_LIST, ]
#   return (sum(!is.na(PATIENT_LIST$Long_term_date)))
# }
# PRESCRIBER_TABLE$num_patient_longterm <- mapply(num_long_term, PRESCRIBER_TABLE$Prescriber)
# write.csv(PRESCRIBER_TABLE, "../Data/PRESCRIBER_TABLE.csv", row.names = FALSE)


# Another table prescriber table, better for later calculation
SAMPLE <- read.csv("../Data/SAMPLE_LABELED.csv")
PRESCRIBER = SAMPLE %>% group_by(prescriber_id, prescription_quarter) %>% 
  summarize(num_patient = n_distinct(patient_id), 
            num_patient_long = n_distinct(patient_id[long_term_yet == 1]),
            num_prescription = n_distinct(prescription_id),
            averageMME = mean(daily_dose),
            average_days_supply = mean(days_supply),
            Zip = prescriber_zip[1])

# Merge the demographic data (do not merge now, not all prescriber zip has info in DEMO)
DEMO <- read.csv("../Data/DEMO.csv")
PRESCRIBER_TABLE <- merge(PRESCRIBER, DEMO[c("Zip", "Population", "PopDensity",
                                             "MedianHHIncome", "MedianAge",
                                             "Poverty", "Unemployment")], by=c("Zip"))

# Find zip code in 

# In our dataset, but not in the CA demo info
library(muRL)
ZipMissed <- subset(PRESCRIBER$Zip, !(PRESCRIBER$Zip %in% DEMO$Zip))
ZipMissed <- as.data.frame(ZipMissed)
colnames(ZipMissed) = 'zip'
zip.plot(ZipMissed, map.type = "usa", cex = 0.6)
zip.plot(ZipMissed, map.type = "state", region = "california", cex = 1)

# write.csv(PRESCRIBER, "../Data/PRESCRIBER_TABLE.csv", row.names = FALSE)

########################################################################
##### Modify benzo table
########################################################################

BENZO_TABLE <- read.csv("../Data/SAMPLE_BENZO.csv")
BENZO_TABLE$presc_until <- as.Date(BENZO_TABLE$date_filled, format = "%m/%d/%Y") + BENZO_TABLE$days_supply
BENZO_TABLE$presc_until <- format(BENZO_TABLE$presc_until, "%m/%d/%Y")
write.csv(BENZO_TABLE, "../Data/SAMPLE_BENZO.csv", row.names = FALSE)
