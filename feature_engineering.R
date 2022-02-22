# Feature engineering
setwd("~/Desktop/Research/Interpretable_Opioid/Code")
SAMPLE <- read.csv("../Data/SAMPLE_LABELED.csv")
PATIENT <- SAMPLE[SAMPLE$patient_id == '228',]
sum(SAMPLE$long_term_yet) # 1138
sum(SAMPLE$long_term_presc) # 2121

########################################################################
##### Whether patient has prescriptions in the past 90 days
########################################################################

past_presc <- function(pat_id, presc_id){
  PATIENT_PRESC <- SAMPLE[SAMPLE$patient_id == pat_id,]
  # for a prescription, look at previous prescription if within 90 days
  presc_index <- which(PATIENT_PRESC$prescription_id == presc_id)
  presc_date <- PATIENT_PRESC[presc_index, c("date_filled")]
  if(presc_index == 1){
    return (0)
  } else{
    prev_presc_date <- PATIENT_PRESC[presc_index - 1, c("date_filled")]
    if(as.numeric(difftime(as.Date(presc_date, format = "%m/%d/%Y"), 
                           as.Date(prev_presc_date, format = "%m/%d/%Y"), units = "days")) 
       <= 90){
      return (1)
    } else{
      return (0)
    }
  }
}
# PATIENT$past_prescription <- mapply(past_presc, PATIENT$patient_id, PATIENT$prescription_id)
SAMPLE$past_prescription <- mapply(past_presc, SAMPLE$patient_id, SAMPLE$prescription_id)

########################################################################
##### Prescriber info: # of patient, # of long term etc.
########################################################################

PRESCRIBER_TABLE <- read.csv("../Data/PRESCRIBER_TABLE.csv")

merge_prescriber_info <- function(presciber_id){
  PRESCRIBER <- PRESCRIBER_TABLE[PRESCRIBER_TABLE$Prescriber == presciber_id, ]
  return (c(PRESCRIBER$num_prescription, PRESCRIBER$num_patient, PRESCRIBER$num_patient_longterm))
}
prescriber_info <- mapply(merge_prescriber_info, SAMPLE$prescriber_id)
SAMPLE$prescriber_num_presc <- prescriber_info[1,]
SAMPLE$prescriber_num_pat <- prescriber_info[2,]
SAMPLE$prescriber_num_pat_long <- prescriber_info[3,]

########################################################################
##### Concurrent opioid prescription
########################################################################

SAMPLE$presc_until <- as.Date(SAMPLE$date_filled, format = "%m/%d/%Y") + SAMPLE$days_supply
SAMPLE$presc_until <- format(SAMPLE$presc_until, "%m/%d/%Y")
write.csv(SAMPLE, "../Data/SAMPLE_FEATURE.csv", row.names = FALSE)
SAMPLE <- read.csv("../Data/SAMPLE_FEATURE.csv")

compute_concurrent_opioid <- function(pat_id, presc_id){
  PATIENT_PRESC <- SAMPLE[SAMPLE$patient_id == pat_id,]
  # for a prescription, look at previous prescription if within 90 days
  presc_index <- which(PATIENT_PRESC$prescription_id == presc_id)
  presc_date <- PATIENT_PRESC[presc_index, c("date_filled")]
  
  if(presc_index == 1){
    return (0)
  } else{
    # for the ith prescription, less than i presc_until, that means one the previous
    # i-1 prescription has presc_until greater that current date filled
    num_presc <- nrow(PATIENT_PRESC[as.Date(PATIENT_PRESC$presc_until, format = "%m/%d/%Y") <= 
                                      as.Date(presc_date, format = "%m/%d/%Y"), ])
    
    if(num_presc < (presc_index - 1)){
      return (1)
    }else{
      return (0)
    }
  }
}
# PATIENT$concurrent_opioid <- mapply(compute_concurrent_opioid, PATIENT$patient_id, PATIENT$prescription_id)
SAMPLE$concurrent_opioid <- mapply(compute_concurrent_opioid, SAMPLE$patient_id, SAMPLE$prescription_id)

########################################################################
##### Benzo Concurrence
########################################################################

BENZO_TABLE <- read.csv("../Data/SAMPLE_BENZO.csv")

compute_concurrent_benzo <- function(pat_id, presc_id){
  PATIENT_PRESC_OPIOIDS <- SAMPLE[SAMPLE$prescription_id == presc_id,]
  presc_date <- PATIENT_PRESC_OPIOIDS$date_filled
  
  PATIENT_PRESC_BENZOS <- BENZO_TABLE[BENZO_TABLE$patient_id == pat_id, ]

  if (nrow(PATIENT_PRESC_BENZOS) == 0){
    return (0)
  } else{
    # another way to justify concurrence
    num_benzo <- nrow(PATIENT_PRESC_BENZOS[as.Date(PATIENT_PRESC_BENZOS$date_filled, format = "%m/%d/%Y") <= 
                                             as.Date(presc_date, format = "%m/%d/%Y") & 
                                             as.Date(PATIENT_PRESC_BENZOS$presc_until, format = "%m/%d/%Y") > 
                                             as.Date(presc_date, format = "%m/%d/%Y"), ])
    if(num_benzo > 0){
      return (1)
    } else {
      return (0)
    }
  }
}

# PATIENT <- SAMPLE[SAMPLE$patient_id == '439',]
# PATIENT$concurrent_benzo <- mapply(compute_concurrent_benzo, PATIENT$patient_id, PATIENT$prescription_id)
SAMPLE$concurrent_benzo <- mapply(compute_concurrent_benzo, SAMPLE$patient_id, SAMPLE$prescription_id)
write.csv(SAMPLE, "../Data/SAMPLE_FEATURE.csv", row.names = FALSE)

