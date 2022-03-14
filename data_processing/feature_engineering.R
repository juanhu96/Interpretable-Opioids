# Feature engineering
library(dplyr)
setwd("~/Desktop/Research/Interpretable_Opioid/Code")
SAMPLE <- read.csv("../Data/SAMPLE_LABELED.csv")
# PATIENT <- SAMPLE[SAMPLE$patient_id == '228',]
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

# We cannot use PRESCRIBER_TABLE because this aggregated over the entire year
# merge_prescriber_info <- function(pat_id, presc_id){
#   PATIENT_PRESC <- SAMPLE[SAMPLE$prescription_id == presc_id,]
#   PRESCRIBER_ID <- PATIENT_PRESC$prescriber_id
#   presc_date <- PATIENT_PRESC$date_filled
#   presc_month <- PATIENT_PRESC$prescription_month
#   
#   # prescriber's prescription in the past three months
#   PRESCRIBER_PRESC <- SAMPLE[SAMPLE$prescriber_id == PRESCRIBER_ID & 
#                                SAMPLE$patient_id != pat_id &
#                                as.Date(SAMPLE$date_filled, format = "%m/%d/%Y") < 
#                                as.Date(presc_date, format = "%m/%d/%Y"),]
#   
#   num_other_presc <- nrow(PRESCRIBER_PRESC)
#   num_other_patient = 0
#   num_other_patient_long_term = 0
#   
#   # if there are prescriptions for other patients
#   if (num_other_presc != 0){
#     patient_list <- unique(PRESCRIBER_PRESC$patient_id)
#     num_other_patient <- length(patient_list)
#     for (i in 1:num_other_patient){
#       other_patient_id <- patient_list[i]
#       OTHER_PATIENT_PRESC <- PRESCRIBER_PRESC[PRESCRIBER_PRESC$patient_id == other_patient_id, ]
#       # see if it has already become long term user
#       if(sum(OTHER_PATIENT_PRESC$long_term_yet) > 0){
#         num_other_patient_long_term = num_other_patient_long_term + 1
#       }
#     }
#   }
#   return (c(num_other_presc, num_other_patient, num_other_patient_long_term))
# }

# prescriber_info <- mapply(merge_prescriber_info, SAMPLE$patient_id, SAMPLE$prescription_id)
# SAMPLE$prescriber_num_presc <- prescriber_info[1,]
# SAMPLE$prescriber_num_pat <- prescriber_info[2,]
# SAMPLE$prescriber_num_pat_long <- prescriber_info[3,]

# Another better way using new PRESCRIBER_TABLE
PRESCRIBER_TABLE <- read.csv("../Data/PRESCRIBER_TABLE.csv")
merge_prescriber_info_new <- function(pat_id, presc_id){
  PATIENT_PRESC <- SAMPLE[SAMPLE$prescription_id == presc_id,]
  PRESCRIBER_ID <- PATIENT_PRESC$prescriber_id
  presc_quarter <- PATIENT_PRESC$prescription_quarter
  
  # use the current quarter instead of previous quarter as approx
  # since we throw away the first quarter as burnin
  PRESCRIBER_SUBTABLE <- PRESCRIBER_TABLE[PRESCRIBER_TABLE$prescriber_id == PRESCRIBER_ID & 
                                            PRESCRIBER_TABLE$prescription_quarter == presc_quarter - 1, ]
  
  num_presc = sum(PRESCRIBER_SUBTABLE$num_prescription)
  num_pat = sum(PRESCRIBER_SUBTABLE$num_patient)
  num_pat_long = sum(PRESCRIBER_SUBTABLE$num_patient_long)
  averageMME = sum(PRESCRIBER_SUBTABLE$averageMME)
  average_days_supply = sum(PRESCRIBER_SUBTABLE$average_days_supply)
  
  return (c(num_pat, num_pat_long, num_presc, averageMME, average_days_supply))
}
prescriber_info <- mapply(merge_prescriber_info_new, SAMPLE$patient_id, SAMPLE$prescription_id)
SAMPLE$prescriber_num_pat <- prescriber_info[1,]
SAMPLE$prescriber_num_pat_long <- prescriber_info[2,]
SAMPLE$prescriber_num_presc <- prescriber_info[3,]
SAMPLE$AvgMME <- prescriber_info[4,]
SAMPLE$Avg_days_supply <- prescriber_info[5,]

########################################################################
##### Concurrent opioid prescription
########################################################################

SAMPLE$presc_until <- as.Date(SAMPLE$date_filled, format = "%m/%d/%Y") + SAMPLE$days_supply
SAMPLE$presc_until <- format(SAMPLE$presc_until, "%m/%d/%Y")

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

########################################################################
##### Age, reorder columns
########################################################################

SAMPLE$Age <- SAMPLE$prescription_year - SAMPLE$patient_birth_year
write.csv(SAMPLE, "../Data/SAMPLE_LABEL_FEATURE.csv", row.names = FALSE)

########################################################################

colnames(SAMPLE)
SAMPLE_reorder <- SAMPLE[, c("prescription_id", "patient_id", "patient_birth_year",
                           "Age", "patient_gender", "patient_zip", "prescriber_id", 
                           "prescriber_zip", "prescriber_num_presc", "prescriber_num_pat",
                           "prescriber_num_pat_long", "AvgMME", "Avg_days_supply",
                           "pharmacy_id", "pharmacy_zip", "strength", "quantity", "days_supply", 
                           "quantity_per_day", "conversion", "date_filled", "presc_until",
                           "class", "drug", "PRODUCTNDC", "PROPRIETARYNAME",
                           "LABELERNAME", "ROUTENAME", "DEASCHEDULE", "MAINDOSE",
                           "daily_dose", "total_dose", "payment", "prescription_month", 
                           "prescription_year", "prescription_quarter",
                           "past_prescription", "concurrent_opioid", "concurrent_benzo",
                           "long_term_yet", "long_term_presc")]

write.csv(SAMPLE_reorder, "../Data/SAMPLE_LABEL_FEATURE.csv", row.names = FALSE)

########################################################################
##### Compute prior information
##### Count of prior prescription, total days of supply, total quantity
########################################################################

SAMPLE <- read.csv("../Data/SAMPLE_LABEL_FEATURE.csv")

compute_prior_info <- function(pat_id, presc_id){
  PATIENT_PRESC <- SAMPLE[SAMPLE$prescription_id == presc_id,]
  PRESC_DATE <- PATIENT_PRESC$date_filled
  
  PATIENT_PRESC_PAST <- SAMPLE[SAMPLE$patient_id == pat_id &
                               as.Date(SAMPLE$date_filled, format = "%m/%d/%Y") < 
                               as.Date(PRESC_DATE, format = "%m/%d/%Y"),]
  
  # In the past 90 days vs. over the entire period?
  # Past_perscription indicates whether there's a prescription in the past 90 days
  # For these we are consider over the entire time period.
  prior_presc <- nrow(PATIENT_PRESC_PAST)
  prior_days_supply <- sum(PATIENT_PRESC_PAST$days_supply)
  prior_quantity <- sum(PATIENT_PRESC_PAST$quantity)
  
  return (c(prior_presc, prior_days_supply, prior_quantity))
}


# PATIENT <- SAMPLE[SAMPLE$patient_id == '439',]
# prior_info <- mapply(compute_prior_info, PATIENT$patient_id, PATIENT$prescription_id)
# PATIENT$count_prior_presc = prior_info[1, ]
# PATIENT$count_prior_days_supply = prior_info[2, ]
# PATIENT$count_prior_quantity = prior_info[3, ]
prior_info <- mapply(compute_prior_info, SAMPLE$patient_id, SAMPLE$prescription_id)
SAMPLE$count_prior_presc = prior_info[1, ]
SAMPLE$count_prior_days_supply = prior_info[2, ]
SAMPLE$count_prior_quantity = prior_info[3, ]

# update the table
write.csv(SAMPLE, "../Data/SAMPLE_LABEL_FEATURE.csv", row.names = FALSE)
