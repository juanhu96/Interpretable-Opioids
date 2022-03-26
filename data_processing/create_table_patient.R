# Create a table of all patients with information during naive period
# This is also our input of the ML model
library(dplyr)
library(lubridate)
setwd("~/Desktop/Research/Interpretable_Opioid/Code")
SAMPLE <- read.csv("../Data/Prescription/SAMPLE_LABEL_FEATURE.csv")

# Compute the basic info & naive period for each patient
PATIENT_BASIC = SAMPLE %>% 
  group_by(patient_id) %>% 
  summarize(Gender = ifelse(patient_gender[1] == "M", 0, 1),
            Year = patient_birth_year[1],
            Zip = patient_zip[1],
            Naive_date = date_filled[1],
            Long_term_user = ifelse(sum(long_term_yet) > 0, 1, 0),
            Long_term_date = ifelse(sum(long_term_yet) > 0, date_filled[long_term_yet == 1][1], NA)
            )
PATIENT_BASIC$Naive_date_end <-  format(as.Date(PATIENT_BASIC$Naive_date, format = "%m/%d/%Y") + 45, "%m/%d/%Y")
PATIENT_BASIC$Age <- year(as.POSIXlt(PATIENT_BASIC$Naive_date, format="%m/%d/%Y")) - PATIENT_BASIC$Year
PATIENT_BASIC$Age_Gender <- PATIENT_BASIC$Age * PATIENT_BASIC$Gender
PATIENT_BASIC = select(PATIENT_BASIC, patient_id, Gender, Age, Age_Gender, Zip, Naive_date, 
                       Naive_date_end, Long_term_user, Long_term_date)

# Filter the prescriptions within naive period
SAMPLE$Naive_date_end <- mapply(function(pat_id){return(PATIENT_BASIC[PATIENT_BASIC$patient_id == pat_id, ]$Naive_date_end[1])}, SAMPLE$patient_id)
SAMPLE_NAIVE_PERIOD = SAMPLE %>% filter(as.Date(date_filled, format = "%m/%d/%Y") < as.Date(Naive_date_end, format = "%m/%d/%Y"))

# Compute the info during naive period
# Multiple insurance/drug during naive period
PATIENT_NAIVE = SAMPLE_NAIVE_PERIOD %>% 
  group_by(patient_id) %>% 
  summarize(Num_presc = n_distinct(prescription_id),
            Total_days_supply = sum(days_supply),
            Avg_MME = mean(daily_dose),
            Concurrent_opioid = ifelse(sum(concurrent_opioid > 0), 1, 0),
            Concurrent_benzo = ifelse(sum(concurrent_benzo > 0), 1, 0),
            Num_drug = n_distinct(drug),
            Insurance_change = ifelse(n_distinct(payment) > 1, 1, 0),
            Codeine = sum(drug == "Codeine"),
            Codeine_MME = mean(daily_dose[drug == "Codeine"]),
            Hydrocodone = sum(drug == "Hydrocodone"),
            Hydrocodone_MME = mean(daily_dose[drug == "Hydrocodone"]),
            Oxycodone = sum(drug == "Oxycodone"),
            Oxycodone_MME = mean(daily_dose[drug == "Oxycodone"]),
            Morphine = sum(drug == "Morphine"),
            Morphine_MME = mean(daily_dose[drug == "Morphine"]),
            Hydromorphone = sum(drug == "Hydromorphone"),
            Hydromorphone_MME = mean(daily_dose[drug == "Hydromorphone"]),
            Methadone = sum(drug == "Methadone"),
            Methadone_MME = mean(daily_dose[drug == "Methadone"]),
            Fentanyl = sum(drug == "Fentanyl"),
            Fentanyl_MME = mean(daily_dose[drug == "Fentanyl"]),
            Oxymorphone = sum(drug == "Oxymorphone"),
            Oxymorphone_MME = mean(daily_dose[drug == "Oxymorphone"]),
            Medicaid = sum(payment == "Medicaid"),
            CommercialIns = sum(payment == "CommercialIns"),
            Medicare = sum(payment == "Medicare"),
            CashCredit = sum(payment == "CashCredit"),
            MilitaryIns = sum(payment == "MilitaryIns"),
            WorkersComp = sum(payment == "WorkersComp"),
            Other = sum(payment == "Other"),
            IndianNation = sum(payment == "IndianNation")
            )

PATIENT_NAIVE[is.na(PATIENT_NAIVE)] <- 0

# Create interaction between insurance_change * insurance
PATIENT_NAIVE$Change_Medicaid = PATIENT_NAIVE$Insurance_change * PATIENT_NAIVE$Medicaid
PATIENT_NAIVE$Change_CommercialIns = PATIENT_NAIVE$Insurance_change * PATIENT_NAIVE$CommercialIns
PATIENT_NAIVE$Change_Medicare = PATIENT_NAIVE$Insurance_change * PATIENT_NAIVE$Medicare
PATIENT_NAIVE$Change_CashCredit = PATIENT_NAIVE$Insurance_change * PATIENT_NAIVE$CashCredit
PATIENT_NAIVE$Change_MilitaryIns = PATIENT_NAIVE$Insurance_change * PATIENT_NAIVE$MilitaryIns
PATIENT_NAIVE$Change_WorkersComp = PATIENT_NAIVE$Insurance_change * PATIENT_NAIVE$WorkersComp
PATIENT_NAIVE$Change_Other = PATIENT_NAIVE$Insurance_change * PATIENT_NAIVE$Other
PATIENT_NAIVE$Change_IndianNation = PATIENT_NAIVE$Insurance_change * PATIENT_NAIVE$IndianNation

# Merge this with the basic info
PATIENT <- merge(PATIENT_BASIC, PATIENT_NAIVE, by = "patient_id")
colnames(PATIENT)[1] <- "Patient_ID"
write.csv(PATIENT, "../Data/PATIENT_TABLE.csv", row.names = FALSE)

