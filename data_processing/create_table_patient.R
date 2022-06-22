# Create a table of all patients with information during naive period
# This is also our input of the ML model
library(dplyr)
library(lubridate)
library(arules)
setwd("~/Desktop/Research/Interpretable Opioid/Code")
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

# For the first presciption, its previous payment is set to current payment
# by default for convenience
SAMPLE_NAIVE_PERIOD = SAMPLE_NAIVE_PERIOD %>% group_by(patient_id) %>% 
  mutate(Previous_payment = ifelse(past_prescription == 1, dplyr::lag(payment), payment))

SAMPLE_NAIVE_PERIOD = SAMPLE_NAIVE_PERIOD %>% 
  mutate(Switch_payment = ifelse(Previous_payment == payment, "None", paste(Previous_payment, payment, sep = "_")))

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
            IndianNation = sum(payment == "IndianNation"),
            Medicaid_CommercialIns = sum(Switch_payment == "Medicaid_CommercialIns"),
            Medicaid_Medicare = sum(Switch_payment == "Medicaid_Medicare"),
            Medicaid_CashCredit = sum(Switch_payment == "Medicaid_CashCredit"),
            Medicaid_MilitaryIns = sum(Switch_payment == "Medicaid_MilitaryIns"),
            Medicaid_WorkersComp = sum(Switch_payment == "Medicaid_WorkersComp"),
            Medicaid_Other = sum(Switch_payment == "Medicaid_Other"),
            Medicaid_IndianNation = sum(Switch_payment == "Medicaid_IndianNation"),
            CommercialIns_Medicaid = sum(Switch_payment == "CommercialIns_Medicaid"),
            CommercialIns_Medicare = sum(Switch_payment == "CommercialIns_Medicare"),
            CommercialIns_CashCredit = sum(Switch_payment == "CommercialIns_CashCredit"),
            CommercialIns_MilitaryIns = sum(Switch_payment == "CommercialIns_MilitaryIns"),
            CommercialIns_WorkersComp = sum(Switch_payment == "CommercialIns_WorkersComp"),
            CommercialIns_Other = sum(Switch_payment == "CommercialIns_Other"),
            CommercialIns_IndianNation = sum(Switch_payment == "CommercialIns_IndianNation"),
            Medicare_Medicaid = sum(Switch_payment == "Medicare_Medicaid"),
            Medicare_CommercialIns = sum(Switch_payment == "Medicare_CommercialIns"),
            Medicare_CashCredit = sum(Switch_payment == "Medicare_CashCredit"),
            Medicare_MilitaryIns = sum(Switch_payment == "Medicare_MilitaryIns"),
            Medicare_WorkersComp = sum(Switch_payment == "Medicare_WorkersComp"),
            Medicare_Other = sum(Switch_payment == "Medicare_Other"),
            Medicare_IndianNation = sum(Switch_payment == "Medicare_IndianNation"),
            CashCredit_Medicaid = sum(Switch_payment == "CashCredit_Medicaid"),
            CashCredit_CommercialIns = sum(Switch_payment == "CashCredit_CommercialIns"),
            CashCredit_Medicare = sum(Switch_payment == "CashCredit_Medicare"), 	
            CashCredit_MilitaryIns = sum(Switch_payment == "CashCredit_MilitaryIns"),
            CashCredit_WorkersComp = sum(Switch_payment == "CashCredit_WorkersComp"),
            CashCredit_Other = sum(Switch_payment == "CashCredit_Other"),
            CashCredit_IndianNation = sum(Switch_payment == "CashCredit_IndianNation"),
            MilitaryIns_Medicaid = sum(Switch_payment == "MilitaryIns_Medicaid"),
            MilitaryIns_CommercialIns = sum(Switch_payment == "MilitaryIns_CommercialIns"),
            MilitaryIns_Medicare = sum(Switch_payment == "MilitaryIns_Medicare"),
            MilitaryIns_CashCredit = sum(Switch_payment == "MilitaryIns_CashCredit"),
            MilitaryIns_WorkersComp = sum(Switch_payment == "MilitaryIns_WorkersComp"),
            MilitaryIns_Other = sum(Switch_payment == "MilitaryIns_Other"),
            MilitaryIns_IndianNation = sum(Switch_payment == "MilitaryIns_IndianNation"),
            WorkersComp_Medicaid = sum(Switch_payment == "WorkersComp_Medicaid"),
            WorkersComp_CommercialIns = sum(Switch_payment == "WorkersComp_CommercialIns"),
            WorkersComp_Medicare = sum(Switch_payment == "WorkersComp_Medicare"),
            WorkersComp_CashCredit = sum(Switch_payment == "WorkersComp_CashCredit"),
            WorkersComp_MilitaryIns = sum(Switch_payment == "WorkersComp_MilitaryIns"),
            WorkersComp_Other = sum(Switch_payment == "WorkersComp_Other"),
            WorkersComp_IndianNation = sum(Switch_payment == "WorkersComp_IndianNation"),
            Other_Medicaid = sum(Switch_payment == "Other_Medicaid"),
            Other_CommercialIns = sum(Switch_payment == "Other_CommercialIns"),
            Other_Medicare = sum(Switch_payment == "Other_Medicare"),
            Other_CashCredit = sum(Switch_payment == "Other_CashCredit"),
            Other_MilitaryIns = sum(Switch_payment == "Other_MilitaryIns"),
            Other_WorkersComp = sum(Switch_payment == "Other_WorkersComp"),
            Other_IndianNation = sum(Switch_payment == "Other_IndianNation"),
            IndianNation_Medicaid = sum(Switch_payment == "IndianNation_Medicaid"),
            IndianNation_CommercialIns = sum(Switch_payment == "IndianNation_CommercialIns"),
            IndianNation_Medicare = sum(Switch_payment == "IndianNation_Medicare"),
            IndianNation_CashCredit = sum(Switch_payment == "IndianNation_CashCredit"),
            IndianNation_MilitaryIns = sum(Switch_payment == "IndianNation_MilitaryIns"),
            IndianNation_WorkersComp = sum(Switch_payment == "IndianNation_WorkersComp"),
            IndianNation_Other = sum(Switch_payment == "IndianNation_Other")
            )
  

PATIENT_NAIVE[is.na(PATIENT_NAIVE)] <- 0

# Merge this with the basic info
PATIENT <- merge(PATIENT_BASIC, PATIENT_NAIVE, by = "patient_id")
colnames(PATIENT)[1] <- "Patient_ID"
write.csv(PATIENT, "../Data/PATIENT_TABLE.csv", row.names = FALSE)

########################################################################

PATIENT <- read.csv("../Data/PATIENT_TABLE.csv")
STUMPS <- read.csv("../Data/stumps.csv")

### Sample a small patient dataset for training
# PATIENT_opioid <- PATIENT %>% filter(Long_term_user == 1)
# PATIENT_none_opioid <- PATIENT %>% filter(Long_term_user == 0)
# SAMPLE_PATIENT_opioid <- PATIENT_opioid[sample(nrow(PATIENT_opioid), 100), ]
# SAMPLE_PATIENT_none_opioid <- PATIENT_none_opioid[sample(nrow(PATIENT_none_opioid), 100), ]
# SAMPLE_PATIENT <- bind_rows(SAMPLE_PATIENT_opioid, SAMPLE_PATIENT_none_opioid)

SAMPLE_PATIENT <- PATIENT

SAMPLE_PATIENT <- select(SAMPLE_PATIENT, -c(Patient_ID, Zip, Naive_date, Naive_date_end, Long_term_date))
SAMPLE_PATIENT <- SAMPLE_PATIENT %>% select(Long_term_user, everything())

write.csv(SAMPLE_PATIENT, "../Data/SAMPLE_PATIENT_DATA.csv", row.names = FALSE)
