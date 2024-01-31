from numpy import random
import pandas as pd

def create_pnr_list(max_pax, min_pax, num_pnr):
    PNR_id = 64124
    PNR_IDs = []
    PNR_pax = []
    PNR_class = []
    PNR_loyalty = []
    PNR_SSRs = []
    PNR_paid_service = []
    
    for pnr in range(num_pnr):
        current_PNR_id = PNR_id
        current_PNR_pax = random.randint(min_pax, max_pax+1)
        current_PNR_class = random.randint(1, 9)
        current_PNR_loyalty = random.randint(1, 5)
        current_PNR_SSR = random.randint(0,2)
        current_PNR_paid_service = random.randint(0,2)

        PNR_IDs.append(current_PNR_id)
        PNR_pax.append(current_PNR_pax)
        PNR_class.append(current_PNR_class)
        PNR_loyalty.append(current_PNR_loyalty)
        PNR_SSRs.append(current_PNR_SSR)
        PNR_paid_service.append(current_PNR_paid_service)

        PNR_id += 1
    
    funny = {
        "PNR_IDs": PNR_IDs,
        "PNR_pax": PNR_pax,
        "PNR_class": PNR_class, 
        "PNR_loyalty": PNR_loyalty, 
        "PNR_SSRs": PNR_SSRs,
        "PNR_paid_service": PNR_paid_service,
    }

    pnr_list = pd.DataFrame(data=funny)
    pnr_list.sort_values(by=["PNR_class", "PNR_loyalty", "PNR_IDs"], inplace=True, kind="stable")
    pnr_list = pnr_list.reset_index()
    pnr_list = pnr_list.iloc[:,1:]

    return pnr_list

max_number_of_pax_per_pnr = 4
min_number_of_pax_per_pnr = 1
number_of_pnr = 100
# class = random
# loyalty = random
# ssr = 1/0
# paid_request = 1/0

pnr_list = create_pnr_list(max_number_of_pax_per_pnr, min_number_of_pax_per_pnr, number_of_pnr)
pnr_list.to_csv("pnr_list.csv")