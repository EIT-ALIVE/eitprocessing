# %%

data_path = "/Volumes/Beademing/COVID-19/Studies/Ptp vs EIT/Data/C055/C055.bin"

# %%
tiv = TIV()
all_mean_tiv = []

for peep_value,peep_seq in zip(peep_values,peep_sequences):
    eit_data = peep_seq.eit_data['masked pixels']
    continuous_data = peep_seq.continuous_data['functional impedance (masked pixels)'] 

    tidal_impedance_variance = tiv.compute_parameter(eit_data, continuous_data, peep_seq).values 
    mean_tiv = np.nanmean(tidal_impedance_variance, axis=0) 
    all_mean_tiv.append(mean_tiv)]

all_mean_tiv = np.array(all_mean_tiv)

# Omzetten naar een lijst met 1024 lijsten (32x32 pixels), elk met 11 getallen (voor elke PEEP-stap)
mean_tiv_list = [list(all_mean_tiv[:, i, j]) for i in range(32) for j in range(32)]

# Alle lists met nans worden eruit gegooid
filtered_mean_tiv_list = [tiv for tiv in mean_tiv_list if not np.isnan(tiv).all()]

# alle NaNs die er nog zijn vervangen met 0 --> BEN HET HIER NOG NIET MEE EENS, MAAR NU KAN IK WEL DE BEREKENINGEN VERDER UITVOEREN
delta_Z_values = [[0 if np.isnan(value) else value for value in sublist] for sublist in filtered_mean_tiv_list] 
