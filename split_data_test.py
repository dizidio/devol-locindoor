import numpy as np

data = np.genfromtxt('hists_test_050718_15aps_25bins_10rows_asus_clean.csv',delimiter=",");
np.random.shuffle(data)

np.savetxt('hists_new_val_050718_15aps_25bins_10rows.txt',data[:2000], delimiter=",",fmt='%d')
np.savetxt('hists_new_test_050718_15aps_25bins_10rows.txt',data[2000:], delimiter=",",fmt='%d')
