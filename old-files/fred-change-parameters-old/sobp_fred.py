import pandas as pd
fred_input = "/home/pablo/ProstateFred/InitialTests/sobp/sobp-fred/1e5-original-fred.inp"
weights_csv = pd.read_csv("/home/pablo/prototwin/activity-super-resolution/fred2topas/numbers_sobp.dat", delimiter='\s+', header=None)

lines = []
for index, row in weights_csv.iterrows():
    if int(row[1]) > 0:
        line = f"pb: {int(row[0])} Phantom; particle = proton; T = {row[2] / 1000:.3f}; v=[-1,0,0]; P=[17, {-row[3] / 10:.1f}, {10 - row[4] / 10:.1f}]; Xsec = gauss; FWHMx=0.664; FWHMy=0.615; nprim={1e5:.0f}; N={1e6*row[1]:.0f};"
        lines.append(line)

# Write all lines at once to reduce I/O operations
with open(fred_input, 'a') as file:
    file.write("\n".join(lines))