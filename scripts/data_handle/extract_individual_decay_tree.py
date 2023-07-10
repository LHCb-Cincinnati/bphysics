import ROOT

# Create a TChain to hold the trees
chain = ROOT.TChain("B2L0pbarKpKp/DecayTree")

# Paths to the original ROOT files
base_path = '/share/lazy/Bu2LambdaPHH/NTuples/RD/'
file_names = ['L0phh_18MU.root', 'L0phh_18MD.root', 'L0phh_17MU.root', 'L0phh_17MD.root', 
              'L0phh_16MU.root', 'L0phh_16MD.root', 'L0phh_15MU.root', 'L0phh_15MD.root']

# Add each file to the chain
for file_name in file_names:
    chain.Add(base_path + file_name)

# Create a new ROOT file
new_file = ROOT.TFile("B2L0pbarKpKp.root", "RECREATE")

# Clone the chain (which is effectively all the trees combined) into a new tree
new_tree = chain.CloneTree()

# Write the new tree to the new file
new_tree.Write()

# Close the new file
new_file.Close()

print("Done!")
