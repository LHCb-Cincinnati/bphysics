import ROOT

# List of decay trees
decay_trees = ["B2L0pbarpippim", "B2L0pbarpippip", "B2L0pbarKmpip", "B2L0pbarKppip", 
               "B2L0pbarKppim", "B2L0pbarKpKm", "B2L0pbarpp", "B2L0pbarKpKp"]

# Paths to the original ROOT files
base_path = '/share/lazy/Bu2LambdaPHH/NTuples/RD/'
file_names = ['L0phh_18MU.root', 'L0phh_18MD.root', 'L0phh_17MU.root', 'L0phh_17MD.root', 
              'L0phh_16MU.root', 'L0phh_16MD.root', 'L0phh_15MU.root', 'L0phh_15MD.root']

# Loop over each decay tree
for decay_tree in decay_trees:
    # Create a TChain to hold the trees
    chain = ROOT.TChain(decay_tree + "/DecayTree")

    # Add each file to the chain
    for file_name in file_names:
        chain.Add(base_path + file_name)

    # Create a new ROOT file for each decay tree
    new_file = ROOT.TFile(decay_tree + ".root", "RECREATE")

    # Clone the chain (which is effectively all the trees combined) into a new tree
    new_tree = chain.CloneTree()

    # Write the new tree to the new file
    new_tree.Write()

    # Close the new file
    new_file.Close()

print("Done!")
