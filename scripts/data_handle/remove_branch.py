import ROOT

# Branches to remove
branches_to_remove = set([
    "p_MC12TuneV4_ProbNNghost",
    "Lp_MC15TuneV1_ProbNNe"
    # ... Add the rest of the branches you want to remove here ...
])

# Open the original ROOT file
original_file = ROOT.TFile.Open("/data/home/melashri/BPhysics/Mohamed/B2L0pbarKpKp.root")

# Access the tree you want to keep
tree = original_file.Get("DecayTree")

# Create a new ROOT file
new_file = ROOT.TFile("B2L0pbarKpKp_filtered.root", "RECREATE")

# Clone the tree without the branches you want to remove
branch_list = tree.GetListOfBranches()
branches_to_keep = [branch.GetName() for branch in branch_list if branch.GetName() not in branches_to_remove]
new_tree = tree.CloneTree(0)
for branch in branches_to_keep:
    tree.SetBranchStatus(branch, 1)
new_tree = tree.CloneTree()

# Write the new tree to the new file
new_tree.Write()

# Close the new file
new_file.Close()

print("Done!")
