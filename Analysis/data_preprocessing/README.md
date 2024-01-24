# LHCb B2L0hhh Data Preprocessing Pipeline

This pipeline is designed to preprocess `.root` files from the LHCb B2L0hhh data. It consists of three main Jupyter notebooks, each fulfilling a specific role in the data preparation process.

## 1. Reduce_Files.ipynb

### Purpose:
`Reduce_Files.ipynb` is responsible for reducing the size of the `.root` files generated from the LHCb bookkeeping system. It filters out unnecessary branches, leaving only those relevant for analysis. It also applies initial cuts to the data.

### Usage:
To use this notebook, specify the year, magnet polarity, decay tree, particle type (`LL` or `DD`), and the number of files you wish to process. The script will then process the specified `.root` files, applying the reduction and cuts as defined.

- **To process a specific configuration**:
  ```python
  year = "2017"
  magnet = "MagDown"
  decay_tree = "B2L0barPKpKm_LL/DecayTree"
  ll_or_dd = "LL"
  num_files = 16  # Number of files to process
  main(year, magnet, decay_tree, ll_or_dd, num_files)
- To process all files:
```python
main(year, magnet, decay_tree, ll_or_dd)  # Process all files
```

- **Process multiple configurations in a loop**:
```python
configurations = [("2017", "MagDown", "LL"), ("2018", "MagUp", "DD")]
for year, magnet, ll_or_dd in configurations:
    decay_tree = f"B2L0barPKpKm_{ll_or_dd}/DecayTree"
    main(year, magnet, decay_tree, ll_or_dd)
```
  
## 2. Check_Files.ipynb
### Purpose:
`Check_Files.ipynb` checks the integrity of the reduced `.root` files. It ensures that the required decay trees are present and that there are no errors or missing data within the files.

### Usage:
Specify the base directory where the processed files are located, along with the decay trees of interest. The script will iterate over the files, performing checks to confirm their integrity.

#### Example usage:
```python
data_folder_path = "/eos/lhcb/user/m/melashri/data/red_RD"
decay_trees = ['B2L0barPKpKm']
check_files(data_folder_path, decay_trees)
```

- **Check files for a single decay tree**:
```python
check_files("/eos/lhcb/user/m/melashri/data/red_RD", ["B2L0barPKpKm"])
```

- **Check files for multiple decay trees:**:
```python
decay_trees = ["B2L0barPKpKm", "B2L0barPPK"]
check_files("/eos/lhcb/user/m/melashri/data/red_RD", decay_trees)
```
- **Check files for different years and configurations:**:
```python
for year in ["2017", "2018"]:
    for decay_tree in ["B2L0barPKpKm", "B2L0barPPK"]:
        check_files("/eos/lhcb/user/m/melashri/data/red_RD", [decay_tree], year=year)
```

  
## 3. Merge_Files.ipynb
### Purpose:
`Merge_Files.ipynb` is used to merge the individual `.root` files into a single file for each specific configuration or for defined configurations.

### Usage:
Define the base directory, decay trees, years, magnet polarities, and particle types (`LL` or `DD`). The script will concatenate the files for each specified configuration.


- **Merge files for a single configuration**:
```python
main("/eos/lhcb/user/m/melashri/data/red_RD", ["B2L0barPKpKm"], ["2017"], ["MagDown"], ["LL"], True)

```

- **Merge files for multiple configurations**:
```python
decay_trees = ["B2L0barPKpKm", "B2L0barPPK"]
years = ["2016", "2017"]
magnets = ["MagDown", "MagUp"]
ll_or_dds = ["DD", "LL"]

for decay_tree in decay_trees:
    for year in years:
        for magnet in magnets:
            for ll_dd in ll_or_dds:
                main("/eos/lhcb/user/m/melashri/data/red_RD", [decay_tree], [year], [magnet], [ll_dd], True)
```

- **Sequential processing for limited resources:**:
```python
# Set use_parallel to False for systems with limited computational resources
main("/eos/lhcb/user/m/melashri/data/red_RD", ["B2L0barPKpKm"], ["2017"], ["MagDown"], ["LL"], False)
```


#### Example usage:
```python
base_dir = "/eos/lhcb/user/m/melashri/data/red_RD"
decay_trees = ["B2L0barPKpKm"]
years = ["2016"]
magnets = ["MagDown"]
ll_or_dds = ["DD", "LL"]
use_parallel = True 
main(base_dir, decay_trees, years, magnets, ll_or_dds, use_parallel)
```

Set `use_parallel` to False to run in serial mode (if you have memory over-use problems)

## General Instructions

- Ensure that all dependencies, such as `uproot`, `numpy`, and others, are installed in your Python environment.
- It's recommended to run these notebooks on a system with sufficient computational resources, especially if processing a large number of files with multiprocessing.
- The pipeline is designed to be modular, so you can run each notebook independently based on your specific requirements.

## Troubleshooting

If you encounter any issues while running these notebooks, consider the following steps:
- Check the file paths and ensure they are correct and accessible.
- Verify that the required Python packages are installed and up to date.
- For large datasets, ensure that your system has enough memory and storage capacity to handle the processing.

---

**Note:** This pipeline is tailored to the specific needs of the LHCb B2L0hhh data analysis based on the design and setup that the group is using right now. 
