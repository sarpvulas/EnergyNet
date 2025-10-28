import platform

if platform.system() == 'Windows':
    BASE_DIR = r'C:\Users\Sarp\Datasets\energynetdata\icc_combined'

elif platform.system() == "Darwin":
    BASE_DIR = r'/Users/sarpvulas/Datasets/energynetdata/icc_combined'
