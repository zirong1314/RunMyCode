1. The Data folder contains a subset of simulation data (two defective conditions: ESX-5 and ESY-5, and one normal condition: NS), as well as measured data under defective conditions. These data are provided for running and verifying the proposed PS method.

2. The MATLAB folder includes the implementation of the PS method. Specifically, PS_application.m and shape_adaptive_method.m correspond to the PS method and the localization method, respectively, while the remaining files are auxiliary subfunctions.

3. The Python folder contains the code for the identification model. The main.m file serves as the main entry point, and shice_qiyi.m is used for experiments on measured data. DCNN-SA-PARA.pth is the saved pretrained model used for defect identification.

4. If access to the complete dataset is required, please contact 23111406@bjtu.edu.cn and clearly state the intended use of the dataset.