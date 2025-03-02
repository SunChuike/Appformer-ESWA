# Appformer-ESWA

Appformer: A Novel Framework for Mobile App Usage Prediction Leveraging Progressive Multi-Modal Data Fusion and Feature Extraction  
[DOI Link](https://doi.org/10.1016/j.eswa.2024.125903)

### Data Setup
1. Download the Tsinghua dataset from [here](https://fi.ee.tsinghua.edu.cn/appusage/).
2. Place `App_Usage_Trace.txt` into the `data/Tsinghua_new` folder.
3. Place `base_poi.txt` into `data/Tsinghua_new/location_clustering`.
4. Run `POI_clustering_and_data_partitioning.py` to get POI clustering results and partition the training and test datasets.

### Model Training & Testing
1. After data processing, run `python main_informer.py` to train and test the models.
2. The weights for the best results are available in the `checkpoints/informer_TSapp_ftS_sl4_ll4_pl1_dm128_nh8_el2_dl2_df512_atfull_fc5_eblearned_dtFalse_mxTrue_test_0_` folder.
3. To test directly, comment out the `exp.train(setting)` line in `main_informer.py`.
