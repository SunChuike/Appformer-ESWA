# Appformer-ESWA

**Appformer: A Novel Framework for Mobile App Usage Prediction Leveraging Progressive Multi-Modal Data Fusion and Feature Extraction (ESWA 2025)**  
*Chuike Sun, Junzhou Chen, Yue Zhao, Hao Han, Ruihai Jing, Guang Tan, Di Wu*  
[DOI Link](https://doi.org/10.1016/j.eswa.2024.125903)  

## **Abstract**  
This article presents Appformer, a novel mobile application prediction framework inspired by the efficiency
of Transformer-like architectures in processing sequential data through self-attention mechanisms. Combining
a Multi-Modal Data Progressive Fusion Module with a sophisticated Feature Extraction Module, Appformer
leverages the synergies of multi-modal data fusion and data mining techniques while maintaining user privacy.
The framework employs Points of Interest (POIs) associated with base stations, optimizing them through
comprehensive comparative experiments to identify the most effective clustering method. These refined inputs
are seamlessly integrated into the initial phases of cross-modal data fusion, where temporal units are encoded
via word embeddings and subsequently merged in later stages. The Feature Extraction Module, employing
Transformer-like architectures specialized for time series analysis, adeptly distils comprehensive features.
It meticulously fine-tunes the outputs from the fusion module, facilitating the extraction of high-caliber,
multi-modal features, thus guaranteeing a robust and efficient extraction process. Extensive experimental
validation confirms Appformer’s effectiveness, attaining state-of-the-art (SOTA) metrics in mobile app usage
prediction, thereby signifying a notable progression in this field. The implementation is available at https:
//github.com/SunChuike/Appformer-ESWA.

---

## **Data Setup**  
1. Download the **Tsinghua dataset** from [here](https://fi.ee.tsinghua.edu.cn/appusage/).  
2. Place `App_Usage_Trace.txt` into `data/Tsinghua_new/`.  
3. Place `base_poi.txt` into `data/Tsinghua_new/location_clustering/`.  
4. Run `POI_clustering_and_data_partitioning.py` to perform POI clustering and dataset partitioning.  

---

## **Model Training & Testing**  
1. **Train & Test**: Run `python main_informer.py` after data processing.  
2. **Pretrained Weights**: Available in `checkpoints/informer_TSapp_ftS_sl4_ll4_pl1_dm128_nh8_el2_dl2_df512_atfull_fc5_eblearned_dtFalse_mxTrue_test_0_/`.  
3. **Direct Testing**: Comment out `exp.train(setting)` in `main_informer.py`.  

---

## **Citation**  
If you use Appformer, please cite:  

```bibtex
@article{SUN2025125903,
  title = {Appformer: A novel framework for mobile app usage prediction leveraging progressive multi-modal data fusion and feature extraction},
  journal = {Expert Systems with Applications},
  volume = {265},
  pages = {125903},
  year = {2025},
  doi = {https://doi.org/10.1016/j.eswa.2024.125903},
  author = {Chuike Sun et al.},
  keywords = {App usage prediction, Transformer, Multi-modal data fusion, Feature extraction, Data mining}
}
```

---

## **Acknowledgment**  
This work builds upon *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting*  
[DOI Link](https://doi.org/10.1609/aaai.v35i12.17325). Thanks to the authors for their outstanding work.  

## **Contact**  
📧 Email: **sunchk3@mail2.sysu.edu.cn**  


