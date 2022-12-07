# Sequence-to-Function-Learning

This repo contains work on sequence-to-function learning based on language models.

The model was trained on 1000 PafA single amino acid mutants (obtained from "Revealing enzyme functional architecture via high-throughput microfluidic enzyme kinetics") and its prediction performance was also shown. 

## Dataset Contains

| Kinetic Params       |  Self-Attention     |     kmer-KNN     |
|    :---:             |     :---:           |     :---:        |
| kcat_cMUP            |     R =  ~ 0.64     |     R =  ~ 0.35  |
| KM_cMUP              |     R =  ?          |     R =  ?       |
| kcatOverKM_cMUP      |     R =  ?          |     R =  ?       |
| kcatOverKM_MeP       |     R =  ?          |     R =  ?       |
| kcatOverKM_MecMUP    |     R =  ?          |     R =  ?       |
| Ki_Pi                |     R =  ?          |     R =  ?       |



## Prediction Pipeline
<p align="center">
  <img width="900"  src="https://user-images.githubusercontent.com/47986787/205684697-7675f4fc-f821-4218-aede-8979aaac8789.png">
</p>


## Prediction Performance
<p align="center">
  <img width="600"  src="https://user-images.githubusercontent.com/47986787/206294274-6d0d8726-3178-4010-9702-df343fbfa40d.png">
</p>



### Make a $y$ vs. $\hat{y}$ plot.
```
from ZX01_PLOT import *
reg_scatter_distn_plot(y_pred_valid,
                        y_real_valid,
                        fig_size        =  (10,8),
                        marker_size     =  35,
                        fit_line_color  =  "brown",
                        distn_color_1   =  "gold",
                        distn_color_2   =  "lightpink",
                        # title         =  "Predictions vs. Actual Values\n R = " + \
                        #                         str(round(r_value,3)) + \
                        #                         ", Epoch: " + str(epoch+1) ,
                        title           =  "",
                        plot_title      =  "R = " + str(round(r_value,3)) + \
                                                  "\nEpoch: " + str(epoch+1) ,
                        x_label         =  "Actual Values",
                        y_label         =  "Predictions",
                        cmap            =  None,
                        cbaxes          =  (0.425, 0.055, 0.525, 0.015),
                        font_size       =  18,
                        result_folder   =  results_sub_folder,
                        file_name       =  output_file_header + "_VA_" + "epoch_" + str(epoch+1),
                        ) #For checking predictions fittings.


```


