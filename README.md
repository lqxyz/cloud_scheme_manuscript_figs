## Scripts to plot figures for the cloud scheme manuscript

#### Datasets
* The reanalysis data sets used in this study are [ERA-interim](https://www.ecmwf.int/en/forecasts/datasets/archive-datasets/reanalysis-datasets/era-interim) and [ERA5](https://cds.climate.copernicus.eu/cdsapp\#!/home). 
* The observed cloud fraction products used for analysis include [ISCCP-H series](https://www.ncdc.noaa.gov/cdr/atmospheric/cloud-properties-isccp), [CALIPSO-GOCCP](https://climserv.ipsl.polytechnique.fr/cfmip-obs/Calipso_goccp.html) and [CloudSat](http://www.cloudsat.cira.colostate.edu/data-products/level-2b/2b-cwc-ro). The scripts to process the CloudSat dataset are available at [cloudsat_cloud_water_path](https://github.com/lqxyz/cloudsat_cloud_water_path) repository.
* The observed cloud radiative effect products are from [CERES-EBAF](https://ceres.larc.nasa.gov/compare_products.php). 
* The CMIP5 ouputs are obtained from the [Centre for Environmental Data Analysis](https://www.ceda.ac.uk). 
* The Isca model outputs produced for this study are available on Zenodo:[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4573610.svg)](https://doi.org/10.5281/zenodo.4573610)
* The input files for simulations are generated based on standard [AMIP SST and sea ice boundary condition dataset](https://pcmdi.llnl.gov/mips/amip/SST.html), and the scripts used to get these files are archived at [input4MIPs](https://github.com/lqxyz/input4MIPs) repository. 

#### Required packages (uncommon)
* `proplot`: https://proplot.readthedocs.io/en/latest/index.html 
* `cmaps`: https://github.com/hhuangwx/cmaps
* `taylorDiagram`: https://gist.github.com/ycopin/3342888#file-taylordiagram-py
* `isca` package: Interpolate from sigma to pressure levels. Install following the instruction on https://execlim.github.io/Isca/latest/html/install.html

#### Figures and tables (Jupyter notebooks)
* Figure 2: [`figure2_fit_coeff.ipynb`](https://github.com/lqxyz/cloud_scheme_manuscript_figs/blob/main/fig2_fit_coeff/figure2_fit_coeff.ipynb). If Github can not load this notebook, you can try this [link](https://nbviewer.jupyter.org/github/lqxyz/cloud_scheme_manuscript_figs/blob/main/fig2_fit_coeff/figure2_fit_coeff.ipynb) on [nbviewer](https://nbviewer.jupyter.org).
* Figure 3: [`figure3_cmp_offline.ipynb`](https://github.com/lqxyz/cloud_scheme_manuscript_figs/blob/main/fig3_cmp_offline/figure3_cmp_offline.ipynb), or try this [link](https://nbviewer.jupyter.org/github/lqxyz/cloud_scheme_manuscript_figs/blob/main/fig3_cmp_offline/figure3_cmp_offline.ipynb).
* Figure 4: [`figure4_qv_threshold.ipynb`](https://github.com/lqxyz/cloud_scheme_manuscript_figs/blob/main/figure4_qv_threshold.ipynb), or try this [link](https://nbviewer.jupyter.org/github/lqxyz/cloud_scheme_manuscript_figs/blob/main/figure4_qv_threshold.ipynb).
* Figure 5: [`figure5_low_cld_proxy.ipynb`](https://github.com/lqxyz/cloud_scheme_manuscript_figs/blob/main/figure5_low_cld_proxy.ipynb), or try this [link](https://nbviewer.jupyter.org/github/lqxyz/cloud_scheme_manuscript_figs/blob/main/figure5_low_cld_proxy.ipynb).
* Figures 6 to 18 and Table 3: [`figures_6_18.ipynb`](https://github.com/lqxyz/cloud_scheme_manuscript_figs/blob/main/figures_6_18.ipynb), or try this [link](https://nbviewer.jupyter.org/github/lqxyz/cloud_scheme_manuscript_figs/blob/main/figures_6_18.ipynb).
* Table 4: [`table4_parameter_sensitivity.ipynb`](https://github.com/lqxyz/cloud_scheme_manuscript_figs/blob/main/table4_parameter_sensitivity.ipynb), or try this [link](https://nbviewer.jupyter.org/github/lqxyz/cloud_scheme_manuscript_figs/blob/main/table4_parameter_sensitivity.ipynb).

