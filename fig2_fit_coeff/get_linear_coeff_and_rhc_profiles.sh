#!/bin/bash
# Run the python scripts to get the linear coefficient and critical RH profiles

basedir=/disca/share/ql260/era5

# ================================================== #
#  Interploate from 1x1 to T42 and T85 resolutions
# ================================================== #
dt_dir=$basedir/data_2017

# Interpolate monthly data (1x1) to T21, T42, T63 and T85 resolutions
# Resolutions refer to: 
# https://climatedataguide.ucar.edu/climate-model-evaluation/common-spectral-model-grid-resolutions

for m in {01..12}
do
    echo Month $m is being interpolated...

    # default 1x1 data
    src_fn=$dt_dir/era5_cld_rh_2017_"$m".nc

    for res in r128x64 r256x128   # r64x32 r192x64
    do
        # echo $res
        dst_fn=$dt_dir/era5_cld_rh_2017_"$m"_"$res".nc
        [[ ! -f $dst_fn ]] && cdo remapbil,"$res" $src_fn $dst_fn
    done
done

# ================================================== #
#  Interploate from 0.5x0.5 to 0.75x0.75 resolution
#  Each day is a file
# ================================================== # 
dt_dir=$basedir/data_2017_high_res
res=r480x240

for m in {01..12}
do
    echo Month $m

    orig_dir=$dt_dir/$m
    out_dir=$dt_dir/"$m"_"$res"
    [[ ! -d $out_dir ]] && mkdir $out_dir

    for d in {01..31}
    do
        # echo $d
        src_fn=$orig_dir/era5_cld_rh_2017_"$m"_"$d"_0.50.nc
        dst_fn=$out_dir/era5_cld_rh_2017_"$m"_"$d"_"$res".nc
        [[ -f $src_fn ]] && [[ ! -f $dst_fn ]] && cdo remapbil,"$res" $src_fn $dst_fn
    done
done

# ===========================================================#
# Run python scripts to get the linear coefficient profiles
# ===========================================================#

for varnm in coeff_a rhc
do
    # For low res
    nohup python -u linear_coeff_and_rhc_profile.py $varnm &> tmp."$varnm".txt &

    # For high res
    nohup python -u linear_coeff_and_rhc_profile_high_res.py $varnm &> tmp."$varnm"_high_res.txt &
done
