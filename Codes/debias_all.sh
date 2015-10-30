# .sh code for running the debiasing procedure on all of the questions in turn.

#python voronoi_binning.py

#python voronoi_binning.py

for var in 't01_smooth_or_features' 't06_odd' 't08_odd_feature' 't07_rounded' 't02_edgeon' \
 't05_bulge_prominence' 't03_bar' 't04_spiral' 't09_bulge_shape' 't10_arms_winding' 't11_arms_number'
do 
    sed -i "1s/.*/question = '$var'/" params.py # replace with each question in order.
    python debiasing.py
done