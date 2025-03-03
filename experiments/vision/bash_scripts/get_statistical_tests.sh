#!/bin/bash

## Run first the statistical analysis
python3 ./../run_statistical_analysis.py --methods shearletX_saliency,shearletX --hypothesis "greater" --metric cp_pixel --apply_filter
python3 ./../run_statistical_analysis.py --methods shearletX_saliency,shearletX_uniform --hypothesis "greater" --metric cp_pixel --apply_filter
python3 ./../run_statistical_analysis.py --methods sharletX_saliency,shearletX --hypothesis "greater" --metric cp_l1 --apply_filter
python3 ./../run_statistical_analysis.py --methods shearletX_saliency,shearletX_uniform --hypothesis "greater" --metric cp_l1 --apply_filter

python3 ./../run_statistical_analysis.py --methods waveletX_saliency,waveletX --hypothesis "greater" --metric cp_pixel --apply_filter
python3 ./../run_statistical_analysis.py --methods waveletX_saliency,waveletX_uniform --hypothesis "greater" --metric cp_pixel --apply_filter
python3 ./../run_statistical_analysis.py --methods waveletX_saliency,waveletX --hypothesis "greater" --metric cp_l1 --apply_filter
python3 ./../run_statistical_analysis.py --methods waveletX_saliency,waveletX_uniform --hypothesis "greater" --metric cp_l1 --apply_filter

python3 ./../run_statistical_analysis.py --methods pixelRDE_saliency,pixelRDE --hypothesis "greater" --metric cp_pixel --apply_filter
python3 ./../run_statistical_analysis.py --methods pixelRDE_saliency,pixelRDE_uniform --hypothesis "greater" --metric cp_pixel --apply_filter
python3 ./../run_statistical_analysis.py --methods pixelRDE_saliency,pixelRDE --hypothesis "greater" --metric cp_l1 --apply_filter
python3 ./../run_statistical_analysis.py --methods pixelRDE_saliency,pixelRDE_uniform --hypothesis "greater" --metric cp_l1 --apply_filter