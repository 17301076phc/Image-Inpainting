# Image-Inpainting
finding dominant offsets for image inpainting in Chinese landscape painting

## Steps
1. Finding similar patches and obtaining their offsets 
2. Finding K dominant offsets through computation of statistics by a 2-D Histogram
3. Finding the optimal labelling for the unknown pixels by “Fast Approximate Energy Minimization via Graph Cuts”
4. Completing the image based on the labels found in the previous step

## Dependence
- Python Version – 3.7
- OpenCV Version – 3.4.3
- PyMaxflow - 1.3
- Numpy - 1.19
- Matplotlib - 3.3.4
- Scipy -1.8

## Usage
change the path code at line 243/244 in find.py file

run python find.py

## Related work
1. He, Kaiming, and Jian Sun.: Statistics of patch offsets for image completion. 
2. He, K., Sun, J.: Computing nearest-neighbor fields via propagation-assisted kdtrees.
3. Boykov, Y., Veksler, O., Zabih, R., Fast approximate energy minimization via graph cuts. 
