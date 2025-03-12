# Betti Number and Bakry-Émery Curvature Calculator

A Python library for calculating the first Betti number of the cell complex arising from a graph by attaching 2-cells to every cycle of length at most five. Additionally, this library provides an efficient implementation of the Bakry-Émery curvature for graphs. For improved performance when analyzing large graphs where only the Bakry-Émery curvature is needed, set init_two_cells to False to skip the initialization of the cell complex.

This implementation is also utilized for computing the Betti numbers and Bakry-Émery curvature of the examples presented in our work *Betti number estimates for non-negatively curved graphs.*  

## Package Requirements  
To use this library, make sure you have the following packages installed:
* [NetworkX](https://networkx.org)
* [Numpy](https://numpy.org)

## Cite
If you use this code in your research, please considering cite our paper:
```
@article{???,
  title={Betti number estimates for non-negatively curved graphs},
  author={Münch, Florentin and Hehl, Moritz},
  journal={????},
  year={2025}
}
