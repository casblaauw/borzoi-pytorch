# borzoi-pytorch
The [Borzoi model](https://www.biorxiv.org/content/10.1101/2023.08.30.555582v1) from Calico, but ported to Pytorch! Original implementation and weights are [here](https://github.com/calico/borzoi).  
We show that the Pytorch version produces the same predictions as the original implementation for all tested notebook examples.  

We include a weight conversion script that ports Borzoi weights from TF-keras to Pytorch.

## Installation
1. Clone the repo and `cd`
2. `pip install -e .`

## Todo
- [ ] Test the model on more sequences to ensure equivalence to the original implementation.  
- [ ] Support reverse complement and shift augmentation  
- [ ] ...

## Misc.  
Enabling tf32 or bf16 and/or compiling with Pytorch 2.0 leads to a speed up (compared to the plain PT version).

## References
<a id="1">[1]</a> 
Predicting RNA-seq coverage from DNA sequence as a unifying model of gene regulation  
Johannes Linder, Divyanshi Srivastava, Han Yuan, Vikram Agarwal, David R. Kelley  
bioRxiv 2023.08.30.555582; doi: [https://doi.org/10.1101/2023.08.30.555582](https://www.biorxiv.org/content/10.1101/2023.08.30.555582v1)  
<a id="2">[2]</a> 
[enformer-pytorch github](https://github.com/lucidrains/enformer-pytorch/),
Phil Wang *lucidrains*
