# Graph-Dynamo

Graph-Dynamo is a computational framework developed to model cellular processes by leveraging the dynamical systems theory, addressing a significant bottleneck in genome-wide modeling due to the absence of sufficient quantitative data. This innovative framework is capable of extracting dynamical information from high-throughput snapshot single cell data, courtesy of advances in single-cell techniques.

## Features

1. **Tangent Space Projection (TSP)**:
   Graph-Dynamo introduces a graph-based machine learning procedure that constrains RNA velocity vectors to lie in the tangent space of the low-dimensional manifold represented by the single cell expression data. Unlike the traditional cosine correlation kernel, TSP maintains the vector's magnitude information while transitioning between different data manifold representations.

2. **Data-Driven Graph Fokker-Planck (FPE) Equation Formalism**:
   The framework incorporates a data-driven graph FPE equation formalism to model the cellular state transition dynamics as a convection-diffusion process on a data-formed graph network. This formalism ensures invariance under representation transformation while preserving the topological and dynamical properties of system dynamics.

3. **Dynamo Framework Integration**:
   Building upon our previously developed dynamo framework, Graph-Dynamo reconstructs genome-wide gene regulation relations from single-cell expression states and RNA velocity data derived from either splicing or metabolic labeling.

## Installation

```bash
pip install graph-dynamo
```

## Quick Start

After installation, you can easily start working with the `graph-dynamo` package:

```python
import graph_dynamo as gd

# Assuming data is loaded as data
tsp_result = gd.tangent_space_projection(data)
fpe_result = gd.graph_fokker_planck_equation(data)
```

## Documentation

The detailed documentation can be found at [Documentation Link](https://link_to_documentation)

## Usage Examples

```python
# More elaborate examples of how to use graph-dynamo
```

## Performance

Numerical tests on both synthetic data and experimental scRNA-seq data underline the capability of the Graph-Dynamo framework. By utilizing the graph TSP/FPE formalism constructed from snapshot single cell data, it successfully recapitulates system dynamics, marking a significant stride in single-cell studies and systems biology.

## Citation

If you utilize Graph-Dynamo in your research, please consider citing our paper:

```bibtex
@article{author2023graph,
  title={Graph-Dynamo: Learning stochastic cellular state transition dynamics from single cell data},
  author={Yan Zhang, Xiaojie Qiu, Ke Ni, Jonathan Weissman, Ivet Bahar, Jianhua Xing},
  year={2023},
  publisher={https://www.biorxiv.org/content/10.1101/2023.09.24.559170v1}
}
```

## Support

For support and further inquiries, feel free to reach out at [xing1@pitt.edu](mailto:xing1@pitt.edu).

## License

Graph-Dynamo is released under the MIT License. See the LICENSE file for further details.

---

This README follows the standard structure, and is intended to provide a basic understanding of the Graph-Dynamo framework, its capabilities, installation instructions, and how to get started quickly. For a more comprehensive insight, please refer to the documentation and the referenced paper.