# AutoGrav Research Paper

This directory contains the academic research paper documenting the AutoGrav library.

## Files

- **autograv_paper.typ** - Main Typst source file for the paper
- **autograv_paper.pdf** - Compiled PDF output
- **references.bib** - BibLaTeX bibliography file
- **README.md** - This file

## Paper Contents

The paper provides comprehensive documentation of the AutoGrav project:

### Sections

1. **Introduction** - Motivation and background on numerical relativity and automatic differentiation
2. **Background** - General relativity fundamentals and autodiff theory
3. **Methods** - Implementation details, architecture, and algorithms
4. **Results** - Numerical validation and performance benchmarks
5. **Discussion** - Advantages, limitations, and comparison to related work
6. **Conclusion** - Summary and future directions
7. **Appendix** - Installation and usage guide

### Key Results

- **Numerical accuracy**: 15+ decimal place precision on Schwarzschild metric
- **Kretschmann invariant**: Exact match between computed and analytical values
- **Validation**: All vacuum solution properties verified (zero Ricci/Einstein tensors)

## Compiling the Paper

To compile the paper from source:

```bash
# Navigate to paper directory
cd paper

# Compile with Typst
typst compile autograv_paper.typ

# Output: autograv_paper.pdf
```

### Requirements

- Typst 0.14.2 or later
- No external packages required (self-contained)

## Citation

If you use AutoGrav in your research, please cite:

```bibtex
@software{autograv2026,
  author = {Kataru, Baalateja},
  title = {AutoGrav: Automatic Differentiation for Numerical Relativity},
  year = {2026},
  url = {https://github.com/bkataru/autograv},
  version = {0.1.0}
}
```

## License

This paper and its source files are released under the MIT License, consistent with the AutoGrav library license.

## References

The paper includes 16 references covering:
- General relativity foundations (Einstein, Schwarzschild, Wald, Carroll)
- Automatic differentiation theory (Baydin et al., Griewank & Walther)
- Software frameworks (JAX, PyTorch, TensorFlow, NumPy)
- Related work (relativity-jax, EinFields, NRPy+)
- Blog post inspiration (Kataru 2024)

## Contact

For questions about the paper or library:
- Email: baalateja.k@gmail.com
- GitHub: https://github.com/bkataru/autograv
- PyPI: https://pypi.org/project/autograv/
