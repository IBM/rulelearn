# symlearn (v0.1)

This package contains a rule induction toolkit to generate readable and editable rules from data. The code was
originally released within the larger [AIX 360 package](https://github.com/Trusted-AI/AIX360) and is provided and 
extended here separately with less dependencies.

It contains the following components:

- Boolean Decision Rules via Column Generation (Light Edition) ([Dash et al., 2018](https://papers.nips.cc/paper/7716-boolean-decision-rules-via-column-generation))
- Generalized Linear Rule Models ([Wei et al., 2019](http://proceedings.mlr.press/v97/wei19a.html))
- Fast Effective Rule Induction (Ripper) ([William W Cohen, 1995](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.107.2612&rep=rep1&type=pdf))
- Relational Rule Network (R2N) ([Kusters et al., 2022](https://arxiv.org/abs/2201.06515))
- trxf - Technical Rule Interchange Format - Rule Set Interchange providing common evaluation tools and PMML export for 
rule sets.


### Installation

```
pip install -r requirements.txt
```
to be completed.


## Acknowledgements

AIX Rules is built with the help of several open source packages. All of these are listed in setup.py and some of these include:
* scikit-learn https://scikit-learn.org/stable/about.html

## License Information

Please view both the [LICENSE](https://github.com/vijay-arya/AIX360/blob/master/LICENSE) file and the folder [supplementary license](https://github.com/vijay-arya/AIX360/tree/master/supplementary%20license) present in the root directory for license information. 


