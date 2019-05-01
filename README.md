This is the code for the NIPS paper ['Natural-Parameter Networks: A Class of Probabilistic Neural Networks'](http://wanghao.in/paper/NIPS16_NPN.pdf). 

Note that this is the code for Gaussian NPN to run on the MNIST and Boston
Housing datasets. For Gamma NPN or Poisson NPN please go to the other repo.

To train the model, run the command:
'cd example'
to go to the directory of the entry point and run
'./run.sh'  or  'sh run.sh'

example/run.sh: entry point
mlp_bayes.m: core model code
default_mlp_bayes.m: initialize NPN

PyTorch version of NPN will be released soon.

#### Reference:
[Natural-Parameter Networks: A Class of Probabilistic Neural Networks](http://wanghao.in/paper/NIPS16_NPN.pdf)
```
@inproceedings{DBLP:conf/nips/WangSY16,
  author    = {Hao Wang and
               Xingjian Shi and
               Dit{-}Yan Yeung},
  title     = {Natural-Parameter Networks: {A} Class of Probabilistic Neural Networks},
  booktitle = {Advances in Neural Information Processing Systems 29: Annual Conference
               on Neural Information Processing Systems 2016, December 5-10, 2016,
               Barcelona, Spain},
  pages     = {118--126},
  year      = {2016}
}
