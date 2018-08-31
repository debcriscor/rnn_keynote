# Application of Time Series to Recurrent Neural Networks

Hands-on workshop presented in the 
[Colloquium on “Irregular engineering vibrations and oscillations”](https://cgi.tu-harburg.de/~dynwww/cgi-bin/colloquium-2018/)


## Jupyter Notebooks

To run the Jupyter Notebooks provided in this repository, you must install the proper envinroment


## Installing Environment

You need to install Python 3.6, Tensorflow and Keras.

### Install Python 3.6

  For Python, install the latest version full version of [Anaconda](https://www.anaconda.com/download/). 
  To do so, follow [this](https://conda.io/docs/user-guide/install/index.html#) instructions.
  
### Create a virtual environment with [conda](https://conda.io/docs/user-guide/tasks/manage-environments.html):

```
conda create -n rnn_keynote python=3.6
```

Then you can activate the virtual environment:

```
source activate rnn_keynote
```

When you have done your work on the virtual environment, do the following to deactivate it:

```
source deactivate rnn_keynote
```

### Install the Python requirements

We need to install numpy, scikit-learn, matplotlib, jupyter, notebook, scipy and pandas

```
conda install numpy scikit-learn matplotlib jupyter notebook scipy pandas
```
### Install the Tensorflow

Without GPU:
```
conda install -c conda-forge tensorflow
```
With GPU:
```
conda install -c anaconda tensorflow-gpu
```
### Install Keras

```
conda install -c conda-forge keras 
```

## Cloning the repository

Install git by using [this](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) tutorial. 
To download the repository without install git clock in the green botton **clone or download** and select the option download zip.
