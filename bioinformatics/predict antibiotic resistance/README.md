# Predicting antibiotic resistance

Machine learning is useful for detecting patterns within data! A usecase is predicting antibiotic resistance from bacterial DNA sequences. 
By training a model on bacterial DNA sequences that is labeled with if the bacteria is resistant to antibiotics, we can learn what features indicate resistance and perhaps identify a specific gene encoding for antibiotic resistance. 

# Notebook Details
## Data
https://www.kaggle.com/code/drscarlat/predict-antibiotic-resistance-w-gene-sequence/input

The data is a synthetic dataset with a gene sequence for antibiotic resistance hidden within the bacteria sequence.
I suspect the data is for one organism but I don't know which (E.Coli, Klebsiella).
The antibiotic resistance is to one antibiotic class, but I don't know which (penicillinase?)

The columns are:
- features: gene sequence
- labels: resistance, with binary True/False value. 

Initial data exploration revealed that the dataset is relatively balanced and there is no missing data. 

## Preprocessing
Since we are working with strings, we need to tokenize the data. We approach the problem from two strategies:
- Tokenizing by the singular nucleotide bases themselves (alphabet =  A/C/G/T)
- Tokenizing by codons or groups of three nucleotide bases, leading to a larger alphabet that is more informative

## Model Building
This is a binary classification supervised learning problem where **order is important**. Shallow models cannot deal with ordered sequences, so we must build a model that can analyze such sequences! We decided to use the Sequential model class from Keras to handle this, and to use **1D convolutional layers and a bidirectional GRU layer** to capture information from sequences. Details on the specific layers are in later section. 
We will split data into training (72%), validation (18%), and a final testing (10%) data set. 


## Testing Results! 
AUC, appropriate for when there are roughly equal numbers of observations for each class
- Initial model using nucelotides: .755
- Model using codons: 0.989!!

I attempted to identify a specific gene encoding for antibiotic resistance SHAP, but that killed my local machine. 
I then tried a combination KerasRegressor, the PermutationImportance class, and ELI5 and identified the most important features which consisted of the 35th through 39th codon. Specifically, 'GTTGAA' showed up in 98% of the dataset. Honestly, this is not a good method for many reasons. For starters
- Difference types of encodings for different types of resistance methods
- indicators being in separate regions
- accetable variation within the regions
I'm just trying to find an answer for what the specific gene encoding is for antibiotic resistance is. 
Model wil work for binary prediction though. 


# TODO: 
- determine the specific gene encoding is for antibiotic resistance is
- literature review
    - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10044642/
    - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9491192/


# Layers:
The rationale behind this order is to start with layers that capture local patterns and gradually move towards layers that capture more global and sequential dependencies. The combination of convolutional and recurrent layers allows the model to learn hierarchical features and dependencies in the data. Additionally, the use of dropout layers helps prevent overfitting. This architecture is often effective for tasks like sequential data classification, such as sentiment analysis or text classification.

1. Embedding Layer: used at the beginning to convert categorical data, such as word or integer indices, into continuous vector representations of fixed size

2. Convolutional Layer: effective at capturing local patterns and feature hierarchies in sequential data.
- Filters: a spatial window containing a set of learnable weights that moves across the input data. The number of filters determines how many different features the convolutional layer can learn. Higher numbers allow the model to learn more complex features.
- Window Size: The window size is the size of the filter (spatial extent) that moves across the input data. We specify a larger window size for the first convolutional layer to capture more global patterns, but this may lead to a larger number of parameters and increased computation.
- Activation:
    - Relu: $f(x)=max(0,x)$, Allows for non-linearity

3. MaxPooling Layer: downsample the output of the convolutional layers by taking the maximum value over a window of specified size, retaining the most important information while reducing dimensionality, makes model more computationally efficient/manageable and reducing the risk of overfitting.
- The primary objective of max pooling is to reduce the amount of information in an image while maintaining the essential features necessary for accurate image recognition. This process helps to make the detection of features in input data invariant to scale and orientation changes and also aids in preventing overfitting.
- https://deepai.org/machine-learning-glossary-and-terms/max-pooling


4. Dropout Layers: introduce reularization and may prevent overfitting by randomly dropping a fraction of units during training. added after convolutional layers and before the recurrent layer
- Randomly abandon nodes each training pass. Result is weights for nearby nodes are not adjusted at the same time, and the decoupling makes the weight changes not correlate to each other and ignore more noise, reducing bias and overfitting
- rate guide: input (0.8-0.9), hidden (0.5-0.8), output (1)
- Sparse activation: less than half the neurons have a non-zero output
- disadvantages: slow convergence and risk of omitting data trends
- https://nnart.org/should-you-use-dropout/

5. Convolutional Layer 2
- Second convolutional layer with a smaller window size, focus on local patterns which is useful for capturing fine-grained details

6. Dropout Layer 2

5. Bidirectional GRU Layer: Recurrent layers, such as GRU, are added after convolutional layers to capture sequential dependencies in the data.
- GRU (Gated Recurrent Unit) learns long-term dependencies in sequential data. It has a gating mechanism to control the flow of information, similar to Long Short-Term Memory (LSTM) units but with a simpler architecture
- bidirectional creates two separate GRU layers to capture bidirectional temporal dependencies, one processing the input from the beginning to end and the other processing it from end to beginning, 
- chosen parameter values aim to balance model complexity and prevent overfitting during training
- https://paperswithcode.com/method/bigru


6. Dense Layer: for the final classification!!
- takes the features learned by the preceding layers and produces the output for binary classification using the sigmoid activation function.




