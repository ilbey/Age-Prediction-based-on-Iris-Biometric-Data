# Age-Prediction-based-on-Iris-Biometric-Data
In this project you are required to implement an age prediction system based on deep learning methods, and to evaluate it with the provided dataset. 

**1. Age prediction based on iris biometric data**

The main purpose of the age prediction systems is to determine age group (group1: <25, group2: 25-
60 and group3: >60 ) of the person in a query image. The prediction is done by evaluating semantic 
contents of the query image. However, there is a diffculty in revealing the semantics of images due 
to the semantic gap. In order to overcome this diffculty, images are described as feature vectors 
which are higher level representations than collection of numbers. 
With these feature vectors, age prediction can be formulated as a learning problem to match an 
image representation with the age group of person in the image. Hence, in this assignment you are 
required to construct a fully-connected network with rectified linear unit (ReLU) as nonlinearity 
function between layers and train it with RMSprop optimizer using the provided feature vectors.
While training the network you are required to use softmax(cross-entropy loss) function to minimize 
the difference between actual age group and the estimated one.

**2. Dataset and feature extraction**

The commercially available data Set 2 (DS2) of the BioSecure Multimodal Database (BMDB) is utilised 
for this project. Four eye images (two left and two right) were acquired in two different sessions with a resolution of 640*480 pixels. The 200 subjects providing the samples contained in this database are within the age range of 18-73. The training and the testing sets were formed to be person-disjoint sets. Approximately 72% of the subjects in each age group are used for training and the remaining subjects used as a testing set.

**3. Age group prediction**

You are required to implement the aforementioned age prediction system using fully connected 
neural networks with four different number of hidden layers (0,1,2,3). 

After implementation, you should evaluate your solution with different configurations as mentioned before using the provided training set. 

Finally, you will decide on the most successful configuration based on your experiments and then 
evaluate the error rate with the testing set. 

An important hint about the implementation is saving model at intermediate epochs. While training 
the network, dataset is usually divided into mini-batches. After computing the loss for each batch, parameters of the network are updated. One pass of whole training set is called an epoch. In order to get a good fit to data, the number of epochs that the network will be trained should be determined. This can be done with the help of loss history plots that shows the loss computed using training and validation sets for each epoch. After examining the plot, one can decide on the number of epochs. In order not to retrain the network, you can save model and optimizer parameters at some epochs (i.e. at each 5 epochs). Another important hint is setting a seed for random number generators. This allows the experiments to be repeatable since whenever it is run, it starts from the same random parameters.

More detailed information, please check the pdf project file => [Project_Description.pdf](https://github.com/ilbey/Age-Prediction-based-on-Iris-Biometric-Data/files/9104799/Project_Description.pdf)


For Project Results and Report, please check the pdf file => IRISDeep_Report.pdf


