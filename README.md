# Age-Prediction-based-on-Iris-Biometric-Data
This project implemented an age prediction system based on deep learning methods and evaluated it with the provided dataset.

1. Age prediction based on iris biometric data

The main purpose of the age prediction systems is to determine the age group (group1: <25, group2: 25- 60, and group3: >60 ) of the person in a query image. The prediction is done by evaluating the semantic contents of the query image. However, there is difficulty in revealing the semantics of images due to the semantic gap. In order to overcome this difficulty, images are described as feature vectors which are higher-level representations than collection of numbers. With these feature vectors, age prediction can be formulated as a learning problem to match an image representation with the age group of the person in the image. Hence, in this assignment, constructed a fully-connected network with the rectified linear unit (ReLU) as a nonlinearity function between layers and trained it with an RMSprop optimizer using the provided feature vectors. While training the network, used the softmax(cross-entropy loss) function to minimize the difference between the actual age group and the estimated one.

2. Dataset and feature extraction

The commercially available data Set 2 (DS2) of the BioSecure Multimodal Database (BMDB) is utilized for this project. Four eye images (two left and two right) were acquired in two different sessions with a resolution of 640*480 pixels. The 200 subjects providing the samples contained in this database are within the age range of 18-73. The training and the testing sets were formed to be person-disjoint sets. Approximately 72% of the subjects in each age group are used for training and the remaining subjects are used as a testing set.

3. Age group prediction

Implemented the aforementioned age prediction system using fully connected neural networks with four different numbers of hidden layers (0,1,2,3).

After implementation, evaluated our solution with different configurations as mentioned before using the provided training set.

Finally, decided on the most successful configuration based on our experiments and then evaluate the error rate with the testing set.

An important trick about the implementation is the saving model at intermediate epochs. While training the network, the dataset is usually divided into mini-batches. After computing the loss for each batch, the parameters of the network are updated. One pass of the whole training set is called an epoch. In order to get a good fit to data, the number of epochs that the network trained was determined. This is done with the help of loss history plots that show the loss computed using training and validation sets for each epoch. After examining the plot, one decided on the number of epochs. In order not to retrain the network, saved model and optimizer parameters at some epochs (i.e. at each 5 epochs). Another important trick is setting a seed for random number generators. This allows the experiments to be repeatable since whenever it is run, it starts from the same random parameters.

For more detailed information, please check the pdf project file => [Project_Description.pdf](https://github.com/ilbey/Age-Prediction-based-on-Iris-Biometric-Data/files/9104799/Project_Description.pdf)

For Project Results and Report, please check the pdf file => [Project_Report.pdf](https://github.com/ilbey/Age-Prediction-based-on-Iris-Biometric-Data/files/9106444/Project_Report.pdf)
