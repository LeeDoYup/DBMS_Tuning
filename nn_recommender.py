import tensorflow as tf
import numpy as np
from matrix import Matrix
from preprocessing import get_shuffle_indices
import pdb
from util import read_wine_quality
import itertools
INFINITE = 9999999999


FIRST_NEURON = 0
SECOND_NEURON = 1
TRAIN_LEARNING_RATE = 2
GENERATOR_LEARNING_RATE = 3
GENERATOR_ITERATION = 4
BATCH_SIZE = 5
EXPERIMENT_ITERATION = 6

class Config():
    """Experiment Setting."""
    first_neuron = range(50,501,50)
    second_neuron = range(50,501,50)
    train_learning_rate = [0.001]
    generator_learning_rate = [0.01]
    generator_iteration = [100]
    batch_size = [100]
    experiment_iteration = 5




def main():

    #------------------------
    # Load raw value of x,y 
    #------------------------
    
    y_data, x_data = read_wine_quality(colors=['red','white'])
    y_data = y_data.reshape([-1, 1]) # expand dimension
    
    pdb.set_trace()

    normalizer = Normalizer()

    # Normalize data.
    x_data, y_data = normalizer.normalize_data(x_data,y_data)

    # Choose best wine feature as initial point.
    x_initial = np.expand_dims(x_data[np.argmax(y_data)],axis=0) 
    
    # Split data.
    x_train,y_train,x_valid,y_valid,x_test,y_test = split_data(x_data,y_data,[0.8,0.1,0.1])
    
    # Execute the experiment.
    config = Config()
    experiment(config,x_train,y_train,x_valid,y_valid,normalizer,x_initial)
    
    return



def experiment(config,x_train,y_train,x_valid,y_valid,normalizer,x_initial):
    """
    Run experiment for given settings.
    Args :
        config : Experiment configuration.
        x_train,y_train : Training data.
        x_valid,y_valid : Validation data.
        normalizer : normalizer for denormalize after generate input feature.
        x_initial : Initial input feature for data generation. 
    Return : 
        best_config : Best configuration. 
    """
    
    num_features_x = x_train.shape[1]
    num_features_y = y_train.shape[1]

    config_list = config_to_list(config)

    
    min_valid_loss_mean = INFINITE
    max_quality_mean = 0

    for setting in config_list:
        
        f = open("nn_recommender_result.txt","a")
        
        train_loss_list = []
        valid_loss_list = []
        quality_list = []
        
        for j in range(config.experiment_iteration):
            nn_recommender = NNRecommender(x_initial,[setting[FIRST_NEURON],setting[SECOND_NEURON]],num_features_x, num_features_y,
                                           setting[TRAIN_LEARNING_RATE],setting[GENERATOR_LEARNING_RATE])
            train_loss, valid_loss = nn_recommender.fit(x_train,y_train,x_valid,y_valid,setting[BATCH_SIZE])
            maximized_quality = nn_recommender.generate(setting[GENERATOR_ITERATION],normalizer)
            
            train_loss_list.append(train_loss)
            valid_loss_list.append(valid_loss)
            quality_list.append(maximized_quality)

            tf.reset_default_graph()

        train_loss_mean = np.mean(train_loss_list)
        train_loss_std = np.std(train_loss_list)
        valid_loss_mean = np.mean(valid_loss_list)
        valid_loss_std = np.std(valid_loss_list)
        quality_mean = np.mean(quality_list)
        quality_std = np.std(quality_list)
        
        hyperparameter_label = " first_neuron    |  second_neuron | train_learning_rate | generator_learning_rate | generator_iteration |  batch_size | \n"
        hyperparameter_value = "{:^17}|{:^16}|{:^21}|{:^25}|{:^21}|{:^13}| \n".format(setting[FIRST_NEURON],setting[SECOND_NEURON],
                setting[TRAIN_LEARNING_RATE],setting[GENERATOR_LEARNING_RATE],setting[GENERATOR_ITERATION],setting[BATCH_SIZE])
       
        result_label =         " train_loss mean | train_loss std |   valid_loss mean   |      valid_loss std     |    quality mean     | quality std | \n"
        result_value = "{:^17.3f}|{:^16.3f}|{:^21.3f}|{:^25.3f}|{:^21.3f}|{:^13.3f}| \n".format(train_loss_mean,train_loss_std,valid_loss_mean,valid_loss_std,quality_mean,quality_std)
        
        if(valid_loss_mean < min_valid_loss_mean):
            min_valid_loss_mean = valid_loss_mean
            best_loss_setting = setting
        if(quality_mean > max_quality_mean):
            max_quality_mean = quality_mean
            best_quality_setting = setting

        f.write(hyperparameter_label)
        f.write(hyperparameter_value)
        f.write(result_label)
        f.write(result_value)
            
        f.close()

    
    f = open("nn_recommender_result.txt","a")

    f.write("\n\n\nbest_loss_setting\n")
    f.write(hyperparameter_label)
    best_loss_setting_value = "{:^17}|{:^16}|{:^21}|{:^25}|{:^21}|{:^13}| \n".format(best_loss_setting[FIRST_NEURON],
            best_loss_setting[SECOND_NEURON],best_loss_setting[TRAIN_LEARNING_RATE],best_loss_setting[GENERATOR_LEARNING_RATE],
            best_loss_setting[GENERATOR_ITERATION],best_loss_setting[BATCH_SIZE])
    best_loss_setting_value = "{:^17}|{:^16}|{:^21}|{:^25}|{:^21}|{:^13}| \n".format(best_quality_setting[FIRST_NEURON],
            best_quality_setting[SECOND_NEURON],best_quality_setting[TRAIN_LEARNING_RATE],best_quality_setting[GENERATOR_LEARNING_RATE],
            best_quality_setting[GENERATOR_ITERATION],best_quality_setting[BATCH_SIZE])
    f.write(best_loss_setting_value)
    f.write("\n\n\nbest_quality_loss_setting\n")
    f.write(hyperparameter_label)
    f.write(best_loss_setting_value)
    
    f.close()

        

    return



def config_to_list(config):
    
    return list(itertools.product(config.first_neuron,config.second_neuron,config.train_learning_rate,
                                  config.generator_learning_rate, config.generator_iteration, config.batch_size))



class NNRecommender:
    def inference(self,generative=False):
        
        with tf.variable_scope("model",reuse=generative):
            # Declare variables.
            W_1 = tf.get_variable("W_1",initializer=tf.truncated_normal([self.num_features_x,self.num_hidden[0]], stddev=0.01))
            b_1 = tf.get_variable("b_1",initializer=tf.truncated_normal([self.num_hidden[0]],stddev=0.01))
            W_2 = tf.get_variable("W_2",initializer=tf.truncated_normal([self.num_hidden[0],self.num_hidden[1]], stddev=0.01))
            b_2 = tf.get_variable("b_2",initializer=tf.truncated_normal([self.num_hidden[1]], stddev=0.01))
            W_f = tf.get_variable("W_f",initializer=tf.truncated_normal([self.num_hidden[1],self.num_features_y], stddev=0.01))
            b_f = tf.get_variable("b_f",initializer=tf.truncated_normal([1],stddev=0.01))

        
        if generative:
            X = self.generated_x
        else:
            X = self.x

        # Build model.
        hidden1 = tf.sigmoid(tf.matmul(X,W_1)+b_1)
        hidden2 = tf.sigmoid(tf.matmul(hidden1,W_2)+b_2)
        y_hat = tf.matmul(hidden2,W_f)+b_f
        
        if generative:
            loss = -y_hat
        else:
            loss = tf.reduce_mean(tf.squared_difference(y_hat,self.y))
       
        return y_hat, loss


    
    
    
    def __init__(self,initial,num_hidden,num_features_x,num_features_y,lr,g_lr):
        
        self.num_hidden = num_hidden
        self.num_features_x = num_features_x
        self.num_features_y = num_features_y
        self.learning_rate = lr

        self.sess = tf.Session()
        # Make placeholder for x and y.
        self.x = tf.placeholder(tf.float32, [None,num_features_x])
        self.y = tf.placeholder(tf.float32, [None,num_features_y])
        
        # Do inference for discriminal model.
        self.y_hat_train, self.loss_train = self.inference()
        optimizer_train = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.step_train = optimizer_train.minimize(self.loss_train)
        
        # Do inference for generative model.
        self.generated_x = tf.get_variable("generated_x",initializer=tf.constant(initial,dtype=tf.float32),dtype=tf.float32)
        self.y_hat_generate, self.loss_generate = self.inference(generative=True)
        optimizer_generate = tf.train.GradientDescentOptimizer(learning_rate=g_lr)
        self.step_generate = optimizer_generate.minimize(self.loss_generate,var_list=[self.generated_x])
        


        # Initialize all variables.
        init = tf.initialize_all_variables()
        self.sess.run(init)

    def fit(self,x_train,y_train,x_valid,y_valid,batch_size):
        
        num_data = np.shape(x_train)[0]
        num_batch = (num_data + batch_size-1)//batch_size
        max_patience = 20
        current_patience = 0
        min_valid_loss = INFINITE

        while True:
            
            # Shuffle train data.
            random_index = np.random.permutation(num_data)
            x_train = x_train[random_index]
            y_train = y_train[random_index]

            # Calculate train loss.
            loss_list = []
            for i in range(num_batch):
                next_batch = {self.x:x_train[i*batch_size:(i+1)*batch_size],
                              self.y:y_train[i*batch_size:(i+1)*batch_size]}
                loss_value,y_hat_value,_ = self.sess.run((self.loss_train,self.y_hat_train,self.step_train),feed_dict=next_batch)
                loss_list.append(loss_value)
            print "train loss : ", np.mean(loss_list)
            print "y_hat : ", y_hat_value[0]
            print "y_train : ", y_train[0]

            # Calculate valid loss.
            valid_feed_dict = {self.x:x_valid,self.y:y_valid}
            valid_loss_value = self.sess.run((self.loss_train),feed_dict=valid_feed_dict)
            print "valid loss : ", valid_loss_value, "current_patience : ", current_patience

            # Early Stopping criteria.
            if valid_loss_value < min_valid_loss:
                min_valid_loss = valid_loss_value
                current_patience = 0
            else:
                current_patience+=1
                if current_patience>max_patience:
                    break       
            

        return loss_value, min_valid_loss


    def generate(self,iteration,normalizer):
        
        for i in range(iteration):
            generated_x_value, y_hat_value, loss_value, _ = self.sess.run((self.generated_x,self.y_hat_generate,
                                                                           self.loss_generate,self.step_generate))
            
            generated_x_value, y_hat_value = normalizer.denormalize_data(generated_x_value,y_hat_value)


            print "Generated x : ", generated_x_value
            print "y_hat : ", y_hat_value
            print "loss : ", loss_value

        return y_hat_value


def split_data(x_data,y_data,ratio):
    
    num_data = x_data.shape[0]

    random_index = np.random.permutation(num_data)
    x_data = x_data[random_index]
    y_data = y_data[random_index]
    
    cum_ratio = [sum(ratio[:i+1]) for i in range(3)]
    
    x_train = x_data[:int(num_data*cum_ratio[0])]
    y_train = y_data[:int(num_data*cum_ratio[0])]
    x_valid = x_data[int(num_data*cum_ratio[0]):int(num_data*cum_ratio[1])]
    y_valid = y_data[int(num_data*cum_ratio[0]):int(num_data*cum_ratio[1])]
    x_test = x_data[int(num_data*cum_ratio[1]):]
    y_test = y_data[int(num_data*cum_ratio[1]):]

    return x_train,y_train,x_valid,y_valid,x_test,y_test

class Normalizer():
    

    def normalize_data(self,x_data,y_data): 

        self.x_mean = np.mean(x_data,axis=0)
        self.x_std = np.std(x_data,axis=0)

        self.y_mean = np.mean(y_data,axis=0)
        self.y_std = np.std(y_data,axis=0)
        
        x_data = (x_data - self.x_mean) / self.x_std
        #y_data = (y_data - self.y_mean) / self.y_std
        
        return x_data,y_data
    
    def denormalize_data(self,x_data,y_data):
        
        x_data = x_data * self.x_std + self.x_mean
        #y_data = y_data * self.y_std + self.y_mean
        
        return x_data, y_data

if __name__ == '__main__':
    main()


