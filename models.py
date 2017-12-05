import pickle
import numpy
import math

from sklearn.preprocessing import StandardScaler

#from keras.models import Sequential
#from keras.layers.core import Dense, Dropout, Activation, Reshape
#from keras.layers.embeddings import Embedding
#from keras.layers import Merge
from keras.callbacks import ModelCheckpoint, TensorBoard

# switch over to new keras
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, Input, Embedding, Reshape, Activation
from keras.layers import concatenate


from prepare_nn_features import split_features

# ahhh this is problem
class KerasModel(object):

    def __init__(self, train_ratio):
        self.train_ratio = train_ratio
        self.__load_data()

    def evaluate(self):
        if self.train_ratio == 1:
            return 0
        total_sqe = 0
        num_real_test = 0
        for record, sales in zip(self.X_val, self.y_val):
            if sales == 0:
                continue
            guessed_sales = self.guess(record)
            sqe = ((sales - guessed_sales) / sales) ** 2
            total_sqe += sqe
            num_real_test += 1
        result = math.sqrt(total_sqe / num_real_test)
        return result

    def __load_data(self):
        f = open('feature_train_data.pickle', 'rb')
        (self.X, self.y) = pickle.load(f)
        self.X = numpy.array(self.X)
        self.y = numpy.array(self.y)
        self.num_records = len(self.X)
        self.train_size = int(self.train_ratio * self.num_records)
        self.test_size = self.num_records - self.train_size
        self.X, self.X_val = self.X[:self.train_size], self.X[self.train_size:]
        self.y, self.y_val = self.y[:self.train_size], self.y[self.train_size:]


class NN_with_EntityEmbedding(KerasModel):

    def __init__(self, train_ratio):
        super().__init__(train_ratio)
        self.build_preprocessor(self.X)
        self.nb_epoch = 20
        self.checkpointer = ModelCheckpoint(filepath="best_model_weights.hdf5", verbose=1, save_best_only=True)
        self.tb_check = TensorBoard(log_dir='./logs', histogram_freq=1, batch_size=32, 
                                    write_graph=True, write_grads=True, write_images=False, 
                                    embeddings_freq=1, embeddings_layer_names=None, embeddings_metadata=None)
        self.max_log_y = numpy.max(numpy.log(self.y))
        self.min_log_y = numpy.min(numpy.log(self.y))
        self.__build_keras_model()
        self.fit()

    def build_preprocessor(self, X):
        X_list = split_features(X)
        # Google trend de
        self.gt_de_enc = StandardScaler()
        self.gt_de_enc.fit(X_list[32])
        # Google trend state
        self.gt_state_enc = StandardScaler()
        self.gt_state_enc.fit(X_list[33])

    def preprocessing(self, X):
        X_list = split_features(X)
        X_list[32] = self.gt_de_enc.transform(X_list[32])
        X_list[33] = self.gt_state_enc.transform(X_list[33])
        return X_list

    def __build_keras_model(self):

        in_vec = []
        models = []

        model_store_in = Input(shape=(1,))
        model_store_embedding = Embedding(1115, 50, input_length = 1)(model_store_in)
        model_store_reshape = Reshape(target_shape=(50,))(model_store_embedding)
        in_vec.append(model_store_in)
        models.append(model_store_reshape)

        model_dow_in = Input(shape=(1,))
        model_dow_embedding = Embedding(7, 6, input_length=1)(model_dow_in)
        model_dow_reshape = Reshape(target_shape=(6,))(model_dow_embedding)
        in_vec.append(model_dow_in)          
        models.append(model_dow_reshape)

        model_promo_in = Input(shape=(1,))
        model_promo_dense = Dense(1, input_dim=1)(model_promo_in)
        in_vec.append(model_promo_in) 
        models.append(model_promo_dense)

        model_year_in = Input(shape=(1,))
        model_year_embedding = Embedding(3, 2, input_length=1)(model_year_in)
        model_year_reshape = Reshape(target_shape=(2,))(model_year_embedding)
        in_vec.append(model_year_in)  
        models.append(model_year_reshape)

        model_month_in = Input(shape=(1,))
        model_month_embedding = Embedding(12, 6, input_length=1)(model_month_in)
        model_month_reshape = Reshape(target_shape=(6,))(model_month_embedding)
        in_vec.append(model_month_in)          	
        models.append(model_month_reshape)

        model_day_in = Input(shape=(1,))
        model_day_embedding = Embedding(31, 10, input_length=1)(model_day_in)
        model_day_reshape = Reshape(target_shape=(10,))(model_day_embedding)
        in_vec.append(model_day_in)  
        models.append(model_day_reshape)

        model_stateholiday_in = Input(shape=(1,))
        model_stateholiday_embedding = Embedding(4, 3, input_length=1)(model_stateholiday_in) 
        model_stateholiday_reshape = Reshape(target_shape=(3,))(model_stateholiday_embedding)
        in_vec.append(model_stateholiday_in) 
        models.append(model_stateholiday_reshape)
	
        model_school_in = Input(shape=(1,))
        model_school_dense = Dense(1, input_dim=1)(model_school_in)
        in_vec.append(model_school_in) 
        models.append(model_school_dense)

        model_competemonths_in = Input(shape=(1,))
        model_competemonths_embedding = Embedding(25, 2, input_length=1)(model_competemonths_in)
        model_competemonths_reshape = Reshape(target_shape=(2,))(model_competemonths_embedding)
        in_vec.append(model_competemonths_in)  
        models.append(model_competemonths_reshape)

        model_promo2weeks_in = Input(shape=(1,))
        model_promo2weeks_embedding = Embedding(26, 1, input_length=1)(model_promo2weeks_in)
        model_promo2weeks_reshape = Reshape(target_shape=(1,))(model_promo2weeks_embedding)
        in_vec.append(model_promo2weeks_in)  
        models.append(model_promo2weeks_reshape)

        model_lastestpromo2months_in = Input(shape=(1,))
        model_lastestpromo2months_embedding = Embedding(4, 1, input_length=1)(model_lastestpromo2months_in) 
        model_lastestpromo2months_reshape = Reshape(target_shape=(1,))(model_lastestpromo2months_embedding)
        in_vec.append(model_lastestpromo2months_in) 
        models.append(model_lastestpromo2months_reshape)

#####
        model_distance_in = Input(shape=(1,))
        model_distance_dense = Dense(1, input_dim=1)(model_distance_in)
        in_vec.append(model_distance_in) 
        models.append(model_distance_dense)

        model_storetype_in = Input(shape=(1,))
        model_storetype_embedding = Embedding(5, 2, input_length=1)(model_storetype_in) 
        model_storetype_reshape = Reshape(target_shape=(2,))(model_storetype_embedding)
        in_vec.append(model_storetype_in) 
        models.append(model_storetype_reshape)

        model_assortment_in = Input(shape=(1,))
        model_assortment_embedding = Embedding(4, 3, input_length=1)(model_assortment_in) 
        model_assortment_reshape = Reshape(target_shape=(3,))(model_assortment_embedding)
        in_vec.append(model_assortment_in) 
        models.append(model_assortment_reshape)

        model_promointerval_in = Input(shape=(1,))
        model_promointerval_embedding = Embedding(4, 3, input_length=1)(model_promointerval_in) 
        model_promointerval_reshape = Reshape(target_shape=(3,))(model_promointerval_embedding)
        in_vec.append(model_promointerval_in) 
        models.append(model_promointerval_reshape)

        model_competyear_in = Input(shape=(1,))
        model_competyear_embedding = Embedding(18, 4, input_length=1)(model_competyear_in) 
        model_competyear_reshape = Reshape(target_shape=(4,))(model_competyear_embedding)
        in_vec.append(model_competyear_in) 	
        models.append(model_competyear_reshape)

        model_promotyear_in = Input(shape=(1,))
        model_promotyear_embedding = Embedding(8, 4, input_length=1)(model_promotyear_in) 
        model_promotyear_reshape = Reshape(target_shape=(4,))(model_promotyear_embedding)
        in_vec.append(model_promotyear_in) 	
        models.append(model_promotyear_reshape)

        model_germanstate_in = Input(shape=(1,))
        model_germanstate_embedding = Embedding(12, 6, input_length=1)(model_germanstate_in) 
        model_germanstate_reshape = Reshape(target_shape=(6,))(model_germanstate_embedding)
        in_vec.append(model_germanstate_in) 	
        models.append(model_germanstate_reshape)

        model_woy_in = Input(shape=(1,))
        model_woy_embedding = Embedding(53, 2, input_length=1)(model_woy_in) 
        model_woy_reshape = Reshape(target_shape=(2,))(model_woy_embedding)
        in_vec.append(model_woy_in) 	
        models.append(model_woy_reshape)
	
        model_temperature_in = Input(shape=(3,)) # may break
        model_temperature_dense = Dense(3, input_dim=3)(model_temperature_in)
        in_vec.append(model_temperature_in)
        models.append(model_temperature_dense)
######

        model_humidity_in = Input(shape=(3,)) # may break
        model_humidity_dense = Dense(3, input_dim=3)(model_humidity_in)
        in_vec.append(model_humidity_in)
        models.append(model_humidity_dense)
	
        model_wind_in = Input(shape=(2,)) # may break
        model_wind_dense = Dense(2, input_dim=2)(model_wind_in)
        in_vec.append(model_wind_in)
        models.append(model_wind_dense)

        model_cloud_in = Input(shape=(1,)) # may break
        model_cloud_dense = Dense(1, input_dim=1)(model_cloud_in)
        in_vec.append(model_cloud_in)
        models.append(model_cloud_dense)

        model_weatherevent_in = Input(shape=(1,))
        model_weatherevent_embedding = Embedding(22, 4, input_length=1)(model_weatherevent_in) 
        model_weatherevent_reshape = Reshape(target_shape=(4,))(model_weatherevent_embedding)
        in_vec.append(model_weatherevent_in) 	
        models.append(model_weatherevent_reshape)

        model_promo_forward_in = Input(shape=(1,))
        model_promo_forward_embedding = Embedding(8, 1, input_length=1)(model_promo_forward_in) 
        model_promo_forward_reshape = Reshape(target_shape=(1,))(model_promo_forward_embedding)
        in_vec.append(model_promo_forward_in) 		
        models.append(model_promo_forward_reshape)

        model_promo_backward_in = Input(shape=(1,))
        model_promo_backward_embedding = Embedding(8, 1, input_length=1)(model_promo_backward_in) 
        model_promo_backward_reshape = Reshape(target_shape=(1,))(model_promo_backward_embedding)
        in_vec.append(model_promo_backward_in) 	
        models.append(model_promo_backward_reshape)

        model_stateholiday_forward_in = Input(shape=(1,))
        model_stateholiday_forward_embedding = Embedding(8, 1, input_length=1)(model_stateholiday_forward_in) 
        model_stateholiday_forward_reshape = Reshape(target_shape=(1,))(model_stateholiday_forward_embedding)
        in_vec.append(model_stateholiday_forward_in) 	
        models.append(model_stateholiday_forward_reshape)

        model_sateholiday_backward_in = Input(shape=(1,))
        model_sateholiday_backward_embedding = Embedding(8, 1, input_length=1)(model_sateholiday_backward_in) 
        model_sateholiday_backward_reshape = Reshape(target_shape=(1,))(model_sateholiday_backward_embedding)
        in_vec.append(model_sateholiday_backward_in) 	
        models.append(model_sateholiday_backward_reshape)

        model_stateholiday_count_forward_in = Input(shape=(1,))
        model_stateholiday_count_forward_embedding = Embedding(3, 1, input_length=1)(model_stateholiday_count_forward_in) 
        model_stateholiday_count_forward_reshape = Reshape(target_shape=(1,))(model_stateholiday_count_forward_embedding)
        in_vec.append(model_stateholiday_count_forward_in) 	
        models.append(model_stateholiday_count_forward_reshape)

        model_stateholiday_count_backward_in = Input(shape=(1,))
        model_stateholiday_count_backward_embedding = Embedding(3, 1, input_length=1)(model_stateholiday_count_backward_in) 
        model_stateholiday_count_backward_reshape = Reshape(target_shape=(1,))(model_stateholiday_count_backward_embedding)
        in_vec.append(model_stateholiday_count_backward_in) 	
        models.append(model_stateholiday_count_backward_reshape)

        model_schoolholiday_forward_in = Input(shape=(1,))
        model_schoolholiday_forward_embedding = Embedding(8, 1, input_length=1)(model_schoolholiday_forward_in) 
        model_schoolholiday_forward_reshape = Reshape(target_shape=(1,))(model_schoolholiday_forward_embedding)
        in_vec.append(model_schoolholiday_forward_in) 		
        models.append(model_schoolholiday_forward_reshape)

        model_schoolholiday_backward_in = Input(shape=(1,))
        model_schoolholiday_backward_embedding = Embedding(8, 1, input_length=1)(model_schoolholiday_backward_in) 
        model_schoolholiday_backward_reshape = Reshape(target_shape=(1,))(model_schoolholiday_backward_embedding)
        in_vec.append(model_schoolholiday_backward_in) 		
        models.append(model_schoolholiday_backward_reshape)
	
        model_googletrend_de_in = Input(shape=(1,)) 
        model_googletrend_de_dense = Dense(1, input_dim=1)(model_googletrend_de_in)
        in_vec.append(model_googletrend_de_in)	
        models.append(model_googletrend_de_dense)

        model_googletrend_state_in = Input(shape=(1,)) 
        model_googletrend_state_dense = Dense(1, input_dim=1)(model_googletrend_state_in)
        in_vec.append(model_googletrend_state_in)			
        models.append(model_googletrend_state_dense)

        # model_weather = Sequential()
        # model_weather.add(Merge([model_temperature, model_humidity, model_wind, model_weatherevent], mode='concat'))
        # model_weather.add(Dense(1))
        # model_weather.add(Activation('relu'))
        # models.append(model_weather)

        main_merger = concatenate(models)
        main_dropout_1 = Dropout(0.02)(main_merger)
        main_dense_1 = Dense(1000, kernel_initializer='uniform')(main_dropout_1)
        main_activation_1 = Activation('relu')(main_dense_1)
        main_dense_2 = Dense(500, kernel_initializer='uniform')(main_activation_1)
        main_activation_2 = Activation('relu')(main_dense_2)
        main_dense_3 = Dense(1)(main_activation_2)
        main_activation_3 = Activation('sigmoid')(main_dense_3)

	# something wrong here need to fix
        NN_model = Model(inputs = in_vec, outputs = main_activation_3)
        NN_model.compile(loss='mean_absolute_error', optimizer='adam')

        self.model = NN_model

    def _val_for_fit(self, val):
        val = numpy.log(val) / self.max_log_y
        return val

    def _val_for_pred(self, val):
        return numpy.exp(val * self.max_log_y)

    def fit(self):
        if self.train_ratio < 1:
            self.model.fit(self.preprocessing(self.X), self._val_for_fit(self.y),
                           validation_data=(self.preprocessing(self.X_val), self._val_for_fit(self.y_val)),
                           nb_epoch=self.nb_epoch, batch_size=128,
                           callbacks=[self.tb_check], # try this
                           )
            # self.model.load_weights('best_model_weights.hdf5')
            print("Result on validation data: ", self.evaluate())
        else:
            self.model.fit(self.preprocessing(self.X), self._val_for_fit(self.y),
                           nb_epoch=self.nb_epoch, batch_size=128)

    def guess(self, feature):
        feature = numpy.array(feature).reshape(1, -1)
        return self._val_for_pred(self.model.predict(self.preprocessing(feature)))[0][0]
