import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# here input elements of the sequence are going to be some num_cluster Dimensional binary vectors 

num_clusters = 100 # number of dimensions of the input vector of the job profile
num_profiles = 3 # this is the number of job profiles and company data for each profile of the candidate
dim_comp_data = 100 # number of dimensions of the input vector of the company data 



job_input = keras.Input(
    shape=(None,num_clusters), name="jobProfiles"
) 

comp_input = keras.Input(shape=(None,dim_comp_data), name="companyData")

hiring_input = keras.Input(shape=(dim_comp_data+num_clusters,),name ="hiringData")
edu_skill_input = keras.Input(shape=(2,),name="EduSkillInput")

job_features_1 = layers.LSTM(num_clusters, return_sequences = True)(job_input)
job_features_2 = layers.LSTM(num_clusters)(job_features_1)

job_features = layers.Dropout(0.1)(job_features_2)


comp_features_1 = layers.LSTM(dim_comp_data, return_sequences = True)(comp_input)
comp_features_2 = layers.LSTM(dim_comp_data)(comp_features_1) 

comp_features = layers.Dropout(0.1)(comp_features_2)

x = layers.concatenate([job_features, comp_features,hiring_input,edu_skill_input])

first_dense = layers.Dense(2*(dim_comp_data+num_clusters)+2, name="firstLayer")(x)
first_dense = layers.Dropout(0.1)(first_dense)

second_dense = layers.Dense(2*(dim_comp_data+num_clusters)+2, name="secondLayer")(first_dense)
second_dense = layers.Dropout(0.1)(second_dense)

output_layer = layers.Dense(3, activation="softmax",name="finalOutput")(second_dense)

model = keras.Model(
    inputs=[job_input, comp_input, hiring_input,edu_skill_input],
    outputs=[output_layer],
)

keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-5)

model.compile(
    optimizer=opt,               
    loss= 'sparse_categorical_crossentropy',
    metrics=['accuracy']
)
# Random training data change here

job_data = np.random.randint(2, size=(1000,3,100))
comp_data = np.random.randint(2, size=(1000,3,100))
hiring_data = np.random.randint(2, size=(1000,200))
edu_skill_data = np.random.randint(2, size=(1000,2))
final_targets = np.random.randint(3, size=(1000,))

model.fit(
    {"jobProfiles": job_data, "companyData": comp_data, "hiringData": hiring_data, "EduSkillInput":edu_skill_data},
    {"finalOutput": final_targets},
    epochs=20,
    batch_size=2,
)

model.save('prototype_one')

# Random test data
job_test = np.random.randint(2, size=(2,3,100))
comp_test = np.random.randint(2, size=(2,3,100))
hiring_test = np.random.randint(2, size=(2,200))
edu_skill_test = np.random.randint(2, size=(2,2))
predictions = model.predict([job_test,comp_test,hiring_test,edu_skill_test])
print(predictions)
print(np.argmax(predictions[0]))