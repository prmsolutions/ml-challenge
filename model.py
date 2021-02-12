import random
import math
from typing import *

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
import numpy as np

# imagine that internal tools is our own library of utility functions
import internal_tools
from internal_tools import get_database_connection


class OrderCartDataGenerator(Sequence):
    def __init__(
        self,
        client_name: str,
        n_products: int,
        batch_size: int = 64
    ):  
        self.n_products = n_products
        self.batch_size = batch_size
        self.db = get_database_connection() # establish a connection to our internal database

    def get_next_traininig_batch(self, index):
        """
        return a tuple cosisting of the next-in-sequence input training data and the target data
        """
        
        # the carts or cart blobs in the order_cart_generator are dictionaries and contain a key called "products".
        # This key contains a list of dictionaries, and each dictionary contains data about a product that was bought in that particular basket
        # In short within each product blob you will find keys such as 
        # 'product_name', 'price', 'product_description', 'product_category', 'manufacturer', 'quantity_purchased', 'purchase_date', 'product_id', 'product_index'
        # the 'product_index' field is unique for each product for a client and represents the position of the product in a virtual array of products
        # You can check out the FAQ section to see what that look likes in more detail if this is not clear (we totally get that, which is why we have the FAQ)

        order_cart_generator = self.db.get_order_cart_generator(
            client=self.client_name, 
            object_type="order_carts", 
            segment=index,
            batch_size=self.batch_size
        )

        X = np.zeros(shape=(self.batch_size, self.n_products))
        Y = np.zeros(shape=(self.batch_size, self.n_products))
        


        for i, cart in enumerate(order_cart_generator):
            products_in_cart = cart['products'] # this will be a list of dictionaries
            product_indices = [p['product_index'] for p in products_in_cart]

            # hide one product from the order cart. The model has to predict the hidden product, given all the other products in the cart
            hidden_product_index = random.choice(product_indices)
            for product_index in product_indices:
                if product_index==hidden_product_index:
                    continue
                X[i][product_index] = 1.0

            Y[i][hidden_product_index] = 1.0

        return X, Y

    def get_num_training_batches(self):
        total_num_documents = self.db.get_document_count(client=self.client_name, type="order_carts")
        return math.ceil(total_num_documents / self.batch_size)
    
    def on_epoch_end(self):
        # reset the index back to zero so we can start training the next epoch
        self.current_training_batch_index = 0


def get_autoencoder_model(
    n_products: int
) -> tf.keras.Model:
    """
    returns an instance of a tensorflow.keras.models Model
    Autoencoder structure 
    """
    input = Input(shape=(n_products,))

    # encoder part of the autoencoder
    x = Dense(4096, activation='relu')(input)
    x = Dense(2048, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    
    # code layer, or latent space
    x = Dense(512, activation='relu')(x)

    # decoder part of the autoencoder
    x = Dense(1024, activation='relu')(x)
    x = Dense(2048, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)

    # output. We are using a softmax activation because we are predicting 1 missing product as the task
    output = Dense(n_products, activation='softmax')(x)

    model = Model(input, output)
    model.compile(optimizer='adam', loss="cosine_similarity")

    return model


def train_model(
    client_name: str, 
    batch_size: int, 
    n_epochs: int = 10
) -> tf.keras.Model:
    # connect to the Proton database and use utility function to generate dictionaries of products
    # each dictionary containes the fields 'product_id', '
    n_products = internal_tools.product_count(client_name) # IMPORTANT this number is usually around 300 000, so keep that in mind
    data_generator = OrderCartDataGenerator(client_name, n_products, batch_size)
    model = get_autoencoder_model(n_products)
    num_training_batches = data_generator.get_num_training_batches()

    for epoch in range(n_epochs):
        for i in range(num_training_batches):
            X, Y = data_generator.get_next_traininig_batch(index=i)
            loss = model.train_on_batch(X, Y)
            print(f"Epoch {epoch}/{n_epochs}. Batch {i}/{num_training_batches}. Model loss: {loss: 0.4f}", end="\r")
        print() # new line for new epoch
    
    return model


def predict_missing_products(
    client_name: str,
    model: tf.keras.models.Model, 
    product_blobs: List[dict],
    k: int = 10
) -> List[dict]:
    """
    Given a list of products and a complete-the-cart model predicts what products are missing from the cart
    """
    n_products = internal_tools.product_count(client_name)
    X = np.zeros(shape=(1, n_products))

    for product_blob in product_blobs:
        product_index = product_blob['product_index']
        X[0][product_index] = 1.0
    
    Y = model.predict(X)
    Y = Y.squeeze() # convert from shape (1, n_products) to (n_products, )
    top_k_indices = Y.argsort()[::-1][:k] # we only want the products corresponding to positions of the top k scores

    missing_products = []
    for index in top_k_indices:
        product_blob = internal_tools.get_index_product(index)
        missing_products.append(product_blob)
    
    return missing_products