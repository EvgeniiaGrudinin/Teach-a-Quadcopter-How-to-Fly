from keras import backend as K
from keras import layers, models, optimizers
from keras.layers import Dense

class Critic:
    def __init__(self, state_size, action_size):
        self.state_size=state_size
        self.action_size=action_size
        
        self.create_model()
    
    def create_model(self):
        state_input=layers.Input(shape=(self.state_size,), name='state_input')
        action_input=layers.Input(shape=(self.action_size,), name='action_input')
        
        hid_state=Dense(32, activation='relu')(state_input)
        hid_state=Dense(64, activation='relu')(hid_state)
        hid_action=Dense(32, activation='relu')(action_input)
        hid_action=Dense(64, activation='relu')(hid_action)

        fin=layers.Add()([hid_state, hid_action])
        fin=layers.Activation('relu')(fin)
        Q_value=Dense(units=1, name='q_value')(fin)
        self.model=models.Model(inputs=[state_input, action_input], outputs=Q_value)
        optimizer=optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss='mse')
        action_gradients=K.gradients(Q_value, action_input)

        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)