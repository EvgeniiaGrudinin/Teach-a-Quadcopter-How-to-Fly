from keras import layers, models, optimizers
from keras import backend as K

class Actor: 
    def __init__(self, state_size, action_size, action_low, action_high):  
        self.state_size=state_size
        self.action_size=action_size
        self.action_low=action_low
        self.action_high=action_high
        
        self.create_model()

    def create_model(self):
        state_input=layers.Input(shape=(self.state_size,), name='state_input')
        hidden=layers.Dense(32, activation='relu')(state_input)
        hidden=layers.Dense(64, activation='relu')(hidden)
        hidden=layers.Dense(32, activation='relu')(hidden)
        output=layers.Dense(units=self.action_size, activation='sigmoid', name='output')(hidden)

        actions=layers.Lambda(lambda x: (x*(self.action_high - self.action_low))+self.action_low, name='actions')(output)
        self.model=models.Model(inputs=state_input, outputs=actions)
        action_gradients=layers.Input(shape=(self.action_size,))
        loss=K.mean(-action_gradients*actions)
        optimizer=optimizers.Adam()
        optimizerN=optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn=K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[])