from keras.models import Model


class IntermediateVisualizer(Model):

  def __init__(self, model, layer_name, data):
    self.data = data
    outputs = model.get_layer(layer_name).output
    super(IntermediateVisualizer, self).__init__(inputs=model.input,
                                                 outputs=outputs)

  def infer(self):
    return self.predict(self.data)
