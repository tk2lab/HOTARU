from keras import Loss


class Minimize(Loss):
    def call(self, y_true, y_pred):
        _ = y_true
        return y_pred
