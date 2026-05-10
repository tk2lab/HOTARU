from keras import Loss


class Minimize(Loss):
    def call(self, y_true, y_pred):
        _ = y_pred
        return y_true
