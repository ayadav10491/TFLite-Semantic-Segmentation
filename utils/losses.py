"""

@Author: Akash
@Github: https://github.com/ayadav10491
@Project: https://github.com/ayadav10491/TFLite-Semantic-Segmentation

"""
import tensorflow as tf

backend = tf.keras.backend


def categorical_crossentropy_with_logits(out_original, out_prediction):
    # compute cross entropy
    cross_entropy = backend.categorical_crossentropy(out_original, out_prediction, from_logits=True)

    # compute loss
    loss = backend.mean(cross_entropy, axis=-1)

    return loss


def focal_loss(alpha=0.25, gamma=2.0):

    def loss(out_original, out_prediction):
        y_pred = backend.softmax(y_pred)
        cross_entropy = backend.categorical_crossentropy(out_original, out_prediction, from_logits=False)
        # compute weights
        weights = backend.sum(alpha * backend.pow(1 - out_prediction, gamma) * out_original, axis=-1)
        return backend.mean(weights * cross_entropy, axis=[1, 2])

    return loss


def miou_loss(weights=None, num_classes=2):

    if weights is not None:
        assert len(weights) == num_classes
        weights = tf.convert_to_tensor(weights)

    else:
        weights = tf.convert_to_tensor([1.] * num_classes)

    def loss(out_original, out_prediction):
        out_prediction = backend.softmax(out_prediction)

        inter = out_prediction * out_original
        inter = backend.sum(inter, axis=[1, 2])

        union = out_prediction + out_original - (out_prediction * out_original)
        union = backend.sum(union, axis=[1, 2])

        return -backend.mean((weights * inter) / (weights * union + 1e-8))

    return loss


