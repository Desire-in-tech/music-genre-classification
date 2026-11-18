"""Classical baselines and a regularized TensorFlow spectrogram CNN."""
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_baseline_models(random_seed=42, n_jobs=-1):
    def pipeline(classifier, scale=False):
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale:
            steps.append(("scaler", StandardScaler()))
        steps.append(("classifier", classifier))
        return Pipeline(steps)

    return {
        "svm_rbf": pipeline(SVC(kernel="rbf", class_weight="balanced"), True),
        "random_forest": pipeline(RandomForestClassifier(
            n_estimators=400, class_weight="balanced", random_state=random_seed, n_jobs=n_jobs
        )),
        "extra_trees": pipeline(ExtraTreesClassifier(
            n_estimators=400, class_weight="balanced", random_state=random_seed, n_jobs=n_jobs
        )),
    }


def build_spectrogram_cnn(input_shape, num_classes, random_seed=42):
    """Build a smaller CNN with train-only spectrogram augmentation and L2 regularization."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "Install TensorFlow to run the CNN experiment (Colab usually includes it)."
        ) from exc

    tf.keras.utils.set_random_seed(random_seed)
    regularizer = tf.keras.regularizers.l2(1e-4)
    inputs = tf.keras.Input(shape=input_shape, name="log_mel_input")

    # Keras augmentation layers are active during fit(training=True) and inactive at validation/inference.
    x = tf.keras.layers.RandomTranslation(
        height_factor=0.035, width_factor=0.07, fill_mode="nearest", name="spec_shift"
    )(inputs)
    x = tf.keras.layers.RandomContrast(0.10, name="spec_contrast")(x)
    x = tf.keras.layers.GaussianNoise(0.02, name="spec_noise")(x)

    for filters, dropout_rate in ((16, 0.10), (32, 0.15), (64, 0.20)):
        x = tf.keras.layers.Conv2D(
            filters, 3, padding="same", use_bias=False, kernel_regularizer=regularizer
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.Dropout(0.50)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="genre_probabilities")(x)
    model = tf.keras.Model(inputs, outputs, name="regularized_logmel_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
