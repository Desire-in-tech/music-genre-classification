"""Classical baselines and a log-Mel spectrogram CNN."""
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


def build_spectrogram_cnn(input_shape, num_classes, random_seed=42, augment=True):
    """Build a moderately sized CNN for fixed-size [0, 1] log-Mel spectrograms.

    Augmentation is deliberately mild: large translations/noise can erase genre cues
    in this small dataset. Set augment=False for train-subset memorization diagnostics.
    """
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "Install TensorFlow to run the CNN experiment (Colab usually includes it)."
        ) from exc

    if len(input_shape) != 3 or input_shape[-1] != 1:
        raise ValueError("input_shape must describe a single-channel 2D spectrogram")
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")

    tf.keras.utils.set_random_seed(random_seed)
    regularizer = tf.keras.regularizers.l2(1e-5)
    inputs = tf.keras.Input(shape=input_shape, name="log_mel_input")
    x = inputs

    if augment:
        x = tf.keras.layers.RandomTranslation(
            height_factor=0.01, width_factor=0.03,
            fill_mode="nearest", name="spec_shift"
        )(x)
        x = tf.keras.layers.GaussianNoise(0.005, name="spec_noise")(x)

    for block, filters, dropout_rate in (
        (1, 32, 0.10),
        (2, 64, 0.15),
        (3, 128, 0.20),
    ):
        x = tf.keras.layers.Conv2D(
            filters, 3, padding="same", use_bias=False,
            kernel_initializer="he_normal", kernel_regularizer=regularizer,
            name=f"conv_{block}"
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"bn_{block}")(x)
        x = tf.keras.layers.Activation("relu", name=f"relu_{block}")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=2, name=f"pool_{block}")(x)
        x = tf.keras.layers.Dropout(dropout_rate, name=f"dropout_{block}")(x)

    avg = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    maximum = tf.keras.layers.GlobalMaxPooling2D(name="global_max_pool")(x)
    x = tf.keras.layers.Concatenate(name="pooled_features")([avg, maximum])
    x = tf.keras.layers.Dense(
        128, activation="relu", kernel_initializer="he_normal",
        kernel_regularizer=regularizer, name="genre_dense"
    )(x)
    x = tf.keras.layers.Dropout(0.35, name="genre_dropout")(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="genre_probabilities"
    )(x)

    model = tf.keras.Model(inputs, outputs, name="logmel_genre_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
