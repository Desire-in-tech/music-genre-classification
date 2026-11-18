"""Classical baselines and an optional TensorFlow spectrogram CNN."""
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def build_baseline_models(random_seed=42,n_jobs=-1):
    def pipeline(classifier,scale=False):
        steps=[("imputer",SimpleImputer(strategy="median"))]
        if scale: steps.append(("scaler",StandardScaler()))
        steps.append(("classifier",classifier))
        return Pipeline(steps)
    return {
      "svm_rbf":pipeline(SVC(kernel="rbf",class_weight="balanced"),True),
      "random_forest":pipeline(RandomForestClassifier(n_estimators=400,class_weight="balanced",random_state=random_seed,n_jobs=n_jobs)),
      "extra_trees":pipeline(ExtraTreesClassifier(n_estimators=400,class_weight="balanced",random_state=random_seed,n_jobs=n_jobs)),
    }

def build_spectrogram_cnn(input_shape,num_classes,random_seed=42):
    try: import tensorflow as tf
    except ImportError as exc: raise ImportError("Install TensorFlow to run the CNN experiment (Colab usually includes it).") from exc
    tf.keras.utils.set_random_seed(random_seed)
    inputs=tf.keras.Input(shape=input_shape)
    x=tf.keras.layers.Conv2D(32,3,padding="same",activation="relu")(inputs)
    x=tf.keras.layers.BatchNormalization()(x); x=tf.keras.layers.MaxPooling2D()(x)
    x=tf.keras.layers.Conv2D(64,3,padding="same",activation="relu")(x)
    x=tf.keras.layers.BatchNormalization()(x); x=tf.keras.layers.MaxPooling2D()(x)
    x=tf.keras.layers.Conv2D(128,3,padding="same",activation="relu")(x)
    x=tf.keras.layers.BatchNormalization()(x); x=tf.keras.layers.MaxPooling2D()(x)
    x=tf.keras.layers.GlobalAveragePooling2D()(x)
    x=tf.keras.layers.Dropout(.35)(x)
    outputs=tf.keras.layers.Dense(num_classes,activation="softmax")(x)
    model=tf.keras.Model(inputs,outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    return model
