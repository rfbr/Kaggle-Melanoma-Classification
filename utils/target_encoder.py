from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing._encoders import _BaseEncoder
from sklearn.preprocessing._label import _encode, _encode_check_unknown
from sklearn.model_selection import KFold
import numpy as np


class _BaseTargetEncoder(_BaseEncoder):
    """
    Base class for target encoders that includes the code to categorize and
    transform the input features.
    """

    def _check_Y(self, y):
        return check_array(y, ensure_2d=False)

    def _fit(self, X, y, handle_unknown_category='error'):
        X_list, n_samples, n_features = self._check_X(X)

        if self.categories != 'auto':
            if len(self.categories) != n_features:
                raise ValueError("Shape mismatch: if categories is an array,"
                                 " it has to be of shape (n_features,).")

        self.categories_ = []
        self.target_stat_by_feature_x_cat_ = {}
        self.target_mean_ = y.mean()
        for i in range(n_features):
            Xi = X_list[i]

            if self.categories == 'auto':
                cats = _encode(Xi)
            else:
                cats = np.array(self.categories[i], dtype=Xi.dtype)
                if Xi.dtype != object:
                    if not np.all(np.sort(cats) == cats):
                        raise ValueError("Unsorted categories are not "
                                         "supported for numerical categories")
                if handle_unknown_category == 'error':
                    diff = _encode_check_unknown(Xi, cats)
                    if diff:
                        msg = ("Found unknown categories {0} in column {1}"
                               " during fit".format(diff, i))
                        raise ValueError(msg)
            self.categories_.append(cats)
            self.target_stat_by_feature_x_cat_[i] = {}
            for cat in cats:
                target_x_cat = y[(Xi == cat)]
                self.target_stat_by_feature_x_cat_[i][cat] = {
                    'mean': target_x_cat.mean(),
                    'count': len(target_x_cat)
                } if self.with_count else {
                    'mean': target_x_cat.mean()
                }
        return X_list, n_samples, n_features

    def _transform(self, X, handle_unknown, with_count):
        X_list, n_samples, n_features = self._check_X(X)

        X_encoded = np.zeros((n_samples, n_features), dtype=np.float)
        X_mask = np.ones((n_samples, n_features), dtype=np.bool)
        if with_count:
            X_count = np.zeros((n_samples, n_features), dtype=np.float)
        if n_features != len(self.categories_):
            raise ValueError(
                "The number of features in X is different to the number of "
                "features of the fitted data. The fitted data had {} features "
                "and the X has {} features.".format(len(self.categories_, ),
                                                    n_features))

        for i in range(n_features):
            Xi = X_list[i]
            diff, valid_mask = _encode_check_unknown(Xi,
                                                     self.categories_[i],
                                                     return_mask=True)
            if not np.all(valid_mask):
                if handle_unknown == 'error':
                    msg = ("Found unknown categories {0} in column {1}"
                           " during transform".format(diff, i))
                    raise ValueError(msg)
                else:
                    # Set the problematic rows to an acceptable value and
                    # continue `The rows are marked `X_mask` and will be
                    # removed later.
                    X_mask[:, i] = valid_mask
                    # cast Xi into the largest string type necessary
                    # to handle different lengths of numpy strings
                    if (self.categories_[i].dtype.kind in ('U', 'S')
                            and self.categories_[i].itemsize > Xi.itemsize):
                        Xi = Xi.astype(self.categories_[i].dtype)
                    else:
                        Xi = Xi.copy()

            # We use check_unknown=False, since _encode_check_unknown was
            # already called above.
            encoded, count = self._encode_category(
                Xi,
                valid_mask,
                self.target_stat_by_feature_x_cat_[i],
                handle_unknown=handle_unknown,
                with_count=self.with_count)
            X_encoded[:, i] = encoded
            if with_count:
                X_count[:, i] = count
        out = np.hstack((X_encoded, X_count)) if with_count else X_encoded
        return out

    def _encode_category(self, values, valid_mask, mapping, handle_unknown,
                         with_count):
        """Custom helper function to factorize and encode values

        Parameters
        ----------
        values : array
            Values to factorize or encode.
        valid_mask : boolean array
            Mask of the same shape as `values` indicating the valid values.
        mapping : dict
            Dictionary mapping categories to the corresponding target mean.
        handle_unknown : {'error', 'target_mean', 'nan'}, default='error'
        Whether to raise an error or ignore if an unknown categorical feature
        is present during transform (default is to raise). When this parameter
        is set to 'target_mean' and an unknown category is encountered during
        transform, the resulting encoded values for this category will be the target mean.
        When this parameter is set to 'nan' and an unknown category is encountered during
        transform, the resulting encoded values for this category will be NaN.

        Returns
        -------
        encoded : array
            Encoded feature
        """
        encoded = []
        count = []
        for idx, cat in enumerate(values):
            if valid_mask[idx]:
                if with_count:
                    encoded.append(mapping[cat]['mean'])
                    count.append(mapping[cat]['count'])
                else:
                    encoded.append(mapping[cat]['mean'])
            else:
                if handle_unknown == 'target_mean':
                    encoded.append(self.target_mean_)
                    if with_count:
                        count.append(0)
                elif handle_unknown == 'nan':
                    encoded.append(np.nan)
                    if with_count:
                        count.append(0)
        return encoded, count


class KFoldTargetEncoder(_BaseTargetEncoder):
    """
    Encode categorical features as the mean of the corresponding target variable.
    The input to this transformer should be an array-like of integers or
    strings, denoting the values taken on by categorical (discrete) features.
    """

    def __init__(self,
                 categories='auto',
                 with_count=False,
                 n_splits=5,
                 shuffle=False,
                 random_state=None,
                 dtype=np.float64,
                 test_encoding='random_fold',
                 handle_fit_unknown='target_mean',
                 handle_transform_unknown='target_mean'):
        if handle_fit_unknown not in ['target_mean', 'nan']:
            raise ValueError(
                "handle_fit_unknown variable should be either 'target_mean' or 'nan'"
            )
        if handle_transform_unknown not in ['target_mean', 'error', 'nan']:
            raise ValueError(
                "handle_transform_unknown variable should be either 'error', 'target_mean' or 'nan'"
            )
        if test_encoding not in ['target_mean', 'random_fold']:
            raise ValueError(
                "test_encoding variable should be either 'target_mean' or 'random_fold'"
            )
        self.categories = categories
        self.with_count = with_count
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.dtype = dtype
        self.test_encoding = test_encoding
        self.handle_fit_unknown = handle_fit_unknown
        self.handle_transform_unknown = handle_transform_unknown

    def fit(self, X, y):
        """
        Fit TargetEncoder to X.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to determine the categories of each feature.

        y : array-like of shape (n_samples,)
            Target values.
        """
        y = self._check_Y(y)
        X_list, _, n_features = self._fit(X, y)
        k_fold = KFold(n_splits=self.n_splits,
                       shuffle=self.shuffle,
                       random_state=self.random_state)
        self.fold_mapping_ = {}

        for fold, (train_index, val_index) in enumerate(k_fold.split(X)):
            self.fold_mapping_[fold] = {}
            self.fold_mapping_[fold]['idx'] = val_index if self.shuffle else [
                val_index[0], val_index[-1]
            ]
            y_train = y[train_index]

            for i in range(n_features):
                self.fold_mapping_[fold][i] = {}

                Xi_train = X_list[i][train_index]

                for category in self.categories_[i]:
                    target_x_cat = y_train[Xi_train == category]
                    if (len(target_x_cat) > 0):
                        self.fold_mapping_[fold][i][category] = {
                            'mean': target_x_cat.mean(),
                            'count': len(target_x_cat)
                        } if self.with_count else {
                            'mean': target_x_cat.mean()
                        }
                    else:
                        if self.handle_fit_unknown == 'nan':
                            self.fold_mapping_[fold][i][category] = {
                                'mean': np.nan,
                                'count': 0
                            } if self.with_count else {
                                'mean': np.nan
                            }
                        elif self.handle_fit_unknown == 'target_mean':
                            self.fold_mapping_[fold][i][category] = {
                                'mean': np.mean(y),
                                'count': 0
                            } if self.with_count else {
                                'mean': np.mean(y)
                            }
        return self

    def transform(self, X):

        check_is_fitted(self)

        if self.test_encoding == 'target_mean':
            return self._transform(X,
                                   handle_unknown=self.handle_transform_unknown,
                                   with_count=self.with_count)
        else:
            X_list, n_samples, n_features = self._check_X(X)

            X_encoded = np.zeros((n_samples, n_features), dtype=np.float)
            X_mask = np.ones((n_samples, n_features), dtype=np.bool)
            if self.with_count:
                X_count = np.zeros((n_samples, n_features), dtype=np.float)
            if n_features != len(self.categories_):
                raise ValueError(
                    "The number of features in X is different to the number of "
                    "features of the fitted data. The fitted data had {} features "
                    "and the X has {} features.".format(
                        len(self.categories_, ), n_features))

            for i in range(n_features):
                Xi = X_list[i]
                diff, valid_mask = _encode_check_unknown(Xi,
                                                         self.categories_[i],
                                                         return_mask=True)
                if not np.all(valid_mask):
                    if self.handle_transform_unknown == 'error':
                        msg = ("Found unknown categories {0} in column {1}"
                               " during transform".format(diff, i))
                        raise ValueError(msg)
                    else:
                        # Set the problematic rows to an acceptable value and
                        # continue `The rows are marked `X_mask` and will be
                        # removed later.
                        X_mask[:, i] = valid_mask
                        # cast Xi into the largest string type necessary
                        # to handle different lengths of numpy strings
                        if (self.categories_[i].dtype.kind in ('U', 'S') and
                                self.categories_[i].itemsize > Xi.itemsize):
                            Xi = Xi.astype(self.categories_[i].dtype)
                        else:
                            Xi = Xi.copy()

                # We use check_unknown=False, since _encode_check_unknown was
                # already called above.

                encoded, count = self._encode_category_via_random_fold(
                    i, Xi, valid_mask)
                X_encoded[:, i] = encoded
                if self.with_count:
                    X_count[:, i] = count
            out = np.hstack(
                (X_encoded, X_count)) if self.with_count else X_encoded
            return out

    def _encode_category_via_random_fold(self, feature, values, valid_mask):
        """
        Custom helper function to factorize and encode values
        """

        encoded = []
        count = []

        for idx, category in enumerate(values):
            # Track undrawn fold
            dic_fold = {str(i): i for i in self.fold_mapping_.keys()}
            if valid_mask[idx]:
                while True:
                    # Draw a random fold
                    rnd_fold = np.random.choice(list(dic_fold.values()))
                    # Check if it contains a value for the given category
                    if (self.fold_mapping_[rnd_fold][feature].get(category)):
                        encoded.append(self.fold_mapping_[rnd_fold][feature]
                                       [category]['mean'])
                        if self.with_count:
                            count.append(self.fold_mapping_[rnd_fold][feature]
                                         [category]['count'])
                        break
                    # If not remove this fold and draw another fold (that will contain the category)
                    else:
                        dic_fold.pop(str(rnd_fold))
            else:
                if self.handle_transform_unknown == 'target_mean':
                    encoded.append(self.target_mean_)
                    if self.with_count:
                        count.append(0)
                elif self.handle_transform_unknown == 'nan':
                    encoded.append(np.nan)
                    if self.with_count:
                        count.append(0)
        return encoded, count

    def fit_transform(self, X, y):
        """
        Fit to data, then transform it.
        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.
        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]
            Training set.
        y : numpy array of shape [n_samples]
            Target values.
        **fit_params : dict
            Additional fit parameters.
        Returns
        -------
        X_new : numpy array of shape [n_samples, n_features_new]
            Transformed array.
        """
        self.fit(X, y)
        y = self._check_Y(y)
        X_list, n_samples, n_features = self._check_X(X)
        X_encoded = np.zeros((n_samples, n_features), dtype=np.float)
        if self.with_count:
            X_count = np.zeros((n_samples, n_features), dtype=np.float)
        for fold in self.fold_mapping_.keys():
            fold_idx = self.fold_mapping_[fold]['idx'] if self.shuffle else [
                i for i in range(self.fold_mapping_[fold]['idx'][0],
                                 self.fold_mapping_[fold]['idx'][1] + 1)
            ]
            for feature in range(n_features):
                encoded_feature = np.zeros(len(fold_idx))
                if self.with_count:
                    count_feature = np.zeros(len(fold_idx))
                for category in self.categories_[feature]:
                    if (self.fold_mapping_[fold][feature].get(category)):
                        encoded_feature[X_list[feature][fold_idx] ==
                                        category] = self.fold_mapping_[fold][
                                            feature][category]['mean']
                        if self.with_count:
                            count_feature[X_list[feature][fold_idx] ==
                                          category] = self.fold_mapping_[fold][
                                feature][category]['count']
                            # count_feature = self.fold_mapping_[fold][feature][
                            #     category]['count']

                X_encoded[fold_idx, feature] = encoded_feature
                if self.with_count:
                    X_count[fold_idx, feature] = count_feature
            out = np.hstack(
                (X_encoded, X_count)) if self.with_count else X_encoded
        return out
