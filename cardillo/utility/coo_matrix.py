import warnings
from scipy.sparse import csc_array, csr_array, coo_array
from scipy.sparse._sputils import isshape, check_shape
from scipy.sparse import spmatrix, sparray
import numpy as np
from numpy import tile, atleast_1d, arange, ndarray
from jax import numpy as jnp


class CooMatrix:
    """Small container storing the sparse matrix shape and three lists for
    accumulating the entries for row, column and data Wiki/COO.

    Parameters
    ----------
    shape : tuple, 2D
        tuple defining the shape of the matrix

    References
    ----------
    Wiki/COO: https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO)
    """

    def __init__(self, shape, manual_sync=False):
        self._buffered = manual_sync
        self.shape = shape
        # check shape input
        # if isinstance(shape, tuple):
        #     pass
        # else:
        #     try:
        #         shape = tuple(shape)
        #     except Exception:
        #         raise ValueError(
        #             "input argument shape is not tuple or cannot be interpreted as tuple"
        #         )

        # # see https://github.com/scipy/scipy/blob/adc4f4f7bab120ccfab9383aba272954a0a12fb0/scipy/sparse/sputils.py#L210
        # if isshape(shape, nonneg=True):
        #     M, N = shape
        #     # see https://github.com/scipy/scipy/blob/adc4f4f7bab120ccfab9383aba272954a0a12fb0/scipy/sparse/sputils.py#L267
        #     self.shape = check_shape((M, N))
        # else:
        #     raise TypeError(
        #         "input argument shape cannot be interpreted as correct shape"
        #     )

        # numpy array as efficient container for numerical data
        self.data = np.empty(0, dtype=float)  # double
        self.row = np.empty(0, dtype=int)  # unsigned int
        self.col = np.empty(0, dtype=int)  # unsigned int

        self._allocation_index = {}
        self._allocation_type = {}
        self._data_buffer = []

    @property
    def not_empty(self):
        return self.data.shape[0] > 0

    def __setitem__(self, key, value):
        # None is returned by every function that does not return. Hence, we
        # can use this to add no contribution to the matrix.
        if value is None:
            return

        if self._buffered:
            self._data_buffer.append((key, value))
            return

        # unpack key
        if len(key) == 4:
            # extract rows and columns to assign
            identifier, rows, cols, reverse = key
        elif len(key) == 3:
            # extract rows and columns to assign
            identifier, rows, cols = key
            reverse = False
        elif len(key) == 2:
            # extract rows and columns to assign
            rows, cols = key
            identifier = None
            reverse = False
        else:
            raise NotImplementedError

        # check allocation
        try:
            self._allocation_index[identifier]
            allocated = True
        except KeyError:
            allocated = False

        # determine value type
        if allocated:
            value_type = self._allocation_type[identifier]
        else:
            if isinstance(value, CooMatrix):
                value_type = "Coo"
            elif isinstance(value, sparray):
                value_type = "sparse"
                coo = value.tocoo()
            elif isinstance(value, spmatrix):
                raise RuntimeError("Do not use sparse matrices, move to sparse array.")
            elif isinstance(value, (ndarray, jnp.ndarray)):
                value_type = "ndarray"
            elif isinstance(value, (int, float)):
                value_type = "digit"
            else:
                raise NotImplementedError
            if identifier is not None:
                self._allocation_type[identifier] = value_type

        # convert value to array
        if value_type == "Coo":
            new_data = value.data
        elif value_type == "sparse":
            try:
                new_data = coo.data
            except UnboundLocalError:
                coo = value.tocoo()
                new_data = coo.data
        elif value_type == "ndarray":
            new_data = value.ravel(order="C")
        elif value_type == "digit":
            new_data = np.array([value])
        else:
            raise NotImplementedError

        # write data
        if allocated:
            id0, id1 = self._allocation_index[identifier]
            if reverse:
                new_data = -new_data
            self.data[id0:id1] = new_data
        else:
            # rows and cols
            if isinstance(rows, slice):
                rows = arange(*rows.indices(self.shape[0]))
            if isinstance(cols, slice):
                cols = arange(*cols.indices(self.shape[1]))
            rows = atleast_1d(rows)
            cols = atleast_1d(cols)
            if value_type == "Coo":
                new_rows = rows[value.row]
                new_cols = cols[value.col]
            elif value_type == "sparse":
                new_rows = rows[coo.row]
                new_cols = cols[coo.col]
            elif value_type == "ndarray":
                new_rows = rows.repeat(len(cols))
                new_cols = tile(cols, len(rows))
            elif value_type == "digit":
                new_rows = rows
                new_cols = cols
            else:
                raise NotImplementedError

            # extend rows and cols
            self.row = np.concatenate([self.row, new_rows])
            self.col = np.concatenate([self.col, new_cols])
            if reverse:
                new_data = -new_data
            self.data = np.concatenate([self.data, new_data])
            id1 = len(self.data)
            id0 = id1 - len(new_data)
            if identifier is not None:
                self._allocation_index[identifier] = (id0, id1)

    def manual_sync(self):
        self._buffered = False
        for key, value in self._data_buffer:
            if isinstance(value, CooMatrix) and value._buffered:
                value._buffered = False
                value.manual_sync()
                value._buffered = True
            self[key] = value
        self._buffered = True
        self._data_buffer = []

    def extend(self, matrix, DOF):
        warnings.warn(
            "Usage of `CooMatrix.extend` is deprecated. "
            "You can simply index the object, e.g., coo[rows, cols] = value",
            category=DeprecationWarning,
        )
        self[DOF[0], DOF[1]] = matrix

    def asformat(self, format, copy=False, fix_size=False):
        """Return this matrix in the passed format.
        Parameters
        ----------
        format : {str, None}
            The desired matrix format ("csr", "csc", "lil", "dok", "array", ...)
            or None for no conversion.
        copy : bool, optional
            If True, the result is guaranteed to not share data with self.
        Returns
        -------
        A : This matrix in the passed format.
        """
        if format == "Coo":
            return self
        try:
            convert_method = getattr(self, "to" + format)
        except AttributeError as e:
            raise ValueError("Format {} is unknown.".format(format)) from e

        # Forward the copy kwarg, if it's accepted.
        try:
            return convert_method(copy=copy, fix_size=fix_size)
        except TypeError:
            return convert_method()

    def __tosparse(self, scipy_matrix, copy=False):
        """Convert container to scipy sparse matrix.

        Parameters
        ----------
        scipy_matrix: scipy.sparse.spmatrix
            scipy sparse matrix format that should be returned
        """
        return scipy_matrix(
            (self.data, (self.row, self.col)), shape=self.shape, copy=copy
        )

    def tocoo(self, copy=False, fix_size=False):
        """Convert container to scipy coo_array."""
        if fix_size:
            try:
                coo = self._coo_cached
                if copy:
                    coo.data = self.data.copy()
                else:
                    coo.data = self.data
            except AttributeError:
                coo = self._coo_cached = self.__tosparse(coo_array, copy=False)
        else:
            coo = self.__tosparse(coo_array, copy=copy)
        return coo

    def tocsc(self, copy=False, fix_size=False):
        """Convert container to scipy csc_array."""
        if fix_size:
            try:
                csc = self._csc_cached
                try:
                    inverse = self.__csc_inverse
                except AttributeError:
                    nrow = self.shape[0]
                    index = self.col * nrow + self.row
                    _, inverse = np.unique(index, return_inverse=True)
                    self.__csc_inverse = inverse
                csc.data = np.bincount(inverse, weights=self.data)
            except AttributeError:
                csc = self._csc_cached = self.__tosparse(csc_array, copy=False)
        else:
            csc = self.__tosparse(csc_array, copy=copy)
        return csc

    def tocsr(self, copy=False, fix_size=False):
        """Convert container to scipy csr_array."""
        if fix_size:
            try:
                csr = self._csr_cached
                try:
                    inverse = self.__csr_inverse
                except AttributeError:
                    ncol = self.shape[1]
                    index = self.row * ncol + self.col
                    _, inverse = np.unique(index, return_inverse=True)
                    self.__csr_inverse = inverse
                csr.data = np.bincount(inverse, weights=self.data)
            except AttributeError:
                csr = self._csr_cached = self.__tosparse(csr_array, copy=False)
        else:
            csr = self.__tosparse(csr_array, copy=copy)
        return csr

    def toarray(self, copy=False, fix_size=False):
        """Convert container to 2D numpy array."""
        return self.tocoo(copy, fix_size=fix_size).toarray()

    def transpose(self, copy=False, coo=None):
        if coo is None:
            ret = CooMatrix((self.shape[1], self.shape[0]))
        else:
            ret = coo
        if copy:
            ret.row = self.col.copy()
            ret.col = self.row.copy()
            ret.data = self.data.copy()
        else:
            ret.row = self.col
            ret.col = self.row
            ret.data = self.data
        return ret

    @property
    def T(self):
        return self.transpose(copy=False)

    def __neg__(self):
        ret = CooMatrix(self.shape)
        ret.row = self.row
        ret.col = self.col
        ret.data = -self.data
        return ret

    def __add__(self, other):
        ret = CooMatrix(self.shape)
        if isinstance(other, CooMatrix):
            ret.data = np.concatenate([self.data, other.data])
            ret.col = np.concatenate([self.col, other.col])
            ret.row = np.concatenate([self.row, other.row])
            return ret
        else:
            return NotImplementedError

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, CooMatrix):
            ret = CooMatrix(self.shape)
            ret.data = np.concatenate([self.data, -other.data])
            ret.col = np.concatenate([self.col, other.col])
            ret.row = np.concatenate([self.row, other.row])
            return ret
        else:
            return NotImplementedError

    def __rsub__(self, other):
        if isinstance(other, CooMatrix):
            ret = CooMatrix(self.shape)
            ret.data = np.concatenate([-self.data, other.data])
            ret.col = np.concatenate([self.col, other.col])
            ret.row = np.concatenate([self.row, other.row])
            return ret
        else:
            return NotImplementedError

    def __mul__(self, other):
        ret = CooMatrix(self.shape)
        ret.row = self.row.copy()
        ret.col = self.col.copy()
        if isinstance(other, (int, float)):
            ret.data = self.data * other
        else:
            return NotImplementedError
        return ret

    def __rmul__(self, other):
        return self.__mul__(other)


class CooArray:
    """
    one dimensional version of CooMatrix
    """

    def __init__(self, length, manual_sync=False):
        self.length = length
        self._buffered = manual_sync
        # check shape input
        if isinstance(length, int):
            pass
        else:
            try:
                length = int(length)
            except Exception:
                raise ValueError(
                    "input argument shape is not int or cannot be interpreted as int"
                )
        # numpy array as efficient container for numerical data
        self.data = np.empty(0, dtype=float)  # double
        self.col = np.empty(0, dtype=int)  # unsigned int

        self._allocation_index = {}
        self._allocation_type = {}
        self._data_buffer = []

    def __setitem__(self, key, value):
        if value is None:
            return

        if self._buffered:
            self._data_buffer.append((key, value))
            return

        # unpack key
        if len(key) == 3:
            identifier, cols, reverse = key
        elif len(key) == 2:
            identifier, cols = key
            reverse = False
        elif isinstance(key, np.ndarray):
            cols = key
            identifier = None
            reverse = False
        else:
            raise NotImplementedError

        # check allocation
        try:
            self._allocation_index[identifier]
            allocated = True
        except KeyError:
            allocated = False

        # determine value type
        if allocated:
            value_type = self._allocation_type[identifier]
        else:
            if isinstance(value, CooArray):
                value_type = "Coo"
            elif isinstance(value, (ndarray, jnp.ndarray)):
                value_type = "ndarray"
            elif isinstance(value, (int, float)):
                value_type = "digit"
            else:
                raise NotImplementedError
            if identifier is not None:
                self._allocation_type[identifier] = value_type

        # convert value to array
        if value_type == "Coo":
            new_data = value.data
        elif value_type == "ndarray":
            new_data = value
        elif value_type == "digit":
            new_data = np.array([value])
        else:
            raise NotImplementedError

        # write data
        if allocated:
            id0, id1 = self._allocation_index[identifier]
            if reverse:
                new_data = -new_data
            self.data[id0:id1] = new_data
        else:
            if isinstance(cols, slice):
                cols = arange(*cols.indices(self.length))
            cols = atleast_1d(cols)
            if value_type == "Coo":
                new_cols = cols[value.col]
            elif value_type == "ndarray":
                new_cols = cols
            elif value_type == "digit":
                new_cols = cols
            else:
                raise NotImplementedError

            # extend cols
            self.col = np.concatenate([self.col, new_cols])

            if reverse:
                new_data = -new_data
            self.data = np.concatenate([self.data, new_data])
            id1 = len(self.data)
            id0 = id1 - len(new_data)
            if identifier is not None:
                self._allocation_index[identifier] = (id0, id1)

    def manual_sync(self):
        self._buffered = False
        for key, value in self._data_buffer:
            if isinstance(value, CooArray) and value._buffered:
                value._buffered = False
                value.manual_sync()
            self[key] = value
        self._data_buffer = []

    def tocoo(self, copy=False, fix_size=False):
        """Convert container to scipy coo_array."""
        if fix_size:
            try:
                coo = self._coo_cached
                if copy:
                    coo.data = self.data.copy()
                else:
                    coo.data = self.data
            except AttributeError:
                coo = self._coo_cached = coo_array(
                    (self.data, (self.col,)), shape=(self.length,), copy=False
                )
        else:
            coo = coo_array((self.data, (self.col,)), shape=(self.length,), copy=copy)
        return coo

    def toarray(self, copy=False, fix_size=False):
        """Convert container to 1D numpy array."""
        return self.tocoo(copy, fix_size=fix_size).toarray()
