import numpy as np


def remove_min_region(index, x_vector, y_vector) -> tuple[np.array]:
    idx1 = idx2 = index
    while idx1 > 0 and y_vector[idx1 - 1] > y_vector[idx1]:
        idx1 -= 1
    while idx2 < len(y_vector) - 1 and y_vector[idx2 + 1] > y_vector[idx2]:
        idx2 += 1
    return (
        np.delete(x_vector, np.s_[idx1 : idx2 + 1]),
        np.delete(y_vector, np.s_[idx1 : idx2 + 1]),
    )


def get_minima(x_vector, y_vector) -> list[float]:
    x_vect = np.copy(x_vector)
    y_vect = np.copy(y_vector)
    minima = []
    while len(y_vect) > 0:
        idx = np.argmin(y_vect)
        minima.append((x_vect[idx], y_vect[idx]))
        x_vect, y_vect = remove_min_region(idx, x_vect, y_vect)
    return minima


if __name__ == "__main__":
    x_vec = np.array(list(range(50)))
    basis = np.array([3, 5, 4, 3, 2, 1, 2, 3, 4, 3])
    arr_list = [basis for i in range(1, 6)]
    y_vec = np.concatenate(arr_list)
    print(y_vec)
    mins = get_minima(x_vec, y_vec)
    print(mins)
