"""
    Module including some implementations of matrix factorization
    Date: 28/Mar/2019
    Author: Li Tang
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import random
import time
import pickle
import json
from multiprocessing import Process, Queue

__author__ = ['Li Tang']
__copyright__ = 'Li Tang'
__credits__ = ['Li Tang']
__license__ = 'MIT'
__version__ = '0.1.6'
__maintainer__ = ['Li Tang']
__email__ = 'litang1025@gmail.com'
__status__ = 'Production'


class FunkSVD:
    def __init__(self, matrix, k=1, name='FunkSVD', version=time.strftime("%Y%m%d", time.localtime())):
        """

        :param matrix:
        :param k:
        """
        assert matrix is not None, "'matrix' cannot be None."
        assert k > 0 and isinstance(k, int), "'k' should be an integer greater than 0."

        matrix = np.array(matrix, dtype=np.float64)
        self.matrix = (matrix - np.nanmin(matrix)) / (np.nanmax(matrix) - np.nanmin(matrix))
        self.k = k
        self.matrix_p = np.random.rand(len(self.matrix), self.k)
        self.matrix_q = np.random.rand(self.k, len(self.matrix[0]))
        self.name = name
        self.version = version

    def train(self, penalty='ridge', penalty_weight=0.5, learning_rate=0.75, learning_rate_decay=0.95,
              min_learning_rate=None, dropout=0.0, epochs=50, early_stopping=10, matrix_p=None, matrix_q=None,
              workers=1):
        """

        :param penalty:
        :param penalty_weight:
        :param learning_rate:
        :param learning_rate_decay:
        :param min_learning_rate:
        :param dropout:
        :param epochs:
        :param early_stopping:
        :param matrix_p:
        :param matrix_q:
        :param workers:
        :return:
        """
        assert penalty in ['ridge', 'lasso'], "'penalty' should be either 'ridge' or 'lasso'."
        assert penalty_weight > 0, "'penalty_weight' should be greater than 0."
        assert learning_rate > 0, "'learning_rate' should be greater than 0."
        assert learning_rate_decay > 0, "'learning_rate_decay' should be greater than 0. Set 1 for no decay in training."
        assert 0 <= dropout < 1.0, "The domain of definition of 'dropout' should be [0, 1)."
        assert isinstance(early_stopping, int), \
            "'early_stopping' should be an integer. Set 0 or any negative integer to interdict early stopping."
        assert isinstance(workers, int) and workers > 0, "'workers' should be an integer greater than 0."

        if matrix_p is not None:
            assert len(matrix_p) == len(self.matrix) and len(
                matrix_p[0]) == self.k, "The size of 'matrix_p' should be len(matrix) * k."
            self.matrix_p = matrix_p

        if matrix_q is not None:
            assert len(matrix_q) == self.k and len(matrix_q[0]) == len(
                self.matrix[0]), "The size of '_matrix_p' should be len(matrix) * k."
            self.matrix_q = matrix_q

        learning_rate /= 1 - dropout

        loss_history = [sys.maxsize], "Function?: max integer?"
        for epoch in range(epochs):
            loss = 0
            trained_samples = 0
            skipped_samples = 0
            matrix_rows_num = len(self.matrix) #The number of rows of the matrix
            matrix_cols_num = len(self.matrix[0]) #The number of columns of the matrix
            section_cols_num = matrix_cols_num / workers #The number of rows of each section
            section_rows_num = matrix_rows_num / workers #The number of columns of each section
            q = Queue()
            record = []

            if min_learning_rate:
                if learning_rate < min_learning_rate:
                    learning_rate = min_learning_rate

            # split matrix into multiple sections for concurrency

            #traverse the sections of the upper part of the matrix
            for section_col in range(workers):
                for section_row in range(section_col + 1):
                    #calculate the coordinate of upleft corner
                    section_upper = np.array([round(section_row * section_rows_num), round((section_col - section_row) * section_cols_num)])
                    # calculate the coordinate of downright corner
                    section_lower = np.array(
                        [round((section_row + 1) * section_rows_num) - 1, round((section_col - section_row + 1) * section_cols_num) - 1])
                    #create a process to run the method fit
                    p = Process(target=self.__fit, args=(q, (section_upper[0], section_lower[0]),
                                                         (section_upper[1], section_lower[1]), learning_rate, penalty,
                                                         penalty_weight, dropout,))
                    p.start()
                    record.append(p)
                #wait for the ending of all the processes
                for p in record:
                    p.join()
                record = []
                #get the result by Queue
                while(not q.empty()):
                    res = q.get()
                    loss += res[1]
                    trained_samples += res[2]
                    skipped_samples += res[3]


            for section_col in range(1, workers):
                for section_row in range(section_col, workers):
                    section_upper = np.array(
                        [round(section_row * section_rows_num), round((workers - section_row + section_col - 1) * section_cols_num)])
                    section_lower = np.array(
                        [round((section_row + 1) * section_rows_num) - 1, round((workers - section_row + section_col) * section_cols_num) - 1])
                    p = Process(target=self.__fit, args=(q, (section_upper[0], section_lower[0]),
                                                         (section_upper[1], section_lower[1]), learning_rate, penalty,
                                                         penalty_weight, dropout,))
                    p.start()
                    record.append(p)
                for p in record:
                    p.join()
                record = []
                while (not q.empty()):
                    res = q.get()
                    loss += res[1]
                    trained_samples += res[2]
                    skipped_samples += res[3]


            # start to train
            # loss, trained_samples, skipped_samples = self.__fit((0, len(self.matrix)), (0, len(self.matrix[0])),
            #                                                     learning_rate, penalty, penalty_weight, dropout)

            print('epoch: {} ==> loss: {}'.format(epoch + 1, loss))

            if dropout > 0:
                print('Trained {} samples and skipped {} samples.'
                      ' The dropout rate is: {}%'.format(trained_samples, skipped_samples, round(
                    skipped_samples / (trained_samples + skipped_samples) * 100)))

            if learning_rate_decay != 1.0:
                print('Current learning rate: {}'.format(learning_rate))
                learning_rate *= learning_rate_decay

            if early_stopping > 0:
                if loss < loss_history[0]:
                    loss_history = [loss]
                else:
                    loss_history.append(loss)

                if len(loss_history) >= early_stopping:
                    print(
                        'Early stopping! The best performance is at No.{} epoch and the loss have not been decreased from then on as {}:'.format(
                            epoch - early_stopping + 2, loss_history))
                    break
            else:
                continue

    def __fit(self,queue, row_purview: tuple, col_purview: tuple, learning_rate, penalty, penalty_weight, dropout):
        loss = 0
        trained_samples = 0
        skipped_samples = 0

        for row in range(row_purview[0], row_purview[1]):
            for col in range(col_purview[0], col_purview[1]):
                if self.matrix[row, col] is None or np.isnan(self.matrix[row, col]):
                    continue
                if random.random() <= 1 - dropout:
                    y_hat = np.matmul(self.matrix_p[row, :], self.matrix_q.T[col, :])

                    if penalty == 'ridge':
                        self.matrix_p[row, :] += learning_rate * ((self.matrix[row, col] - y_hat) *
                                                                  self.matrix_q[:, col] - penalty_weight *
                                                                  self.matrix_p[row, :]) / self.k

                        self.matrix_q[:, col] += learning_rate * ((self.matrix[row, col] - y_hat) *
                                                                  self.matrix_p[row, :] - penalty_weight *
                                                                  self.matrix_q[:, col]) / self.k

                        loss += ((self.matrix[row, col] - y_hat) ** 2 + penalty_weight * (
                                np.linalg.norm(self.matrix_p[row, :]) + np.linalg.norm(
                            self.matrix_q.T[col, :]))) / self.k
                    elif penalty == 'lasso':
                        self.matrix_p[row, :] += learning_rate * ((self.matrix[row, col] - y_hat) *
                                                                  self.matrix_q[:, col] - penalty_weight) / self.k
                        self.matrix_q[:, col] += learning_rate * ((self.matrix[row, col] - y_hat) *
                                                                  self.matrix_p[row, :] - penalty_weight) / self.k
                        loss += ((self.matrix[row, col] - y_hat) ** 2 + penalty_weight * (
                                np.linalg.norm(self.matrix_p[row, :], ord=1) + np.linalg.norm(
                            self.matrix_q.T[col, :], ord=1))) / self.k
                    else:
                        raise ValueError
                    trained_samples += 1
                else:
                    skipped_samples += 1
        res = [loss,trained_samples, skipped_samples]
        queue.put(res)


    def predict(self, topk=20, result_path=None):
        """

        :param topk:
        :return:
        """
        result_dict = {}
        for row in range(len(self.matrix)):
            topk_reco = []
            score_list = []
            for col in range(len(self.matrix[row])):
                if self.matrix[row, col] is None or np.isnan(self.matrix[row, col]):
                    score = np.matmul(self.matrix_p[row, :], self.matrix_q.T[col, :])
                    if len(topk_reco) < topk:
                        topk_reco.append(col)
                        score_list.append(score)
                    elif min(score_list) < score:
                        idx = score_list.index(min(score_list))
                        topk_reco[idx] = col
                        score_list[idx] = score
                    else:
                        continue
            result_dict[row] = topk_reco

        if result_path is not None:
            result_file = '{}{}_{}_{}_result.json'.format(result_path, self.name, str(self.k), self.version)
            try:
                with open(result_file, 'w') as result_output:
                    json.dump([result_dict], result_output)
                print('Result is saved into {}.'.format(result_file))
            except Exception as e:
                print('Failed to save result.')
                print(e)

        return result_dict

    def dump(self, model_file_path):
        model_file = '{}{}_{}_{}.pkl'.format(model_file_path, self.name, str(self.k), self.version)
        try:
            with open(model_file, 'wb') as model_output:
                model_info = dict()
                model_info['name'] = self.name
                model_info['version'] = self.version
                model_info['k'] = self.k
                model_info['matrix_p'] = self.matrix_p
                model_info['matrix_q'] = self.matrix_q
                pickle.dump(model_info, model_output)

            print('Model is saved into {}.'.format(model_file))

        except Exception as e:
            print('Failed to dump model.')
            print(e)

    def restore(self, model_file_path):
        try:
            with open(model_file_path, 'rb') as model_input:
                model_info = pickle.load(model_input)
                self.name = model_info['name']
                self.version = model_info['version']
                self.k = model_info['k']
                self.matrix_p = model_info['matrix_p']
                self.matrix_q = model_info['matrix_q']

            print('Model is loaded from {}.'.format(model_file_path))

        except Exception as e:
            print('Failed to restore model.')
            print(e)

    def add(self, target, value, initializer='mean'):
        assert target in ['k', 'matrix_p', 'matrix_q'], "'target' cannot be found."
        assert isinstance(value, int) and value > 0, "'value' should be an integer greater than 0."
        assert initializer in ['mean', 'random'], "'initializer' should be either 'mean' or 'random'."

