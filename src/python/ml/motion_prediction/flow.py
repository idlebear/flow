# implementing motion flow in a matrix -- complete with test cases!

from enum import IntEnum
from math import floor
import matplotlib.pyplot as plt
import numpy as np
from timeit import timeit

from numpy.core.shape_base import block


class FlowGrid:

    def __init__(self, initial_prob, score=None, motion=None, scale=1) -> None:
        '''
        Initialize the FlowGrid with a starting probability and motion vectors.  The
        motion and score parameters are applied over T time steps, with K instances for
        each one.  The score parameter allocates the strength of each vector set.

        Parameters:
        -----------
        scale: Scale of each element in the grid, used to determine motion distances

        score: Proportion of each motion vector to apply over the K vectors. The score can be
               constant over all time intervals or specified for each one.  Score has the
               shape:
                        (T, K, 1, x_dim, y_dim)  --or-- (T, 1, 1, x_dim, y_dim)

        motion: X and Y motion vector grids of the shape:

                         (T, K, 2, x_dim, y_dim)
        '''
        self.scale = scale
        self.pad = 1
        self.prob = np.pad(initial_prob, self.pad, mode='constant')
        pad = ((0,), (0,), (0, ), (self.pad, ), (self.pad, ))

        self.motion = np.pad(motion, pad, mode='constant', constant_values=0) if motion is not None else None
        self.score = np.pad(score, pad, mode='constant', constant_values=0) if score is not None else None

    def flow(self, mode='bilinear', output='single', dt=0.1):
        '''
        flow all of the elements of the grid according to their velocities

        The probability of a cell being flowed into is 1 - PI[ 1 - F(i->j)] where
        F(i->j) is the probability of cell i moving into j
        '''
        flow_prob = np.expand_dims(np.array(self.prob), 0)

        if mode == 'nearest':
            def map_fn(mo, int_mo): return ((int_mo - mo) > 0.5).astype(float)
        else:  # 'bilinear'
            def map_fn(mo, int_mo): return (int_mo - mo)

        # Taking the max of the current and the previous can lead to a more obvious flow, but
        # it also means that the occupancy stretches from the start at the last time step
        # instead of being an expanding wave.
        #
        # Being that we can't be sure where in that cloud the car really is, maybe that's
        # the right way to do it?  It's also much noisier, resulting in a larger cloud
        # of uncertainty
        if output == 'cummulative':
            def output_fn(prev, cur): return np.maximum(prev, cur)
        else:
            def output_fn(prev, cur): return cur

        if self.motion is None:
            raise ValueError('No MOTION vectors supplied')

        (T, K, L, W, H) = self.motion.shape

        last_prob = self.prob
        for t in range(T):
            next_prob = np.ones_like(last_prob)

            for k in range(K):

                if self.score is not None and K > 1:
                    if self.score.shape[1] == 1:
                        prob = last_prob * self.score[t, 0, 0, :, :]
                    else:
                        prob = last_prob * self.score[t, k, 0, :, :]
                else:
                    prob = last_prob

                # TODO: Reversed X and Y here in flow because the data from Carla has the X, Y coordinates flipped
                yy, xx = np.meshgrid(np.arange(H), np.arange(W))
                nx = xx + self.motion[t, k, 0, :, :] * dt / self.scale
                ny = yy + self.motion[t, k, 1, :, :] * dt / self.scale

                x1 = np.floor(nx)
                y1 = np.floor(ny)
                x2 = x1 + 1.
                y2 = y1 + 1.

                x1prop = map_fn(nx, x2)
                y1prop = map_fn(ny, y2)

                x2prop = 1. - x1prop
                y2prop = 1. - y1prop

                # after proportioning the blame, clip the indices so they
                # still fit in the grid -- this means our zero rows are probably
                # garbage
                # TODO: Add a padding row around the outside to absorb the
                #       out-of-bounds moves
                x1 = np.clip(x1, 0, W-1).astype(int)
                x2 = np.clip(x2, 0, W-1).astype(int)
                y1 = np.clip(y1, 0, H-1).astype(int)
                y2 = np.clip(y2, 0, H-1).astype(int)

                k_prob = np.ones_like(last_prob)
                k_prob[x1, y1] *= (1 - prob * x1prop * y1prop)
                k_prob[x1, y2] *= (1 - prob * x1prop * y2prop)
                k_prob[x2, y1] *= (1 - prob * x2prop * y1prop)
                k_prob[x2, y2] *= (1 - prob * x2prop * y2prop)

                # add the cummulative prob for this K to the prob of not flowing
                next_prob *= k_prob

            # reverse the probabilistic sense to get prob of flow for all 'j' cells and clip to
            # ensure we remain in bounds
            next_prob = np.clip(1 - next_prob, 0, 1)

            last_prob = output_fn(last_prob, next_prob)

            flow_prob = np.vstack([flow_prob, np.expand_dims(last_prob, 0)])

        # return the final result, dropping the padding
        return flow_prob[:, self.pad:-1, self.pad:-1]

    def draw(self, dt):

        flow = self.flow(mode='bilinear', dt=dt)

        N, H, W = flow.shape
        for i in range(1, N):
            flow[0, :, :] += flow[i, :, :]

        flow = np.clip(flow[0, :, :], 0, 1)

        figure = plt.figure(figsize=[10, 10])
        plt.imshow(flow)
        plt.show()


def main():
    T = 5
    size = 10
    prob = np.zeros((size, size))
    prob[2:5, 2:4] = 1
    prob[7:9, 7:9] = 1

    score = np.ones((T, 2, 1, size, size))*0.5
    motion = np.zeros((T, 2, 2, size, size))
    motion[:, 0, 0, ...] = 0.5
    motion[:, 1, 1, ...] = -0.75

    fg = FlowGrid(prob, score=score, motion=motion)
    fgg = fg.flow(mode='bilinear', dt=1)

    fgg_nn = fg.flow(mode='nearest', dt=1)

    fig, ax = plt.subplots(2, T+1)
    for i in range(T+1):
        ax[0, i].imshow(fgg[i, :, :])
        ax[1, i].imshow(fgg_nn[i, :, :])

    plt.show(block=True)


if __name__ == '__main__':
    main()
